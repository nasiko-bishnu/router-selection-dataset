"""CLM (Contrastive-LM, Qwen3-8B encoder + projection heads) on our tier sets.

Needs the CLM package and an encoder at CLM_EMB_URL (default http://127.0.0.1:8090/v1/embeddings;
on a Mac, the PyTorch stand-in for vLLM). Run with the CLM environment's python:

    python router_eval/clm/run_clm.py check       # README examples, to verify the setup first
    python router_eval/clm/run_clm.py zeroshot    # Jev's exact questions on the 300 set -> results/
    python router_eval/clm/run_clm.py finetune    # 3-tier head fine-tune: train 600, test 300 (retrain harness)

zeroshot asks what Jev was asked in jev/run_jev.py (same state {"request": question}):
  choice : LEVEL_QUESTION (high / mid / low)        -> pick, probabilities, confidence
  yesno  : YESNO_QUESTION (P(yes) per tier, noul)   -> p_yes
finetune warm-starts both heads and the logit scale from the reference checkpoint and trains them
with a softmax over each question's three tier options (CLM's `--task choice --loss softce`,
hard targets), on embeddings of exactly the text CLM's engine builds (clm.schema.build_pairs).
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "jev"))
sys.path.insert(0, str(ROOT))
os.environ.setdefault("USE_TF", "0")
from run_jev import LEVEL_QUESTION, YESNO_QUESTION  # noqa: E402

from clm import Engine  # noqa: E402
from clm.embedder import Embedder  # noqa: E402
from clm.heads import HeadPair, default_checkpoint  # noqa: E402
from clm.schema import build_pairs  # noqa: E402

RESULTS = HERE / "results"
TIERS = ["low", "mid", "high"]
EMB_URL = os.environ.get("CLM_EMB_URL", "http://127.0.0.1:8090/v1/embeddings")


def engine():
    return Engine(emb_url=EMB_URL)


def check():
    e = engine()
    r = e.answer("Customer: my invoice was charged twice and nobody answers the phone!", {
        "urgency": {"type": "noul", "instructions": "Is this urgent?"},
        "department": {"type": "choice", "instructions": "Which team should handle this?",
                       "criteria": {"billing": "Charges, invoices, refunds", "technical": "Bugs and outages"}},
        "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                        "criteria": ["Calm", "Frustrated", "Very angry"]}})
    a = r["answers"]
    print("urgency noul      ", round(a["urgency"]["noul"], 5), "  README: 0.41022")
    print("department        ", a["department"]["choice"], {k: round(v, 5) for k, v in a["department"]["probabilities"].items()},
          "  README: billing {'billing': 0.93878, 'technical': 0.06122}")
    print("frustration score ", round(a["frustration"]["score"], 5), "  README: 1.98386")
    top = e.rank("What causes tides on Earth?", ["The Moon's gravitational pull.", "Photosynthesis in plants.",
                                                 "Because the Earth is round."])[0]
    print("rank top          ", top["candidate"], round(top["prob"], 3), "  README: The Moon's gravitational pull. 0.997")


def load(name):
    return json.loads((ROOT / "router_eval" / "data" / f"{name}.json").read_text())["examples"]


def zeroshot():
    e = engine()
    ex = load("own_300")
    RESULTS.mkdir(parents=True, exist_ok=True)
    for setup, q in (("choice", LEVEL_QUESTION), ("yesno", YESNO_QUESTION)):
        e.answer({"request": "warm up"}, q)
        with (RESULTS / f"decisions_clm-{setup}_own_300.jsonl").open("w") as f:
            for x in ex:
                t0 = time.perf_counter()
                a = e.answer({"request": x["query"]}, q)["answers"]
                ms = (time.perf_counter() - t0) * 1000
                row = {"qid": x["id"], "label": x["label"], "latency_ms": ms}
                if setup == "choice":
                    row.update(choice=a["tier"]["choice"], confidence=a["tier"]["confidence"], probabilities=a["tier"]["probabilities"])
                else:
                    row.update(p_yes={k: a[k]["noul"] for k in YESNO_QUESTION})
                f.write(json.dumps(row) + "\n")
        print(f"zeroshot {setup}: done", flush=True)


class ClmTierHead:
    """CLM's two projection heads, fine-tuned to pick one of the three tier options."""
    emb = None  # one Embedder (and its cache) shared by every fit: the encoder is frozen

    def __init__(self, epochs=20, batch=256, lr=5e-4, val_frac=0.1, patience=5, seed=1234):
        self.epochs, self.batch, self.lr, self.val_frac, self.patience, self.seed = epochs, batch, lr, val_frac, patience, seed
        if ClmTierHead.emb is None:
            ClmTierHead.emb = Embedder(url=EMB_URL)

    def _texts(self, queries):
        pairs = [build_pairs({"request": q}, LEVEL_QUESTION)["tier"] for q in queries]
        keys = pairs[0][1]  # option order from the question: high, mid, low
        return [p[0] for p in pairs], pairs[0][2], keys

    def fit(self, queries, contexts, labels):
        import torch
        import torch.nn.functional as F
        torch.manual_seed(self.seed)
        states, cands, self.keys = self._texts(queries)
        xs = torch.from_numpy(self.emb.embed(states)[0])
        xc = torch.from_numpy(self.emb.embed(cands)[0])
        y = torch.tensor([self.keys.index(t) for t in labels])
        ref = HeadPair("ref", default_checkpoint(), "cpu").ensure()
        self.sh, self.ah = ref.state_head.train(), ref.action_head.train()
        log_scale = torch.nn.Parameter(torch.tensor(float(np.log(ref.scale))))
        params = [*self.sh.parameters(), *self.ah.parameters(), log_scale]
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=0.0)
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(y))
        n_val = max(1, int(len(y) * self.val_frac))
        val, tr = idx[:n_val], idx[n_val:]

        def logits(rows):
            zs = F.normalize(self.sh(xs[rows]), dim=-1)
            zc = F.normalize(self.ah(xc), dim=-1)
            return log_scale.exp().clamp(max=100.0) * zs @ zc.T

        best, best_state, bad = None, None, 0
        for _ in range(self.epochs):
            for b in range(0, len(tr), self.batch):
                rows = tr[rng.permutation(len(tr))][: self.batch] if b == 0 else tr[b:b + self.batch]
                loss = F.cross_entropy(logits(rows), y[rows])
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                v = F.cross_entropy(logits(val), y[val]).item()
            if best is None or v < best - 1e-4:
                best, bad = v, 0
                best_state = ({k: t.clone() for k, t in self.sh.state_dict().items()},
                              {k: t.clone() for k, t in self.ah.state_dict().items()}, log_scale.detach().clone())
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.sh.load_state_dict(best_state[0]); self.ah.load_state_dict(best_state[1])
        self.scale = best_state[2].exp().clamp(max=100.0)
        self.sh.eval(); self.ah.eval()
        with torch.no_grad():
            self.zc = F.normalize(self.ah(xc), dim=-1)

    def predict(self, queries, contexts):
        import torch
        import torch.nn.functional as F
        states, _, _ = self._texts(queries)
        with torch.no_grad():
            zs = F.normalize(self.sh(torch.from_numpy(self.emb.embed(states)[0])), dim=-1)
            pick = (zs @ self.zc.T).argmax(1).tolist()
        return [self.keys[i] for i in pick]


def finetune():
    sys.path.insert(0, str(ROOT / "router_eval" / "retrain3"))
    from common import run
    run(name="clm-finetuned", pr=None, setup="CLM (Qwen3-8B frozen + projection heads), heads fine-tuned on the 3 tier options",
        make=ClmTierHead,
        notes="Warm start from CLM_v0.1-8B; CLM's choice/softce recipe (AdamW 5e-4, batch 256, 20 epochs, 10% validation, "
              "patience 5 on validation loss); Jev's high/mid/low question text. Encoder: Qwen3-8B bf16 on the Mac GPU "
              "via a PyTorch stand-in for vLLM (same last-token pooling).")


if __name__ == "__main__":
    {"check": check, "zeroshot": zeroshot, "finetune": finetune}[sys.argv[1]]()
