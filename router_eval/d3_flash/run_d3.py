"""d3-flash (vllm-sr/d3-flash, Decision 3.0: a 9B Qwen3.5 decision model) on our 300-question tier set.

Asks exactly what Jev and CLM were asked (jev/run_jev.py), with the same state {"request": question}:
  choice : LEVEL_QUESTION (high / mid / low)       -> pick, probabilities, confidence
  yesno  : YESNO_QUESTION (P(yes) per tier, noul)  -> p_yes
Results go to router_eval/d3_flash/results/; router_eval/retrain3/threshold_sweeps.py turns them into
the sweep and reference rows of TIER_SELECTION_RESULTS.md. `features` + `finetune` train its readout on the 600 set.

Runs the model's own code (trust_remote_code) with transformers==5.17.0, on the Mac GPU (mps) by
default; the ROCm/Triton fast kernels aren't available there, so transformers' PyTorch path is used.

    D3_PATH=<local snapshot> python router_eval/d3_flash/run_d3.py check     # model card example
    D3_PATH=<local snapshot> python router_eval/d3_flash/run_d3.py zeroshot  # 300 set
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "jev"))
sys.path.insert(0, str(ROOT))
from run_jev import LEVEL_QUESTION, YESNO_QUESTION  # noqa: E402

RESULTS = HERE / "results"


def load_model():
    import torch
    from transformers import AutoModel

    t0 = time.time()
    model = AutoModel.from_pretrained(os.environ.get("D3_PATH", "vllm-sr/d3-flash"), trust_remote_code=True,
                                      device=os.environ.get("D3_DEVICE", "mps"),
                                      batch_size=int(os.environ.get("D3_BATCH", "2")))
    print(f"loaded in {time.time() - t0:.0f} s on {model.runtime.device}", flush=True)
    if model.runtime.device.type == "mps":
        torch.mps.empty_cache()
    return model


def check():
    m = load_model()
    r = m.system_one(state="The order arrived damaged yesterday. I want my money back and I still have the receipt.",
                     questions={"route": {"type": "choice", "instructions": "Which team should handle this request?",
                                          "criteria": {"returns": "Returns and refunds", "billing": "Charges and invoices",
                                                       "technical": "Bugs and outages"}},
                                "receipt": {"type": "noul", "instructions": "Does the customer have a receipt?"},
                                "urgency": {"type": "score", "instructions": "How urgent is this request?",
                                            "criteria": ["Routine", "Soon", "Today"]}})
    print(json.dumps(r["answers"], indent=1)[:1500])


def noul(answer):
    """P(yes) from a noul answer, whichever key the runtime uses."""
    if "noul" in answer:
        return float(answer["noul"])
    return float(answer["probabilities"]["true"])


def zeroshot():
    import torch
    m = load_model()
    ex = json.loads((ROOT / "router_eval" / "data" / "own_300.json").read_text())["examples"]
    RESULTS.mkdir(parents=True, exist_ok=True)
    for setup, q in (("choice", LEVEL_QUESTION), ("yesno", YESNO_QUESTION)):
        m.system_one(state={"request": "warm up"}, questions=q)
        with (RESULTS / f"decisions_d3-{setup}_own_300.jsonl").open("w") as f:
            for k, x in enumerate(ex):
                t0 = time.perf_counter()
                a = m.system_one(state={"request": x["query"]}, questions=q)["answers"]
                ms = (time.perf_counter() - t0) * 1000
                row = {"qid": x["id"], "label": x["label"], "latency_ms": ms}
                if setup == "choice":
                    row.update(choice=a["tier"]["choice"], confidence=a["tier"]["confidence"], probabilities=a["tier"]["probabilities"])
                else:
                    row.update(p_yes={t: noul(a[t]) for t in YESNO_QUESTION})
                f.write(json.dumps(row) + "\n")
                if m.runtime.device.type == "mps" and k % 25 == 0:
                    torch.mps.empty_cache()  # keep PyTorch's GPU cache from pushing the 17 GB model into swap
                    print(f"  {setup} {k}/{len(ex)}", flush=True)
        print(f"zeroshot {setup}: done", flush=True)


FEATURES = Path(os.environ.get("D3_FEATURES", "/tmp/d3_features.npz"))


def capture_hidden(model):
    """Make the runtime keep the last-token hidden state it scores with the readout (its own code path)."""
    import types

    rt, kept = model.runtime, []

    def logits(self, sequences, counts):  # the runtime's logits(), keeping `hidden` on the way
        torch = self.torch
        width = max(len(s) for s in sequences)
        ids = torch.full((len(sequences), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(sequences), width), dtype=torch.long)
        for i, s in enumerate(sequences):
            ids[i, width - len(s):] = torch.as_tensor(s, dtype=torch.long)
            mask[i, width - len(s):] = 1
        ids, mask = ids.to(self.device), mask.to(self.device)
        with torch.inference_mode():
            if self.encoder is not None:  # noncausal_full_attention: this checkpoint's mode
                hidden = self.encoder(self.backbone.get_input_embeddings()(ids), mask)[:, -1]
            else:
                hidden = self.backbone(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state[:, -1]
            logits = hidden.float() @ self.readout.T  # readout_dtype is float32 for this checkpoint
        kept.append(hidden.float().cpu())
        limit = torch.as_tensor(list(counts), device=self.device)[:, None]
        invalid = torch.arange(self.readout.shape[0], device=self.device)[None] >= limit
        return logits.masked_fill(invalid, float("-inf"))

    rt.logits = types.MethodType(logits, rt)
    return kept


def features():
    """Hidden states for the tier question on the 600 and 300 sets, plus the readout, saved to FEATURES."""
    import numpy as np
    import torch
    m = load_model()
    kept = capture_hidden(m)
    out = {}
    for name in ("eval_600", "own_300"):
        ex = json.loads((ROOT / "router_eval" / "data" / f"{name}.json").read_text())["examples"]
        rows, probs = [], []
        for k, x in enumerate(ex):
            kept.clear()
            a = m.system_one(state={"request": x["query"]}, questions=LEVEL_QUESTION)["answers"]["tier"]
            rows.append(kept[-1][0].numpy())
            probs.append([a["probabilities"][t] for t in LEVEL_QUESTION["tier"]["criteria"]])
            if m.runtime.device.type == "mps" and k % 25 == 0:
                torch.mps.empty_cache()
                print(f"  {name} {k}/{len(ex)}", flush=True)
        out[f"{name}_x"], out[f"{name}_p"] = np.stack(rows), np.array(probs)
        out[f"{name}_q"] = np.array([x["query"] for x in ex])
    out["readout"] = m.runtime.readout[:3].float().cpu().numpy()  # rows of codes A, B, C = high, mid, low
    out["keys"] = np.array(list(LEVEL_QUESTION["tier"]["criteria"]))
    np.savez(FEATURES, **out)
    # The captured hidden states must reproduce the model's own probabilities through the readout.
    z = out["own_300_x"] @ out["readout"].T / m.runtime.temperature
    p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    print(f"saved {FEATURES}; max |prob difference| vs the model's own answers: {np.abs(p - out['own_300_p']).max():.2e}")


class D3Readout:
    """d3-flash's readout rows for the three tier codes, fine-tuned on frozen backbone features."""
    data = None

    def __init__(self, epochs=20, batch=256, lr=5e-4, val_frac=0.1, patience=5, seed=1234):
        import numpy as np
        self.epochs, self.batch, self.lr, self.val_frac, self.patience, self.seed = epochs, batch, lr, val_frac, patience, seed
        if D3Readout.data is None:
            d = np.load(FEATURES)
            D3Readout.data = {"x": {q: v for n in ("eval_600", "own_300") for q, v in zip(d[f"{n}_q"], d[f"{n}_x"])},
                              "w": d["readout"], "keys": list(d["keys"])}

    def _x(self, queries):
        import numpy as np
        import torch
        return torch.tensor(np.stack([self.data["x"][q] for q in queries]))

    def fit(self, queries, contexts, labels):
        import numpy as np
        import torch
        import torch.nn.functional as F
        torch.manual_seed(self.seed)
        x, keys = self._x(queries), self.data["keys"]
        y = torch.tensor([keys.index(t) for t in labels])
        w = torch.nn.Parameter(torch.tensor(self.data["w"]))
        opt = torch.optim.AdamW([w], lr=self.lr, weight_decay=0.0)
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(y))
        n_val = max(1, int(len(y) * self.val_frac))
        val, tr = idx[:n_val], idx[n_val:]
        best, best_w, bad = None, w.detach().clone(), 0
        for _ in range(self.epochs):
            order = tr[rng.permutation(len(tr))]
            for b in range(0, len(order), self.batch):
                rows = order[b:b + self.batch]
                loss = F.cross_entropy(x[rows] @ w.T, y[rows])
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                v = F.cross_entropy(x[val] @ w.T, y[val]).item()
            if best is None or v < best - 1e-4:
                best, best_w, bad = v, w.detach().clone(), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.w = best_w

    def predict(self, queries, contexts):
        import torch
        with torch.no_grad():
            pick = (self._x(queries) @ self.w.T).argmax(1).tolist()
        return [self.data["keys"][i] for i in pick]


def finetune():
    sys.path.insert(0, str(ROOT / "router_eval" / "retrain3"))
    from common import run
    run(name="d3-flash-finetuned", pr=None, setup="d3-flash (9B Qwen3.5 backbone frozen), readout rows for the 3 tier codes fine-tuned",
        make=D3Readout,
        notes="Warm start from d3-flash's own readout rows (codes A/B/C = high/mid/low of Jev's question); trained on "
              "frozen last-token features of the 600 set with CLM's recipe (AdamW 5e-4, batch 256, 20 epochs, 10% "
              "validation, patience 5); temperature 1 as in the checkpoint. Features captured on the Mac GPU without "
              "the ROCm kernels; p50 latency here is the readout only, the backbone takes ~0.8 s per question.")


if __name__ == "__main__":
    {"check": check, "zeroshot": zeroshot, "features": features, "finetune": finetune}[sys.argv[1]]()
