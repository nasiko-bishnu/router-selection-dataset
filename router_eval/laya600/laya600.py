# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas", "pyarrow", "numpy", "huggingface_hub", "laya", "torch", "safetensors", "requests"]
# ///
"""Fine-tune the three local Laya models on our 600-question set and test them once on the 300 set,
the same way the other models in TIER_SELECTION_RESULTS.md were trained.

Labels are used directly, as for every other model trained on the 600 set: Laya is asked the same
high / mid / low question Jev, CLM and d3-flash were asked (jev/run_jev.py LEVEL_QUESTION), so its
three options are low, mid and high themselves, trained with plain cross-entropy. The training
recipe is train_laya.py's (AdamW: head 1e-4, top 4 encoder layers 2e-5, weight decay 0.01, 6% warm-up
then linear decay, batch 16 in micro-batches of 4, gradient clipping 1.0). Like the CLM and d3-flash
heads, 10% of the 600 set is held out inside fit() to keep the epoch with the lowest validation loss.
Question text only, no context. The prediction is Laya's top option (argmax of its option scores).

    uv run router_eval/laya600/laya600.py check                     # this code path == Laya's own predict()
    uv run router_eval/laya600/laya600.py train english             # or multilingual / typed-decisions / all
"""
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "jev"))
sys.path.insert(0, str(HERE.parent / "retrain3"))
os.environ.setdefault("USE_TF", "0")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import laya  # noqa: E402
from bench import laya_items  # noqa: E402
from run_jev import LEVEL_QUESTION  # noqa: E402
from train_laya import BASES, REPO, all_logits, batch_logits, free_cache, set_trainable  # noqa: E402

OPTIONS = list(LEVEL_QUESTION["tier"]["criteria"])  # Laya's options, in order: high, mid, low = our labels
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class LayaTier:
    """A Laya base model fine-tuned on our tier labels."""

    def __init__(self, base="english", epochs=8, batch=16, micro_batch=4, lr_head=1e-4, lr_encoder=2e-5,
                 unfreeze_layers=4, val_frac=0.1, seed=0, eval_batch=32, train=True):
        self.base, self.epochs, self.batch, self.micro_batch = base, epochs, batch, micro_batch
        self.lr_head, self.lr_encoder, self.unfreeze_layers = lr_head, lr_encoder, unfreeze_layers
        self.val_frac, self.seed, self.eval_batch, self.train = val_frac, seed, eval_batch, train
        self.log = {}

    def _items(self, queries, labels=None):
        items = laya_items(self.tok, self.cfg, [q[:2000] for q in queries], LEVEL_QUESTION)
        for it, lab in zip(items, labels or [None] * len(items)):
            if lab is not None:
                it["label"] = OPTIONS.index(lab)
        return items

    def fit(self, queries, contexts, labels):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        agent = laya.load(REPO, device=str(DEVICE), subfolder=BASES[self.base])
        self.model, self.tok, self.cfg = agent.model, agent.tok, agent.cfg
        self.pad = self.tok.pad_token_id
        if not self.train:  # zero-shot: the base model as it is
            return
        items = self._items(queries, labels)
        idx = rng.permutation(len(items))
        n_val = max(1, int(len(items) * self.val_frac))
        va, tr = [items[i] for i in idx[:n_val]], [items[i] for i in idx[n_val:]]
        y_va = torch.tensor([it["label"] for it in va])

        n_trainable = set_trainable(self.model, self.unfreeze_layers)
        head = [p for n, p in self.model.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
        enc = [p for n, p in self.model.named_parameters() if p.requires_grad and n.startswith("encoder.")]
        opt = torch.optim.AdamW([{"params": head, "lr": self.lr_head}, {"params": enc, "lr": self.lr_encoder}], weight_decay=0.01)
        steps = self.epochs * ((len(tr) + self.batch - 1) // self.batch)
        warm = max(1, int(0.06 * steps))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))

        def val_loss():
            free_cache(DEVICE)
            return float(F.cross_entropy(all_logits(self.model, va, DEVICE, self.pad, self.eval_batch).float(), y_va))

        best = {"loss": val_loss(), "epoch": 0, "state": None}
        self.log = {"trainable_params": n_trainable, "n_train": len(tr), "n_val": len(va), "epochs": [{"epoch": 0, "val_loss": round(best["loss"], 4)}]}
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            t0, tot = time.time(), 0.0
            order = rng.permutation(len(tr))
            for i in range(0, len(order), self.batch):
                chunk = [tr[j] for j in order[i:i + self.batch]]
                y = torch.tensor([it["label"] for it in chunk], device=DEVICE)
                opt.zero_grad(set_to_none=True)
                for k in range(0, len(chunk), self.micro_batch):  # gradient accumulation: same update, less memory
                    sub = chunk[k:k + self.micro_batch]
                    logits = batch_logits(self.model, sub, DEVICE, self.pad, detach_encoder=self.unfreeze_layers == 0)
                    part = F.cross_entropy(logits.float(), y[k:k + len(sub)], reduction="sum") / len(chunk)
                    part.backward()
                    tot += part.item() * len(chunk)
                torch.nn.utils.clip_grad_norm_(head + enc, 1.0)
                opt.step(); sched.step()
            free_cache(DEVICE)
            v = val_loss()
            self.log["epochs"].append({"epoch": epoch, "train_loss": round(tot / len(tr), 4), "val_loss": round(v, 4),
                                       "seconds": round(time.time() - t0)})
            print(f"  {self.base} epoch {epoch}: train {tot / len(tr):.4f} | validation {v:.4f} | {time.time() - t0:.0f}s", flush=True)
            if v < best["loss"]:
                best = {"loss": v, "epoch": epoch,
                        "state": {n: p.detach().cpu().clone() for n, p in self.model.named_parameters() if p.requires_grad}}
        if best["state"] is not None:
            with torch.no_grad():
                params = dict(self.model.named_parameters())
                for n, t in best["state"].items():
                    params[n].copy_(t.to(DEVICE))
        self.log["best_epoch"] = best["epoch"]
        self.model.eval()
        free_cache(DEVICE)

    def predict(self, queries, contexts):
        logits = all_logits(self.model, self._items(queries), DEVICE, self.pad, self.eval_batch)
        return [OPTIONS[i] for i in logits.argmax(-1).tolist()]


def main():
    from common import load, run
    mode = sys.argv[1]
    if mode == "check":  # this code path must pick what Laya's own predict() picks
        tq, tc, ty, _ = load("own_300")
        for base in BASES:
            m = LayaTier(base, train=False)
            m.fit([], [], [])
            agent = laya.load(REPO, device=str(DEVICE), subfolder=BASES[base])
            ours = m.predict(tq, tc)
            theirs = [agent.predict({"request": q[:2000]}, LEVEL_QUESTION)["answers"]["tier"]["choice"] for q in tq[:60]]
            acc = 100 * np.mean([p == t for p, t in zip(ours, ty)])
            print(f"{base}: agrees with predict() on {sum(a == b for a, b in zip(ours, theirs))}/60 | "
                  f"zero-shot on the 300 set with the low/mid/high question: {acc:.1f}%", flush=True)
            del agent, m
            free_cache(DEVICE)
        return
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("mode"); ap.add_argument("base")
    ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--unfreeze", type=int, default=4)
    ap.add_argument("--lr-head", type=float, default=1e-4); ap.add_argument("--lr-encoder", type=float, default=2e-5)
    ap.add_argument("--tag", default="", help="suffix for the run name, e.g. -long")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    bases = list(BASES) if a.base == "all" else [a.base]
    for base in bases:
        holder = {}

        def make(base=base):
            holder["m"] = LayaTier(base, epochs=a.epochs, unfreeze_layers=a.unfreeze, lr_head=a.lr_head, lr_encoder=a.lr_encoder, seed=a.seed)
            return holder["m"]

        run(name=f"laya600-{base}{a.tag}", pr=None, make=make, cv=False,
            setup=f"Laya {base}, fine-tuned on the 600 set's labels (options low / mid / high)",
            notes=f"Asked Jev's high / mid / low question, so the options are our labels. train_laya.py's recipe: plain cross-entropy, AdamW head {a.lr_head:g} / top {a.unfreeze} encoder layers {a.lr_encoder:g}, "
                  f"batch 16, up to {a.epochs} epochs, best epoch by loss on 10% of the 600 set held out "
                  f"(chosen: {{best}}). Question text only. No CV on 600 (each fit is a full fine-tune).")
        best = holder["m"].log.get("best_epoch")
        # put the chosen epoch into the stored notes
        import json
        p = HERE.parent / "retrain3" / "results" / f"laya600-{base}{a.tag}.json"
        r = json.loads(p.read_text()); r["notes"] = r["notes"].replace("{best}", f"epoch {best}"); r["training_log"] = holder["m"].log
        p.write_text(json.dumps(r, indent=2) + "\n")
    from common import report
    report()


if __name__ == "__main__":
    main()
