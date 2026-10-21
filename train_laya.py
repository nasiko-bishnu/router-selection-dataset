# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas", "pyarrow", "numpy", "huggingface_hub", "laya", "torch", "safetensors", "requests"]
# ///
"""Fine-tune a Laya checkpoint as a tier picker, by gradient descent on labelled questions.

Input: the labelled files from make_tier_dataset.py. Each row is a question, the tier it needs
(the cheapest that reaches the best score) and each tier's score. For every batch Laya reads the
questions through the same T1/T2/T3 choice bench.py asks (TIER_QUESTION), the loss against the
target is backpropagated, and AdamW updates the weights. Two targets (--target):

  tier      the one ideal tier: cross-entropy over Laya's softmax across the three options.
  per-tier  whether each tier gets the question right (score >= RIGHT): each option's raw score
            is the log-odds for that tier, trained with binary cross-entropy, so the tiers no
            longer compete. A validation cut-off then routes to the cheapest tier whose chance is
            above it (T1 when none is); bench.py reads that cut-off from the checkpoint.

The validation file picks the best epoch (by vs_blind_avg, the routing gain over a blind split
averaged across budgets; --select nll keeps the lowest-loss epoch as runs 1-2 did) and fits the
output temperature (tier) or the cut-off (per-tier). The test split is not read here; score the
saved checkpoint with bench.py like every other selector:

    uv run make_tier_dataset.py                                # data/tier_train.jsonl, data/tier_val.jsonl
    uv run train_laya.py --base multilingual                   # -> checkpoints/tuned-multilingual/
    uv run train_laya.py --base multilingual --class-weights sqrt --out checkpoints/tuned-multilingual-cw
    uv run train_laya.py --base multilingual --target per-tier --out checkpoints/tuned-multilingual-pertier
    uv run bench.py --n 0 --selectors laya-tuned-multilingual,laya-tuned-multilingual-pertier
"""
import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path

# Cap PyTorch's Mac-GPU (MPS) memory. The default lets its cache overcommit into swap, and with a
# different sequence length every batch that cache kept growing (a 20 GB run). Set before torch
# touches MPS; override by exporting the variables. This changes memory use, not the maths.
# The low watermark (default 1.4) must not exceed the high one, so both are set.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.6")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

os.environ.setdefault("USE_TF", "0")
import laya  # noqa: E402
from laya.common import collate_items  # noqa: E402

from bench import (TIER_QUESTION, TIERS, laya_items, laya_option_logits, per_tier_enough, route,  # noqa: E402
                   routing_curve, tier_enough, vs_blind_avg)

HERE = Path(__file__).parent
REPO = "convaiinnovations/laya"
BASES = {"english": None, "multilingual": "multilingual", "typed-decisions": "typed-decisions"}
RIGHT = 0.5  # a tier got it right at score >= 0.5 (squad's partial-credit F1; every other task scores 0 or 1)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def make_items(rows: list[dict], tok, cfg: dict) -> list[dict]:
    """Tokenise each row exactly as Laya's predict() would, with both targets: label = the ideal
    tier's index, right = whether each tier (T1, T2, T3) gets it right."""
    items = laya_items(tok, cfg, [r["question"] for r in rows], TIER_QUESTION)
    for it, r in zip(items, rows):
        it.update(label=TIERS.index(r["label"]), right=[float(r["scores"][t] >= RIGHT) for t in TIERS], qid=r["qid"])
    return items


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


def set_trainable(model, unfreeze_layers: int) -> int:
    """Train the decision head (+ the top `unfreeze_layers` encoder layers; -1 = all). Returns #params."""
    for p in model.parameters():
        p.requires_grad = False
    for part in (model.head, model.type_emb, model.scorer):  # act_head is unused by this loss
        if part is not None:
            for p in part.parameters():
                p.requires_grad = True
    enc = model.encoder
    if unfreeze_layers < 0:
        for p in enc.parameters():
            p.requires_grad = True
    elif unfreeze_layers > 0:
        for layer in enc.layers[-unfreeze_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in enc.final_norm.parameters():
            p.requires_grad = True
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def batch_logits(model, items: list[dict], device, pad_id: int, detach_encoder: bool) -> torch.Tensor:
    """Laya's forward pass: one score per option marker -> [batch, 3] logits over T1/T2/T3."""
    b = collate_items([items], pad_id)
    logits, _ = model(b["input_ids"].to(device), b["attention_mask"].to(device), b["marker_pos"].to(device),
                      b["marker_mask"].to(device), b["qtype"].to(device), detach_encoder=detach_encoder)
    return logits


@torch.no_grad()
def all_logits(model, items: list[dict], device, pad_id: int, bs: int) -> torch.Tensor:
    model.eval()
    return torch.cat([batch_logits(model, items[i:i + bs], device, pad_id, True).float().cpu()
                      for i in range(0, len(items), bs)])


# --------------------------------------------------------------------------
# Validation metrics
# --------------------------------------------------------------------------


def route_metrics(rows: list[dict], picks: np.ndarray) -> dict:
    """Accuracy kept / money saved vs always-T1, and vs the blind split at the same cost."""
    perf = np.array([[r["scores"][t] for t in TIERS] for r in rows])
    cost = np.array([[r["costs"][t] for t in TIERS] for r in rows])
    p, c = perf.mean(0), cost.mean(0)
    pf, cf = perf[np.arange(len(rows)), picks].mean(), cost[np.arange(len(rows)), picks].mean()
    if cf <= c[1]:
        a = (cf - c[2]) / (c[1] - c[2]); blind = a * p[1] + (1 - a) * p[2]
    else:
        a = (cf - c[1]) / (c[0] - c[1]); blind = a * p[0] + (1 - a) * p[1]
    return {"kept": round(float(pf / p[0] * 100), 2), "saved": round(float((1 - cf / c[0]) * 100), 2),
            "vs_blind": round(float((pf - blind) / p[0] * 100), 2),
            "mix": [round(float((picks == i).mean()) * 100, 1) for i in range(len(TIERS))]}


def best_cutoff(rows: list[dict], enough: np.ndarray) -> float:
    """The cut-off with the highest vs blind split: the one operating point bench.py routes by."""
    cuts, gain, _, _ = routing_curve(rows, enough)
    return float(cuts[np.argmax(gain)])


def auc(right: np.ndarray, score: np.ndarray) -> float:
    """Chance a right answer's question scores above a wrong one's (0.5 = no signal)."""
    ranks = np.empty(len(score)); ranks[np.argsort(score)] = np.arange(1, len(score) + 1)
    n1 = right.sum(); n0 = len(right) - n1
    return float((ranks[right == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def evaluate(logits: torch.Tensor, items: list[dict], rows: list[dict], temperature: float = 1.0,
             class_weight: torch.Tensor | None = None, target: str = "tier") -> dict:
    """tier: nll = the training objective (class-weighted when training is), nll_plain = unweighted,
    routed by argmax. per-tier: nll = binary cross-entropy, auc per tier, routed at the best cut-off."""
    if target == "per-tier":
        right = torch.tensor([it["right"] for it in items])
        p = torch.sigmoid(logits.float()).numpy()
        enough = per_tier_enough(p)
        cut = best_cutoff(rows, enough)
        return {"nll": round(float(F.binary_cross_entropy_with_logits(logits.float(), right)), 4),
                "auc": [round(auc(right[:, i].numpy(), p[:, i]), 3) for i in range(len(TIERS))],
                "cutoff": cut, **route_metrics(rows, route(enough, cut)), "vs_blind_avg": vs_blind_avg(rows, enough)}
    labels = torch.tensor([it["label"] for it in items])
    nll_plain = F.cross_entropy(logits / temperature, labels)
    nll = F.cross_entropy(logits / temperature, labels, weight=class_weight) if class_weight is not None else nll_plain
    picks = logits.argmax(-1)
    return {"nll": round(float(nll), 4), "nll_plain": round(float(nll_plain), 4),
            "ideal_acc": round(float((picks == labels).float().mean()) * 100, 2), **route_metrics(rows, picks.numpy()),
            "vs_blind_avg": vs_blind_avg(rows, tier_enough(torch.softmax(logits.float() / temperature, -1).numpy()))}


def class_weights(items: list[dict], mode: str) -> torch.Tensor | None:
    """none | sqrt | balanced. balanced = N / (K * count), so each tier counts equally in the loss;
    sqrt = its square root, a milder push towards the rare T1/T2 labels."""
    if mode == "none":
        return None
    counts = torch.bincount(torch.tensor([it["label"] for it in items]), minlength=len(TIERS)).float()
    w = counts.sum() / (len(TIERS) * counts.clamp(min=1))
    return w.sqrt() if mode == "sqrt" else w


def gpu_memory(device) -> str:
    if device.type == "mps":
        return f"{torch.mps.driver_allocated_memory() / 1e9:.1f} GB"
    if device.type == "cuda":
        return f"{torch.cuda.memory_reserved() / 1e9:.1f} GB"
    return "n/a"


def free_cache(device):
    """Hand cached-but-unused GPU memory back. No effect on results."""
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def fit_temperature(logits: torch.Tensor, items: list[dict]) -> float:
    """1-parameter calibration on validation, inside Laya's allowed range [0.5, 5]."""
    labels = torch.tensor([it["label"] for it in items])
    return float(min(np.linspace(0.5, 5.0, 91), key=lambda t: float(F.cross_entropy(logits / t, labels))))


# --------------------------------------------------------------------------
# Checkpoint
# --------------------------------------------------------------------------


def base_dir(subfolder: str | None) -> Path:
    from huggingface_hub import snapshot_download

    prefix = f"{subfolder}/" if subfolder else ""
    root = snapshot_download(REPO, allow_patterns=[prefix + n for n in ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")])
    return Path(root) / subfolder if subfolder else Path(root)


def save_checkpoint(model, cfg: dict, src: Path, out: Path, temperature: float | None, meta: dict):
    """A normal Laya checkpoint directory, loadable with laya.load(path). temperature None keeps the base's."""
    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    for sub in ("tokenizer", "encoder"):
        if (src / sub).exists():
            shutil.copytree(src / sub, out / sub, dirs_exist_ok=True)
    cfg = dict(cfg)
    if temperature is not None:
        cfg["temperature_by_options"] = {**cfg.get("temperature_by_options", {}), "choice:3-5": temperature}
    cfg["fine_tuned"] = meta
    (out / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2))
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(out / "model.safetensors"))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default=str(HERE / "data" / "tier_train.jsonl"))
    ap.add_argument("--val", default=str(HERE / "data" / "tier_val.jsonl"))
    ap.add_argument("--base", choices=list(BASES), default="multilingual")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16, help="questions per weight update")
    ap.add_argument("--micro-batch", type=int, default=4,
                    help="questions per forward/backward pass; gradients are accumulated up to --batch (same update, less memory)")
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--lr-encoder", type=float, default=2e-5)
    ap.add_argument("--unfreeze-layers", type=int, default=4, help="top encoder layers to train (0 = head only, -1 = all)")
    ap.add_argument("--target", choices=["tier", "per-tier"], default="tier",
                    help="tier = the one ideal tier (runs 1-2); per-tier = whether each tier gets it right")
    ap.add_argument("--class-weights", choices=["none", "sqrt", "balanced"], default="none",
                    help="weight rare labels (T1, T2) more in the loss; none = plain cross-entropy (tier target only)")
    ap.add_argument("--select", choices=["route", "nll"], default="route",
                    help="keep the epoch with the best vs_blind_avg on validation (route) or the lowest loss (nll, runs 1-2)")
    ap.add_argument("--eval-batch", type=int, default=64, help="questions per validation forward pass (memory only)")
    ap.add_argument("--empty-cache-every", type=int, default=10, help="steps between GPU cache releases (0 = never)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="default: checkpoints/tuned-<base>")
    args = ap.parse_args()
    per_tier = args.target == "per-tier"
    if per_tier and args.class_weights != "none":
        ap.error("--class-weights applies to the tier target only")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out) if args.out else HERE / "checkpoints" / f"tuned-{args.base}"

    train_rows, val_rows = read_rows(Path(args.train)), read_rows(Path(args.val))
    assert not {r["qid"] for r in train_rows} & {r["qid"] for r in val_rows}, "train and validation overlap"
    print(f"{len(train_rows)} training / {len(val_rows)} validation questions; training labels T1/T2/T3: "
          f"{[sum(r['label'] == t for r in train_rows) for t in TIERS]}; share right T1/T2/T3: "
          f"{[round(np.mean([r['scores'][t] >= RIGHT for r in train_rows]) * 100, 1) for t in TIERS]}")

    agent = laya.load(REPO, device=str(device), subfolder=BASES[args.base])
    model, tok, cfg = agent.model, agent.tok, agent.cfg
    pad = tok.pad_token_id
    tr, va = make_items(train_rows, tok, cfg), make_items(val_rows, tok, cfg)
    cw = class_weights(tr, args.class_weights)
    cw_dev = cw.to(device) if cw is not None else None
    if cw is not None:
        print(f"class weights ({args.class_weights}) T1/T2/T3: {[round(float(x), 3) for x in cw]}")

    base_t = agent.temperature_by_options.get("choice:3-5", agent.temperature[0])
    log = {"args": vars(args), "device": str(device), "n_train": len(tr), "n_val": len(va),
           "class_weights": None if cw is None else [float(x) for x in cw],
           "base": evaluate(all_logits(model, va, device, pad, args.eval_batch), va, val_rows, base_t, cw, args.target),
           "epochs": []}
    free_cache(device)
    print(f"base Laya on validation: {log['base']} | gpu {gpu_memory(device)}")

    n_trainable = set_trainable(model, args.unfreeze_layers)
    head_ps = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
    enc_ps = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": head_ps, "lr": args.lr_head}, {"params": enc_ps, "lr": args.lr_encoder}],
                            weight_decay=0.01)
    steps = args.epochs * ((len(tr) + args.batch - 1) // args.batch)
    warm = max(1, int(0.06 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
    print(f"training {n_trainable / 1e6:.1f}M of {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters "
          f"on {device}, {steps} steps")

    best, step = {"score": -float("inf")}, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(len(tr))
        t0, tot = time.time(), 0.0
        for i in range(0, len(order), args.batch):
            chunk = [tr[j] for j in order[i:i + args.batch]]
            labels = torch.tensor([it["label"] for it in chunk], device=device)
            right = torch.tensor([it["right"] for it in chunk], device=device)
            w = cw_dev[labels] if cw_dev is not None else torch.ones(len(chunk), device=device)
            opt.zero_grad(set_to_none=True)
            loss = 0.0
            # Gradient accumulation: run the batch in micro-batches and add their gradients. Each part is
            # divided by the whole batch's weight sum, so the total equals cross_entropy(weight=...) on
            # the full batch - the same update, with a fraction of the activation memory.
            for k in range(0, len(chunk), args.micro_batch):
                sub = chunk[k:k + args.micro_batch]
                logits = batch_logits(model, sub, device, pad, detach_encoder=args.unfreeze_layers == 0)  # forward
                if per_tier:  # vs each tier's right/wrong, averaged over the 3 tiers
                    per_q = F.binary_cross_entropy_with_logits(logits.float(), right[k:k + len(sub)], reduction="none").mean(1)
                else:         # vs the ideal-tier label
                    per_q = F.cross_entropy(logits.float(), labels[k:k + len(sub)], reduction="none")
                part = (per_q * w[k:k + len(sub)]).sum() / w.sum()
                part.backward()                                                                         # gradients
                loss += part.detach().item()
            torch.nn.utils.clip_grad_norm_(head_ps + enc_ps, 1.0)
            opt.step(); sched.step(); step += 1                                                         # update
            tot += loss * len(chunk)
            if args.empty_cache_every and step % args.empty_cache_every == 0:
                free_cache(device)
            if step % 25 == 0:
                print(f"  epoch {epoch} step {step}/{steps} train loss {tot / (i + len(chunk)):.4f} "
                      f"({time.time() - t0:.0f}s, gpu {gpu_memory(device)})", flush=True)
        free_cache(device)  # training's cached memory + a validation batch overflowed on typed-decisions
        m = evaluate(all_logits(model, va, device, pad, args.eval_batch), va, val_rows, class_weight=cw, target=args.target)
        free_cache(device)
        log["epochs"].append({"epoch": epoch, "train_loss": round(tot / len(tr), 4), **m, "seconds": round(time.time() - t0)})
        print(f"epoch {epoch}: train loss {tot / len(tr):.4f} | validation {m} | {time.time() - t0:.0f}s", flush=True)
        score = -m["nll"] if args.select == "nll" else m["vs_blind_avg"]
        if score > best["score"]:
            best = {"score": score, "epoch": epoch,
                    "state": {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}}

    # Restore the best epoch, calibrate its confidence (tier) or keep its cut-off (per-tier), save
    with torch.no_grad():
        params = dict(model.named_parameters())
        for n, v in best["state"].items():
            params[n].copy_(v.to(device))
    vl = all_logits(model, va, device, pad, args.eval_batch)
    temp = None if per_tier else fit_temperature(vl, va)
    log["best_epoch"], log["temperature"] = best["epoch"], temp
    log["tuned"] = evaluate(vl, va, val_rows, temp or 1.0, cw, args.target)
    print(f"best epoch {best['epoch']}, " + (f"cut-off {log['tuned']['cutoff']:.4f}" if per_tier else f"temperature {temp:.2f}")
          + f" | tuned on validation: {log['tuned']}")

    meta = {"base": args.base, "train_file": Path(args.train).name, "n_train": len(tr), "best_epoch": best["epoch"],
            "target": args.target, "select": args.select, "class_weights": args.class_weights,
            "question": "bench.TIER_QUESTION (run 3 input)", "validation": log["tuned"]}
    if per_tier:
        meta.update(cutoff=log["tuned"]["cutoff"], right_threshold=RIGHT)
    save_checkpoint(model, cfg, base_dir(BASES[args.base]), out, temp, meta)
    (out / "training_log.json").write_text(json.dumps(log, indent=2))

    # The saved checkpoint must load through laya and reproduce what we just measured, by the path
    # bench.py uses: predict() for the tier target, the raw option scores for per-tier
    reload = laya.load(str(out), device=str(device))
    if per_tier:
        got = 1 / (1 + np.exp(-laya_option_logits(reload, [r["question"] for r in val_rows[:16]], TIER_QUESTION)))
        want = torch.sigmoid(vl[:16].float()).numpy()
    else:
        got = np.array([[reload.predict({"request": r["question"]}, TIER_QUESTION)["answers"]["tier"]["probabilities"][t]
                         for t in TIERS] for r in val_rows[:16]])
        want = torch.softmax(vl[:16] / temp, -1).numpy()
    diff = float(np.abs(got - want).max())
    print(f"saved to {out} | reload check on 16 validation questions: max probability difference {diff:.5f}")
    assert diff < 1e-3, "reloaded checkpoint does not reproduce the trained model"


if __name__ == "__main__":
    main()
