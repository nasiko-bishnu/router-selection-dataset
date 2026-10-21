# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas", "pyarrow", "numpy", "huggingface_hub", "laya", "torch", "safetensors", "requests"]
# ///
"""The three local Laya models (English, multilingual, typed-decisions) on the measured 600, zero-shot and fine-tuned.

Laya is asked Jev's low / mid / high question, so its three options are the labels themselves
(../laya600/laya600.py LayaTier, unchanged). Fine-tuned = the setting chosen earlier by validation loss
(../laya600/README.md): 20 epochs, top 8 encoder layers, lr head 2e-4 / encoder 4e-5, seed 0, best epoch on
10% of eval_600 held out. Runs on the Mac GPU.

    uv run router_eval/measured600/test_laya.py            # all three bases, zero-shot then fine-tuned
    uv run router_eval/measured600/test_laya.py english    # one base
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "laya600"))
sys.path.insert(0, str(HERE))
from harness import evaluate_model, report  # noqa: E402
from laya600 import DEVICE, OPTIONS, LayaTier  # noqa: E402
import torch  # noqa: E402
from train_laya import BASES, all_logits, free_cache  # noqa: E402


def probs(m, qs):  # softmax of Laya's option scores; its argmax is predict()'s pick
    p = torch.softmax(all_logits(m.model, m._items(qs), DEVICE, m.pad, m.eval_batch).float(), -1).tolist()
    return [dict(zip(OPTIONS, r)) for r in p]


for base in (sys.argv[1:] or list(BASES)):
    m = LayaTier(base, train=False)
    evaluate_model(f"laya-{base}-zeroshot", "Decision model, zero-shot (local)", f"Laya {base} asked low / mid / high, top option",
                   "The base model as published, no training.", m, train=False,
                   proba=lambda qs: probs(m, qs))
    del m
    free_cache(DEVICE)
    m = LayaTier(base, epochs=20, unfreeze_layers=8, lr_head=2e-4, lr_encoder=4e-5, seed=0)
    evaluate_model(f"laya-{base}-finetuned", "Decision model, fine-tuned (local)", f"Laya {base} fine-tuned on eval_600 (options low / mid / high)",
                       "LayaTier from ../laya600: plain cross-entropy, AdamW, top 8 encoder layers, up to 20 epochs, best epoch "
                       "by loss on 10% of eval_600 held out.", m, proba=lambda qs: probs(m, qs))
    p = HERE / "results" / f"laya-{base}-finetuned.json"  # record the chosen epoch with the result
    rec = json.loads(p.read_text()); rec["notes"] += f" Chosen epoch: {m.log.get('best_epoch')}."; rec["training_log"] = m.log
    p.write_text(json.dumps(rec, indent=2) + "\n")
    report()
    print(f"laya-{base}-finetuned: best epoch {m.log.get('best_epoch')}", flush=True)
    del m
    free_cache(DEVICE)
