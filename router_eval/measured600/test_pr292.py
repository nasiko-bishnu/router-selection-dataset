# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["torch", "transformers", "safetensors", "numpy", "scikit-learn"]
# ///
"""PR #292: fine-tuned all-MiniLM-L6-v2, retrained with one 3-tier head on the 600 training set, tested on the measured 600.

The model and recipe are the ones in ../retrain3/pr_292.py, unchanged. MINILM_BASE points at the PR's MiniLM
weights (the default is where the PR test checked them out).

    uv run router_eval/measured600/test_pr292.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "retrain3"))
sys.path.insert(0, str(HERE))
from harness import evaluate_model  # noqa: E402
import torch  # noqa: E402
from pr_292 import TIERS, FineTunedMiniLm  # noqa: E402


def probs(m, qs):  # softmax of the tier head; its argmax is predict()'s pick
    with torch.no_grad():
        p = torch.softmax(m.model(**m.encode(qs, [""] * len(qs))), -1).tolist()
    return [dict(zip(TIERS, r)) for r in p]


evaluate_model("pr-292", "PR, retrained", "all-MiniLM-L6-v2 fine-tuned (last 2 layers + tier head), 3-class",
               "Model and recipe as in ../retrain3/pr_292.py (16 epochs, best epoch on a 20% hold-out of eval_600).",
               m := FineTunedMiniLm(), proba=lambda qs: probs(m, qs))
