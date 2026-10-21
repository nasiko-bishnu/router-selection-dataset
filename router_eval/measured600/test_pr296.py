# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["fastembed", "numpy", "scipy", "scikit-learn"]
# ///
"""PR #296, both backends, retrained as 3-tier classifiers on the 600 training set, tested on the measured 600.

The models are the ones in ../retrain3/pr_296_semantic.py (bge-small-en-v1.5 + shape features + LR) and
../retrain3/pr_296_lexical.py (word/char n-grams + shape features + LR), unchanged.

    uv run router_eval/measured600/test_pr296.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "retrain3"))
sys.path.insert(0, str(HERE))
from harness import evaluate_model  # noqa: E402
from pr_296_lexical import Lexical  # noqa: E402
from pr_296_semantic import BgeLR, featurize  # noqa: E402

sem, lex = BgeLR(), Lexical()  # the pick is the LR's most likely class, as in predict(); its probabilities are kept
evaluate_model("pr-296-semantic", "PR, retrained", "bge-small-en-v1.5 (fastembed) + shape features + LR (C=4), 3-class",
               "Model as in ../retrain3/pr_296_semantic.py, trained on eval_600.", sem,
               proba=lambda qs: [dict(zip(sem.clf.classes_, p)) for p in sem.clf.predict_proba(featurize(qs, [""] * len(qs)))])
evaluate_model("pr-296-lexical", "PR, retrained", "word 1-2 grams + char 3-5 grams + shape features + LR (C=20), 3-class",
               "Model as in ../retrain3/pr_296_lexical.py, trained on eval_600.", lex,
               proba=lambda qs: [dict(zip(lex.clf.classes_, p)) for p in lex.clf.predict_proba(lex.matrix(qs, [""] * len(qs)))])
