"""Shared helpers for measuring tiers: candidate ids and file locations."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CANDIDATES = ROOT / "llm_tier_selection_candidates_600.jsonl"
ANSWERS = HERE / "answers"            # one file per model: answers/<model slug>.jsonl
PACK = HERE / "judge_pack"            # what the judge session sees: anonymised and shuffled
KEY = Path.home() / ".cache" / "xroutebench" / "judge_key.json"  # which model wrote which item: outside the repo, away from the judge
GRADES = HERE / "grades"              # written by the judge session
TIERS = ("LOW", "MID", "HIGH")


def qid(query):
    """Stable id for a candidate question (its text decides it)."""
    return hashlib.sha1(query.encode("utf-8")).hexdigest()[:12]


def candidates():
    rows = [json.loads(l) for l in CANDIDATES.read_text().splitlines() if l.strip()]
    for r in rows:
        r["id"] = qid(r["query"])
    assert len({r["id"] for r in rows}) == len(rows), "duplicate candidate ids"
    return rows


def slug(model):
    return model.replace("/", "__").replace(":", "_")
