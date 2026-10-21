"""Build llm_tier_selection_600_measured.jsonl: all 600 candidates, labelled with their measured tier.

The label comes from router_eval/tier_measure/measured_tiers.jsonl, where the lower tier has priority:
LOW when both LOW models pass; otherwise MID when both MID models pass; otherwise HIGH when the HIGH
model passes. A question that every tier failed (11 of them) is labelled HIGH, since a router should
send what no tier solves to the strongest model.

    python3 router_eval/measured600/make_measured_600.py
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DIR.parent / "tier_measure"))
from common import HERE, candidates  # noqa: E402

OUT = DIR / "data" / "llm_tier_selection_600_measured.jsonl"


def main():
    measured = {m["id"]: m["measured"] for m in map(json.loads, (HERE / "measured_tiers.jsonl").read_text().splitlines())}
    rows = []
    for c in candidates():
        tier = measured[c["id"]]
        assert tier in ("LOW", "MID", "HIGH", "NONE"), f"{c['id']} has no measured tier"
        row = {"query": c["query"], "label": "HIGH" if tier == "NONE" else tier, "topic": c["topic"],
               "word_count": c["word_count"]}
        if c.get("reference_answer"):
            row["reference_answer"] = c["reference_answer"]
        rows.append(row)
    assert len(rows) == 600
    random.Random(0).shuffle(rows)  # mix authored and oasst2 questions
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(rows)} rows -> {OUT.name} | labels: {dict(Counter(r['label'] for r in rows))}")


if __name__ == "__main__":
    main()
