"""Turn the judge's grades into measured tiers.

Joins grades/*.jsonl with judge_key.json, then for each question and tier: the tier passes when every
model run for that tier passed (the strict rule). The measured tier is the cheapest tier that passes;
NONE when even HIGH fails. Questions where the two models of a tier disagree are listed as borderline.

    python3 router_eval/tier_measure/measure_tiers.py --tier LOW=openai:gpt-4.1-nano,mlx:... \
        --tier MID=openai:gpt-4.1-mini,mlx:... --tier HIGH=openai:gpt-5.5,openai:gpt-5.4
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import GRADES, HERE, KEY, TIERS, candidates  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", action="append", required=True, metavar="TIER=model,model")
    a = ap.parse_args()
    tier_models = {t: m.split(",") for t, m in (x.split("=", 1) for x in a.tier)}
    # the key make_judge_pack.py wrote (kept away from the judge), else the archived copy of the published grades' key
    key = json.loads((KEY if KEY.exists() else HERE / "judge_key.json").read_text())
    passed = defaultdict(dict)  # question_id -> model -> pass
    for f in sorted(GRADES.glob("*.jsonl")):
        for g in map(json.loads, f.read_text().splitlines()):
            k = key[g["item_id"]]
            passed[k["question_id"]][k["model"]] = bool(g["pass"])
    out, borderline = [], []
    for c in candidates():
        p = passed.get(c["id"], {})
        tier_pass = {}
        for t in TIERS:
            ms = tier_models.get(t, [])
            if ms and all(m in p for m in ms):
                tier_pass[t] = all(p[m] for m in ms)
                if len({p[m] for m in ms}) > 1:
                    borderline.append((c["id"], t))
        done = [t for t in TIERS if t in tier_models]
        measured = None
        if all(t in tier_pass for t in done):
            measured = next((t for t in done if tier_pass[t]), "NONE")
        out.append({"id": c["id"], "query": c["query"], "label_guess": c.get("label"), "measured": measured,
                    "tier_pass": tier_pass, "source": c.get("source")})
    (HERE / "measured_tiers.jsonl").write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out))
    m = [o for o in out if o["measured"]]
    print(f"measured {len(m)} of {len(out)} questions | tiers: {dict(Counter(o['measured'] for o in m))}")
    g = [o for o in m if o["label_guess"]]
    if g:
        agree = sum(o["label_guess"] == o["measured"] for o in g)
        print(f"authored labels that match the measured tier: {agree}/{len(g)} = {100 * agree / len(g):.1f}%")
    print(f"borderline (a tier's models disagree): {len(borderline)}")


if __name__ == "__main__":
    main()
