"""Threshold tables for every model whose predictions carry a confidence and tier probabilities.

Two rules, as in ../retrain3/thresholds.md (the 300-set tables):
  confidence < X -> high : keep the model's pick unless its confidence is below X; then send the question to high
  cascade cut-off C      : low if P(low) >= C, else mid if P(low) + P(mid) >= C, else high
                           (the cheapest tier whose chance of being enough reaches C)
Writes thresholds.md, which harness.report() puts into RESULTS.md. Every row is scored on the same 600 test
questions, so a "best" row is picked with the test labels and is optimistic. The rows marked "300-set pick" use
the threshold that was best on the authored 300 set, so they are an honest out-of-sample check.

    python3 router_eval/measured600/thresholds.py
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import PREDS, report, score  # noqa: E402

# Models whose confidence is their own figure (not the top probability), and the threshold for
# "confidence < X -> high" that was best on the authored 300 set (../retrain3/thresholds.md)
OWN_CONFIDENCE = {"jev-direct": "Jev's own confidence", "d3-flash-zeroshot": "d3-flash's own confidence (not its top probability)",
                  "clm-zeroshot": "CLM's own confidence"}
PICK_300 = {"jev-direct": 0.95, "d3-flash-zeroshot": 0.5, "clm-zeroshot": 0.5}


def models():
    """Every result whose predictions carry tier probabilities, best plain balanced accuracy first."""
    recs = sorted((json.loads(p.read_text()) for p in (HERE / "results").glob("*.json")),
                  key=lambda r: -r["metrics"]["balanced_accuracy"])
    out = []
    for r in recs:
        f = PREDS / f"{r['name']}.jsonl"
        if f.exists() and "probabilities" in json.loads(f.read_text().splitlines()[0]):
            out.append((r["name"], OWN_CONFIDENCE.get(r["name"], "top probability"), PICK_300.get(r["name"])))
    return out


CONF_X = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
CASCADE_C = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
BEST = HERE / "results" / "thresholds" / "best.json"


def folds(y, k=5, seed=0):
    """Stratified folds: each fold gets the same mix of tiers. Returns a fold number per question."""
    rng = random.Random(seed)
    fold = [0] * len(y)
    for t in sorted(set(y)):
        idx = [i for i, v in enumerate(y) if v == t]
        rng.shuffle(idx)
        for n, i in enumerate(idx):
            fold[i] = n % k
    return fold


def cv_choice(variants, y, k=5):
    """For each fold, choose the variant with the best balanced accuracy on the OTHER folds (ties: the earlier one,
    so "as answered" wins a tie), apply it to this fold, and score all 600 picks together. Honest: no question is
    scored with a threshold chosen using its own label."""
    fold = folds(y, k)
    picks, chosen = [None] * len(y), []
    for f in range(k):
        train = [i for i in range(len(y)) if fold[i] != f]
        best = max(range(len(variants)), key=lambda v: (score([y[i] for i in train], [variants[v][2][i] for i in train])
                                                        ["balanced_accuracy"], -v))
        chosen.append(f"{variants[best][0]} {variants[best][1].replace(' (300-set pick)', '')}".replace(" —", ""))
        for i in range(len(y)):
            if fold[i] == f:
                picks[i] = variants[best][2][i]
    return score(y, picks), chosen


def cascade(p, c):
    return "low" if p["low"] >= c else "mid" if p["low"] + p["mid"] >= c else "high"


def main():
    lines = ["## 🎚️ Results at different thresholds", "",
             "Every model that gives tier probabilities is thresholded: all runs except Nasiko's router, which only "
             "samples a tier. Where a model has no confidence of its own, its top probability is used.", "",
             "**Rules:**",
             "- **Confidence < X → high:** keep the model's pick unless its confidence is below X; then send the question to high.",
             "- **Cascade, cut-off C:** low if P(low) ≥ C, else mid if P(low) + P(mid) ≥ C, else high: the cheapest tier "
             "whose chance of being enough reaches C.", "",
             "> [!WARNING]",
             "> Every row is scored on the same 600 questions, so the **best** row is chosen with the test labels and is "
             "optimistic. Two honest checks: **cross-validated** (5 stratified folds; for each fold the rule and threshold, "
             "or the plain answer, are chosen on the other 4 folds and applied to this one) and **300-set pick** (the "
             "threshold that was best on the authored 300 set).", ""]
    summary, best_rows = [], {}
    for name, conf_def, pick300 in models():
        title = name
        f = PREDS / f"{name}.jsonl"
        rows = [json.loads(l) for l in f.read_text().splitlines()]
        y = [r["label"] for r in rows]
        variants = [("As answered", "—", [r["pick"] for r in rows])]
        variants += [("Confidence < X → high", f"{x:g}" + (" (300-set pick)" if x == pick300 else ""),
                      [r["pick"] if r["confidence"] >= x else "high" for r in rows]) for x in CONF_X]
        variants += [("Cascade", f"{c:g}", [cascade(r["probabilities"], c) for r in rows]) for c in CASCADE_C]
        scored = [(rule, t, score(y, picks)) for rule, t, picks in variants]
        best = max(m["balanced_accuracy"] for _, _, m in scored)
        lines += [f"### {title}", "", f"*Confidence: {conf_def}.*", "",
                  "| Rule | Threshold | **Balanced acc.** | Accuracy | Recall low / mid / high | Too small | Too big | Picks low / mid / high |",
                  "|---|---|---:|---:|---|---:|---:|---|"]
        for rule, t, m in scored:
            bal = f"**{m['balanced_accuracy']}%**" if m["balanced_accuracy"] == best else f"{m['balanced_accuracy']}%"
            lines.append(f"| {rule} | {t} | {bal} | {m['accuracy']}% | {m['recall_low']:g} / {m['recall_mid']:g} / "
                         f"{m['recall_high']:g}% | {m['too_small']}% | {m['too_big']}% | {m['picks']} |")
        cv, chosen = cv_choice(variants, y)
        lines += ["", f"**Cross-validated (honest):** {cv['balanced_accuracy']}% balanced, {cv['accuracy']}% accuracy, recall "
                  f"{cv['recall_low']:g} / {cv['recall_mid']:g} / {cv['recall_high']:g}%. Chosen per fold: {', '.join(chosen)}.", ""]
        top = max(scored, key=lambda s: s[2]["balanced_accuracy"])
        p300 = next((s for s in scored if "300-set pick" in s[1]), None)
        summary.append((title, scored[0][2]["balanced_accuracy"], top, p300, cv))
        best_rows[name] = {"rule": top[0], "threshold": top[1].replace(" (300-set pick)", ""), "metrics": top[2],
                           "cv": {"metrics": cv, "chosen": chosen}}
    head = ["### Best per model", "",
            "| Model | As answered | **Cross-validated threshold (honest)** | Best on these labels (optimistic) | Best rule and threshold | 300-set pick (honest) |",
            "|---|---:|---:|---:|---|---:|"]
    best_cv = max((cv["balanced_accuracy"] for *_, cv in summary), default=0)
    head += [f"| **{t}** | {a}% | {'**' if cv['balanced_accuracy'] == best_cv else ''}{cv['balanced_accuracy']}%"
             f"{'**' if cv['balanced_accuracy'] == best_cv else ''} | {top[2]['balanced_accuracy']}% | "
             f"{top[0]}, {top[1].replace(' (300-set pick)', '')} | {'—' if p is None else str(p[2]['balanced_accuracy']) + '%'} |"
             for t, a, top, p, cv in summary]
    i = lines.index("**Rules:**")
    lines[i:i] = head + [""]
    (HERE / "thresholds.md").write_text("\n".join(lines) + "\n")
    BEST.parent.mkdir(parents=True, exist_ok=True)  # read by harness.report() for the rows under each model
    BEST.write_text(json.dumps(best_rows, indent=2) + "\n")
    report()
    for t, a, top, p, cv in summary:
        print(f"{t}: as answered {a}% | CV-chosen {cv['balanced_accuracy']}% | best on labels {top[2]['balanced_accuracy']}% "
              f"({top[0]} {top[1]}) | 300-set pick {'—' if p is None else p[2]['balanced_accuracy']}")


if __name__ == "__main__":
    main()
