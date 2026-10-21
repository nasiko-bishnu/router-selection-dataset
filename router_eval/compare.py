# /// script
# requires-python = ">=3.10"
# ///
"""Score classifier outputs on our router sets and rebuild router_eval/PR_COMPARISON.md.

Every PR's classifier_eval writes one JSON line per question:
  {"id", "request_type", "complexity", "confidence", "latency_us"}
The PRs give complexity 1-5, which maps to our tiers like the public sample:
  1-2 -> low, 3 -> mid, 4-5 -> high.  A missing complexity means no tier.

    python3 router_eval/compare.py score --name main --data eval_600 --out out.jsonl \
        --setup "regex (main)" --build-seconds 300 --run-seconds 1.2 --cost 0
    python3 router_eval/compare.py report      # rebuild PR_COMPARISON.md from results/*.json
"""
import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS, DOC = HERE / "results", HERE / "PR_COMPARISON.md"
DATASETS = {"eval_600": HERE / "data" / "eval_600.json", "own_300": HERE / "data" / "own_300.json",
            "oob_60": HERE / "data" / "oob_60.json"}
TIERS = ["low", "mid", "high"]
RANK = {"low": 0, "mid": 1, "high": 2}
PRICE = {"low": 0.017, "mid": 0.275, "high": 0.458}  # $ per 1,000 questions, relative index from our benchmark
TYPES = ["code_generation", "code_understanding", "technical_design", "analytical_reasoning", "writing",
         "factual_lookup", "general"]


def tier_of(complexity):
    try:
        c = int(round(float(complexity)))
    except (TypeError, ValueError):
        return None
    return "low" if c <= 2 else "mid" if c == 3 else "high"


def pct(x):
    return None if x is None else round(100 * x, 1)


def score(data_name: str, out_path: Path) -> dict:
    examples = json.loads(DATASETS[data_name].read_text())["examples"]
    outs = {}
    for line in out_path.read_text().splitlines():
        if line.strip():
            try:
                o = json.loads(line)
                outs[o["id"]] = o
            except (json.JSONDecodeError, KeyError):
                pass
    n = len(examples)
    got = [outs.get(e["id"], {}) for e in examples]
    m = {"questions": n, "answered": sum(bool(g) for g in got)}

    if "request_type" in examples[0] and any(g.get("request_type") for g in got):  # skipped when the classifier gives no type
        right = [g.get("request_type") == e["request_type"] for g, e in zip(got, examples)]
        m["type_accuracy"] = pct(sum(right) / n)
        m["type_general_share"] = pct(sum(g.get("request_type") == "general" for g in got) / n)
        f1 = []
        for t in TYPES:
            tp = sum(g.get("request_type") == t and e["request_type"] == t for g, e in zip(got, examples))
            fp = sum(g.get("request_type") == t and e["request_type"] != t for g, e in zip(got, examples))
            fn = sum(g.get("request_type") != t and e["request_type"] == t for g, e in zip(got, examples))
            f1.append(0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
        m["type_macro_f1"] = round(sum(f1) / len(f1), 3)

    tiers = [tier_of(g.get("complexity")) for g in got]
    has = [t is not None for t in tiers]
    m["tier_coverage"] = pct(sum(has) / n)
    if any(has):
        labels = [e["label"] for e in examples]
        m["tier_accuracy"] = pct(sum(t == l for t, l in zip(tiers, labels)) / n)  # missing counts as wrong
        for t in TIERS:
            idx = [i for i, l in enumerate(labels) if l == t]
            m[f"recall_{t}"] = pct(sum(tiers[i] == t for i in idx) / len(idx))
        scored = [(t, l) for t, l in zip(tiers, labels) if t]
        m["too_small"] = pct(sum(RANK[t] < RANK[l] for t, l in scored) / len(scored))
        m["too_big"] = pct(sum(RANK[t] > RANK[l] for t, l in scored) / len(scored))
        m["cost_per_1k"] = round(statistics.mean(PRICE[t] for t, _ in scored), 3)
        m["picks"] = "/".join(str(sum(t == x for t in tiers)) for x in TIERS)

    lat = sorted(float(g["latency_us"]) / 1000 for g in got if g.get("latency_us") is not None)
    if lat:
        q = lambda p: round(lat[min(len(lat) - 1, int(p * len(lat)))], 3)  # noqa: E731
        m["latency_ms_p50"], m["latency_ms_p95"], m["latency_ms_p99"] = q(0.5), q(0.95), q(0.99)
        m["latency_ms_mean"] = round(statistics.mean(lat), 3)
    conf = [g.get("confidence") for g in got if isinstance(g.get("confidence"), (int, float))]
    m["confidence_reported"] = pct(len(conf) / n)
    return m


def cmd_score(a):
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{a.name}.json"
    rec = json.loads(path.read_text()) if path.exists() else {"name": a.name, "runs": {}}
    rec.update({k: v for k, v in {"setup": a.setup, "pr": a.pr, "notes": a.notes, "build_seconds": a.build_seconds}.items() if v is not None})
    m = score(a.data, Path(a.out))
    m.update({"run_seconds": a.run_seconds, "cost_usd": a.cost, "scored_at": datetime.now().isoformat(timespec="seconds")})
    rec["runs"][a.data] = m
    path.write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps({a.data: m}, indent=1))


def fmt(v, suffix=""):
    return "—" if v is None else f"{v}{suffix}"


def usd(v):
    """Dollar amount for the tables; `\\$` because Markdown previews read $...$ pairs as math."""
    return "—" if v is None else "\\$" + (f"{v:.5f}".rstrip("0").rstrip(".") if v else "0")


def cmd_report(_):
    recs = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("*.json"))]
    order = lambda r: (0 if r["name"].startswith("baseline") else 1 if r["name"].startswith("laya") else 2, r["name"])  # noqa: E731
    recs.sort(key=order)
    spent = sum(m.get("cost_usd") or 0 for r in recs for m in r["runs"].values())
    lines = ["# Router classifier comparison: our 300 and 600 question sets", "",
             "> **TL;DR** — Every classifier tested so far, on the same two sets, scored the same way. "
             f"Updated after each test. Spent on hosted classifiers so far: {usd(round(spent, 3))} (approved Jev runs only).", "",
             f"*Last updated: {datetime.now().strftime('%d %b %Y %H:%M')} · {len(recs)} runs*", "", "---", "",
             *(((HERE / "comparison_notes.md").read_text().rstrip().split("\n") + ["", "---", ""]) if (HERE / "comparison_notes.md").exists() else []),
             "## 📊 600-question set (tiers + request types)", "",
             "| Run | Setup | **Type acc.** | Type macro-F1 | **Tier acc.** | Recall low / mid / high | Too small | Too big | Cost / 1k | p50 latency | Run time | Decision cost |",
             "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in recs:
        m = r["runs"].get("eval_600")
        if not m:
            continue
        rec_ = " / ".join(fmt(m.get(f"recall_{t}"), "%") for t in TIERS) if m.get("tier_accuracy") is not None else "—"
        lines.append(f"| **{r['name']}** | {r.get('setup', '')} | {fmt(m.get('type_accuracy'), '%')} | {fmt(m.get('type_macro_f1'))} | "
                     f"{fmt(m.get('tier_accuracy'), '%')} | {rec_} | {fmt(m.get('too_small'), '%')} | {fmt(m.get('too_big'), '%')} | "
                     f"{usd(m.get('cost_per_1k'))} | {fmt(m.get('latency_ms_p50'), ' ms')} | "
                     f"{fmt(m.get('run_seconds'), ' s')} | {usd(m.get('cost_usd') or 0)} |")
    lines += ["", "## 📊 300-question set (tiers only)", "",
              "| Run | Setup | **Tier acc.** | Recall low / mid / high | Too small | Too big | Cost / 1k | p50 latency | Run time | Decision cost |",
              "|---|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in recs:
        m = r["runs"].get("own_300")
        if not m:
            continue
        rec_ = " / ".join(fmt(m.get(f"recall_{t}"), "%") for t in TIERS) if m.get("tier_accuracy") is not None else "—"
        lines.append(f"| **{r['name']}** | {r.get('setup', '')} | {fmt(m.get('tier_accuracy'), '%')} | {rec_} | {fmt(m.get('too_small'), '%')} | "
                     f"{fmt(m.get('too_big'), '%')} | {usd(m.get('cost_per_1k'))} | "
                     f"{fmt(m.get('latency_ms_p50'), ' ms')} | {fmt(m.get('run_seconds'), ' s')} | {usd(m.get('cost_usd') or 0)} |")
    lines += ["", "## 🧾 Notes per run", ""]
    for r in recs:
        b = r.get("build_seconds")
        lines.append(f"- **{r['name']}**{' (PR #' + str(r['pr']) + ')' if r.get('pr') else ''}: {r.get('notes', '')}"
                     f"{' Build ' + str(round(b)) + ' s.' if b else ''}")
    lines += ["", "---", "", "## 📖 How to read this", "",
              "| Column | Meaning |", "|---|---|",
              "| **Type acc.** | Share of questions where the predicted request type equals our label (600 set only; random = 14.3%) |",
              "| Type macro-F1 | Average F1 over the 7 types, so rare and common types count equally |",
              "| **Tier acc.** | The PR's complexity mapped to a tier (1–2 low, 3 mid, 4–5 high) and compared with our label; no complexity counts as wrong (random = 33.3%) |",
              "| Recall low / mid / high | Of the questions with that label, the share given that tier |",
              "| Too small / Too big | Share of tiered questions sent to a cheaper / dearer tier than the label (answer at risk / money wasted) |",
              "| Cost / 1k | Answering cost of the picked tiers per 1,000 questions, with our relative prices (low \\$0.017, mid \\$0.275, high \\$0.458); perfect picks cost \\$0.250 |",
              "| p50 latency | Median time to classify one question, as reported by the evaluator |",
              "| Run time | Wall time to classify the whole set |",
              "| Decision cost | Money spent on the classifier itself (local = \\$0) |",
              "", "> [!NOTE]", "> Our labels are judgement, not measured model results. Request-type scores only exist for the 600 set."]
    DOC.write_text("\n".join(lines) + "\n")
    print(f"wrote {DOC} ({len(recs)} runs)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--name", required=True); s.add_argument("--data", choices=list(DATASETS), required=True)
    s.add_argument("--out", required=True); s.add_argument("--setup"); s.add_argument("--pr", type=int)
    s.add_argument("--notes"); s.add_argument("--build-seconds", type=float); s.add_argument("--run-seconds", type=float)
    s.add_argument("--cost", type=float, default=0.0)
    sub.add_parser("report")
    a = ap.parse_args()
    cmd_score(a) if a.cmd == "score" else cmd_report(a)


if __name__ == "__main__":
    main()
