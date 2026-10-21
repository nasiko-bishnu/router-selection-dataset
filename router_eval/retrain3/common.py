"""Shared harness for retraining PR classifiers as direct 3-tier models.

Every model sees exactly three classes (low / mid / high), never a 1-5 complexity that is
mapped afterwards. Train: our 600 set (native tier labels). Test: our 300 set, used once.
Also reports 5-fold cross-validated accuracy on the 600 set, for in-distribution context.

Context is dropped (passed as ""): in the 600 set most low questions carry a context and few
mid/high ones do, so a model would learn "has context -> low"; the 300 set has no context at all.

A PR script defines `make()` returning an object with fit(queries, contexts, labels) and
predict(queries, contexts) -> list of "low"/"mid"/"high", then calls
    run(name="pr-319", pr=319, setup="...", make=make, notes="...")
which trains, times prediction per question, scores and rebuilds TIER_SELECTION_RESULTS.md.
"""
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from compare import TIERS, score  # noqa: E402

RESULTS, OUTS, DOC = HERE / "results", HERE / "outputs", ROOT / "TIER_SELECTION_RESULTS.md"
COMPLEXITY = {"low": 1, "mid": 3, "high": 5}  # how a tier is written so compare.score reads it back unchanged


def load(name):
    ex = json.loads((ROOT / "data" / f"{name}.json").read_text())["examples"]
    return [e["query"] for e in ex], ["" for _ in ex], [e["label"] for e in ex], ex  # context dropped, see above


def cv_accuracy(make, q, c, y, folds=5, seed=0):
    idx = list(range(len(q)))
    random.Random(seed).shuffle(idx)
    right = 0
    for k in range(folds):
        test = set(idx[k::folds])
        tr = [i for i in idx if i not in test]
        m = make()
        m.fit([q[i] for i in tr], [c[i] for i in tr], [y[i] for i in tr])
        te = sorted(test)
        pred = m.predict([q[i] for i in te], [c[i] for i in te])
        right += sum(p == y[i] for p, i in zip(pred, te))
    return round(100 * right / len(q), 1)


def run(name, pr, setup, make, notes="", cv=True):
    q, c, y, _ = load("eval_600")
    tq, tc, _, test = load("own_300")
    cv_acc = cv_accuracy(make, q, c, y) if cv else None
    model = make()
    t0 = time.time()
    model.fit(q, c, y)
    train_s = round(time.time() - t0, 1)
    model.predict(tq[:1], tc[:1])  # warm up
    OUTS.mkdir(parents=True, exist_ok=True)
    out = OUTS / f"{name}_own_300.jsonl"
    t0 = time.time()
    with out.open("w") as f:
        for e, qq, cc in zip(test, tq, tc):
            s = time.perf_counter()
            tier = model.predict([qq], [cc])[0]
            us = (time.perf_counter() - s) * 1e6
            assert tier in TIERS, f"{name} predicted {tier!r}, expected one of {TIERS}"
            f.write(json.dumps({"id": e["id"], "request_type": None, "complexity": COMPLEXITY[tier],
                                "confidence": None, "latency_us": us}) + "\n")
    m = score("own_300", out)
    m.update({"run_seconds": round(time.time() - t0, 2), "train_seconds": train_s, "cv_600": cv_acc})
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(
        {"name": name, "pr": pr, "setup": setup, "notes": notes, "own_300": m,
         "scored_at": datetime.now().isoformat(timespec="seconds")}, indent=2) + "\n")
    print(f"{name}: test tier acc {m['tier_accuracy']}% | CV on 600 {cv_acc}% | picks {m['picks']}", flush=True)
    report()
    return m


def report():
    recs = sorted((json.loads(p.read_text()) for p in RESULTS.glob("*.json")),
                  key=lambda r: -(r["own_300"].get("tier_accuracy") or 0))
    orig = {}
    for p in (ROOT / "results").glob("pr-*.json"):
        r = json.loads(p.read_text())
        m = r.get("runs", {}).get("own_300")
        if r.get("pr") and m and m.get("tier_accuracy") is not None:
            orig[r["pr"]] = max(orig.get(r["pr"], 0), m["tier_accuracy"])
    best = max((r["own_300"].get("tier_accuracy") or 0 for r in recs), default=0)
    lines = ["# Tier selection results: every classifier on our 300-question set", "",
             "> **TL;DR** — Which model tier (low / mid / high) does each question need? Every classifier we tested, scored on "
             "the same 300 questions: the PR models retrained as direct 3-tier classifiers (no 1–5 complexity mapped afterwards), "
             "the decision models CLM and d3-flash with small heads trained, and Jev, CLM, d3-flash and Laya zero-shot at "
             "different thresholds. Models are trained on our 600 set and tested once on the 300 set. "
             f"Best test accuracy so far: **{best}%** (random = 33.3%). Terms are explained in the [Glossary](#-glossary) at the end.", "",
             f"*Last updated: {datetime.now().strftime('%d %b %Y %H:%M')} · {len(recs)} models*", "", "---", "",
             *(((HERE / "recommendation.md").read_text().rstrip().split("\n") + ["", "---", ""]) if (HERE / "recommendation.md").exists() else []),
             *(((HERE / "findings.md").read_text().rstrip().split("\n") + ["", "---", ""]) if (HERE / "findings.md").exists() else []),
             "## 📊 Results on the 300 set", "",
             "| Model | PR | Setup | **Tier acc.** | Original PR (5 levels mapped) | Change | CV on 600 | Recall low / mid / high | Too small | Too big | Picks low / mid / high | p50 latency |",
             "|---|---:|---|---:|---:|---:|---:|---|---:|---:|---|---:|"]
    for r in recs:
        m = r["own_300"]
        o = orig.get(r["pr"])
        acc = m.get("tier_accuracy")
        diff = "—" if o is None or acc is None else f"{acc - o:+.1f}"
        acc_s = f"**{acc}%**" if acc == best else f"{acc}%"
        lines.append(f"| **{r['name']}** | {r['pr'] or '—'} | {r['setup']} | {acc_s} | {'—' if o is None else str(o) + '%'} | {diff} | "
                     f"{'—' if m.get('cv_600') is None else str(m['cv_600']) + '%'} | "
                     f"{' / '.join(str(m.get('recall_' + t)) + '%' for t in TIERS)} | {m.get('too_small')}% | {m.get('too_big')}% | "
                     f"{m.get('picks')} | {m.get('latency_ms_p50')} ms |")
    lines += reference_section()
    if (HERE / "thresholds.md").exists():
        lines += (HERE / "thresholds.md").read_text().rstrip().split("\n")
    lines += ["", "## 🧾 Notes per model", ""]
    lines += [f"- **{r['name']}**: {r['notes']} Train {r['own_300'].get('train_seconds')} s." for r in recs]
    lines += ["", "---", "", "## ⚙️ How it was tested", "",
              "- **Train:** `router_eval/data/eval_600.json`, 200 questions per tier, native low / mid / high labels, question text only.",
              "- **Context dropped:** in the 600 set, 165 of 200 low questions have a context but only 58 mid and 39 high do, so a model would learn \"has context = low\". The 300 set has no context.",
              "- **Test:** `router_eval/data/own_300.json`, 100 per tier, no overlap with the training set, used once per model.",
              "- **CV on 600:** 5-fold cross-validated accuracy on the training set: in-distribution, so expect it to be higher than the test.",
              "- **Original PR:** the same PR's shipped model on the 300 set, with its 1–5 complexity mapped to tiers (best run if it has several modes).",
              "- **Latency:** time to classify one question in Python, not in the PR's Rust runtime.",
              "", "> [!NOTE]", "> Our tier labels are judgement, not measured model results.",
              "", "> [!WARNING]", "> Jev, Laya and LLM-based classifiers are excluded from this test by request. "
              "Rule-only PRs have nothing to train, so they're not included.",
              "", "> [!WARNING]", "> The 600 set still has a length hint: low questions are shorter (median 9 words, against 14 for mid and 16 for high). "
              "The 300 set has no such hint (18–20 words for every tier), so a model that learns it is penalised in the test, not rewarded."]
    if (HERE / "glossary.md").exists():
        lines += ["", "---", ""] + (HERE / "glossary.md").read_text().rstrip().split("\n")
    DOC.write_text("\n".join(lines) + "\n")


def reference_section():
    """Earlier Jev and Laya results on the same 300 questions (not retrained), for comparison."""
    rows = []
    for f in sorted(HERE.glob("reference_*_direct.json")):  # Jev, CLM: decision models asked directly
        ref = json.loads(f.read_text())
        for r in ref["rows"]:
            rows.append((r["name"], ref["kind"], r["how"], r["tier_accuracy"], r["recall"], r["too_small"], r["too_big"],
                         r["picks"], ref["p50_latency_ms"], "**yes**" if r["tuned_on_test"] else "no"))
    for p in sorted((ROOT / "results").glob("*.json")):
        r = json.loads(p.read_text())
        m = r.get("runs", {}).get("own_300")
        if not m or m.get("tier_accuracy") is None:
            continue
        if r["name"].startswith("laya-"):
            group = "Laya direct (local)"
        elif r["name"].endswith("-laya"):
            group = "Laya through a PR (local)"
        elif r["name"].endswith("-jev"):
            group = "Jev through a PR (hosted)"
        else:
            continue
        rows.append((r["name"], group, r.get("setup", ""), m["tier_accuracy"], [m.get(f"recall_{t}") for t in TIERS],
                     m.get("too_small"), m.get("too_big"), [int(x) for x in m["picks"].split("/")],
                     round(m["latency_ms_p50"]) if m.get("latency_ms_p50") else None, "no"))
    rows.sort(key=lambda x: -x[3])
    lines = ["", "## 🧭 Jev, CLM, d3-flash and Laya on the same 300 set (not retrained)", "",
             "| Model | Kind | How it picks | **Tier acc.** | Recall low / mid / high | Too small | Too big | Picks low / mid / high | p50 latency | Threshold tuned on this set |",
             "|---|---|---|---:|---|---:|---:|---|---:|---|"]
    for name, group, how, acc, rec, small, big, picks, lat, tuned in rows:
        lines.append(f"| **{name}** | {group} | {how} | {acc}% | {' / '.join(f'{x:g}%' for x in rec)} | {small}% | {big}% | "
                     f"{'/'.join(map(str, picks))} | {'—' if lat is None else f'{lat} ms'} | {tuned} |")
    lines += ["", "> [!NOTE]",
              "> - **d3-flash direct:** vllm-sr/d3-flash (9B Qwen3.5 decision model) asked Jev's exact questions, zero-shot, on the Mac GPU without its ROCm fast kernels.",
              "> - **CLM direct:** CLM asked Jev's exact questions; Qwen3-8B run in PyTorch on the Mac, not official vLLM serving (see the CLM section).",
              "> - **Jev direct:** from `jev/OWN_300_REPORT.md`. The 66% rows send low-confidence picks to high, with the threshold chosen by sweeping on these same 300 questions, so they're optimistic.",
              "> - **Through a PR:** the PR's 1–5 complexity mapped to tiers.",
              "> - **Laya direct:** asked for the tier itself. The `tuned-*` checkpoints were fine-tuned on xRouteBench, not on our sets."]
    return lines
