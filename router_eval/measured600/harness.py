"""Shared harness: test tier classifiers on the measured 600 set.

Test set: data/llm_tier_selection_600_measured.jsonl. Its labels are measured, not guessed: the
cheapest tier whose models all answered correctly, as graded by a blind judge (../tier_measure/).
Train set (for fine-tuned models): ../data/eval_600.json, the same 600 questions every fine-tuned
model in TIER_SELECTION_RESULTS.md was trained on. The two sets share no questions.

A test script either passes a model with fit(queries, contexts, labels) / predict(queries, contexts)
to evaluate_model(), or passes its own predictions to save(). Both write
    results/<name>.json              metrics, setup and notes
    results/predictions/<name>.jsonl one line per test question: label, pick, latency
and rebuild RESULTS.md. Console output of each script goes to results/logs/ (see run_all.sh).
"""
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS, PREDS, DOC = HERE / "results", HERE / "results" / "predictions", HERE / "RESULTS.md"
TEST = HERE / "data" / "llm_tier_selection_600_measured.jsonl"
TRAIN = HERE.parent / "data" / "eval_600.json"
TIERS = ["low", "mid", "high"]
# Parameters per model, counted from the weight files (details in sizes.md); matched by name prefix
SIZES = [("d3-flash", "8.39 B"), ("clm", "8.19 B"), ("jev", "not published"), ("laya-multilingual", "322 M"),
         ("laya-", "421 M"), ("pr-296-semantic", "33.4 M"), ("pr-292", "22.7 M"), ("pr-296-lexical", "< 1 M (no network)"),
         ("nasiko", "0 (rules)")]


def size_of(name):
    return next((v for k, v in SIZES if name.startswith(k)), "—")


def test_set():
    rows = [json.loads(l) for l in TEST.read_text().splitlines() if l.strip()]
    return [r["query"] for r in rows], [r["label"].lower() for r in rows]


def train_set():
    ex = json.loads(TRAIN.read_text())["examples"]
    return [e["query"] for e in ex], ["" for _ in ex], [e["label"] for e in ex]  # question text only, as before


def score(labels, picks, latencies_ms=None):
    n = len(labels)
    rank = {t: i for i, t in enumerate(TIERS)}
    recall = {t: 100 * sum(p == t for p, l in zip(picks, labels) if l == t) / max(1, labels.count(t)) for t in TIERS}
    m = {"n": n, "accuracy": round(100 * sum(p == l for p, l in zip(picks, labels)) / n, 1),
         "balanced_accuracy": round(sum(recall.values()) / 3, 1),  # from the unrounded recalls
         **{f"recall_{t}": round(recall[t], 1) for t in TIERS},
         "too_small": round(100 * sum(rank[p] < rank[l] for p, l in zip(picks, labels)) / n, 1),
         "too_big": round(100 * sum(rank[p] > rank[l] for p, l in zip(picks, labels)) / n, 1),
         "picks": "/".join(str(picks.count(t)) for t in TIERS)}
    if latencies_ms:
        m["latency_ms_p50"] = round(statistics.median(latencies_ms), 1)
    return m


def save(name, kind, setup, notes, picks, latencies_ms=None, extra=None, train_seconds=None):
    """Score one model's picks on the test set, write its files and rebuild RESULTS.md."""
    queries, labels = test_set()
    assert len(picks) == len(labels) and all(p in TIERS for p in picks), f"{name}: bad picks"
    PREDS.mkdir(parents=True, exist_ok=True)
    with (PREDS / f"{name}.jsonl").open("w") as f:
        for i, (q, l, p) in enumerate(zip(queries, labels, picks)):
            row = {"query": q, "label": l, "pick": p}
            if latencies_ms:
                row["latency_ms"] = round(latencies_ms[i], 2)
            if extra:
                row.update(extra[i])
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    m = score(labels, picks, latencies_ms)
    if train_seconds is not None:
        m["train_seconds"] = round(train_seconds, 1)
    (RESULTS / f"{name}.json").write_text(json.dumps({"name": name, "kind": kind, "setup": setup, "notes": notes,
                                                      "metrics": m, "scored_at": datetime.now().isoformat(timespec="seconds")},
                                                     indent=2) + "\n")
    print(f"{name}: accuracy {m['accuracy']}% | balanced {m['balanced_accuracy']}% | recall low/mid/high "
          f"{m['recall_low']}/{m['recall_mid']}/{m['recall_high']}% | picks {m['picks']}", flush=True)
    report()
    return m


def evaluate_model(name, kind, setup, notes, model, train=True, proba=None):
    """Fit on the 600 training set (unless train=False), then classify each test question one at a time.

    proba(queries) -> [{tier: probability}], if given, is used instead of model.predict: the pick is the most likely
    tier (what each model's predict() returns) and the probabilities are kept for the threshold tables."""
    t0 = time.time()
    if train:
        model.fit(*train_set())
    else:
        model.fit([], [], [])  # loads the model only
    train_s = time.time() - t0
    queries, _ = test_set()
    model.predict(queries[:1], [""])  # warm up
    picks, lat, extra = [], [], []
    for k, q in enumerate(queries):
        s = time.perf_counter()
        if proba:
            p = {t: float(v) for t, v in proba([q])[0].items()}
            picks.append(max(TIERS, key=p.get))
            extra.append({"confidence": max(p.values()), "probabilities": p})
        else:
            picks.append(model.predict([q], [""])[0])
        lat.append((time.perf_counter() - s) * 1000)
        if k % 100 == 0:
            print(f"  {name}: {k}/{len(queries)}", flush=True)
    return save(name, kind, setup, notes, picks, lat, extra=extra or None, train_seconds=train_s if train else None)


def report():
    recs = sorted((json.loads(p.read_text()) for p in RESULTS.glob("*.json")), key=lambda r: -r["metrics"]["balanced_accuracy"])
    _, labels = test_set()
    counts = {t: labels.count(t) for t in TIERS}
    best = max((r["metrics"]["balanced_accuracy"] for r in recs), default=0)
    thr_file = RESULTS / "thresholds" / "best.json"
    best_thr = json.loads(thr_file.read_text()) if thr_file.exists() else {}
    lines = ["# Tier classifiers on the measured 600 set", "",
             f"> **TL;DR** — {len(recs)} classifier runs tested on 600 questions whose tiers were **measured** with real models "
             f"(LOW {counts['low']} / MID {counts['mid']} / HIGH {counts['high']}). The set is unbalanced, so models are ranked by "
             f"**balanced accuracy** (mean recall over the three tiers; chance = 33.3%). Always answering LOW gets "
             f"{round(100 * counts['low'] / len(labels), 1)}% plain accuracy but only 33.3% balanced. Best balanced accuracy: **{best}%**. "
             "Terms are explained in the [Glossary](#-glossary) at the end.", "",
             f"*Last updated: {datetime.now().strftime('%d %b %Y %H:%M')}*", ""]
    if (HERE / "findings.md").exists():
        lines += (HERE / "findings.md").read_text().rstrip().split("\n") + [""]
    lines += ["## 📊 Results", "",
              "| Model | Kind | Params | **Balanced acc.** | Accuracy | Recall low / mid / high | Too small | Too big | Picks low / mid / high | p50 latency |",
              "|---|---|---:|---:|---:|---|---:|---:|---|---:|",
              f"| *Always LOW (baseline)* | — | — | 33.3% | {round(100 * counts['low'] / len(labels), 1)}% | 100 / 0 / 0% | "
              f"{round(100 * (counts['mid'] + counts['high']) / len(labels), 1)}% | 0.0% | {len(labels)}/0/0 | — |"]
    top = recs[0] if recs else None
    cv_best = max(((n, t["cv"]["metrics"]["balanced_accuracy"]) for n, t in best_thr.items() if "cv" in t), key=lambda x: x[1], default=None)
    cv_top = cv_best[0] if cv_best and top and cv_best[1] > top["metrics"]["balanced_accuracy"] else None
    if top:  # the best model, called out above the table and highlighted in it
        tm = top["metrics"]
        i = lines.index("## 📊 Results") + 2
        lines[i:i] = ["> [!TIP]",
                      f"> **Best model: `{top['name']}`**: {tm['balanced_accuracy']}% balanced accuracy "
                      f"({tm['accuracy']}% accuracy, recall low / mid / high {tm['recall_low']:g} / {tm['recall_mid']:g} / "
                      f"{tm['recall_high']:g}%), as answered."]
        if cv_top:
            cm = best_thr[cv_top]["cv"]["metrics"]
            lines[i + 2:i + 2] = [f"> **Best with a threshold chosen by cross-validation: `{cv_top}`**: {cm['balanced_accuracy']}% balanced "
                                  f"({cm['accuracy']}% accuracy). The two are a statistical tie (see Findings). Both are highlighted."]
        lines[i + 2 + bool(cv_top):i + 2 + bool(cv_top)] = [""]
    for r in recs:
        m = r["metrics"]
        p50 = m.get("latency_ms_p50")
        lat = "—" if p50 is None else "<0.1 ms" if p50 < 0.1 else f"{p50:g} ms"
        cells = [r["kind"], size_of(r["name"]), f"{m['balanced_accuracy']}%", f"{m['accuracy']}%",
                 f"{m['recall_low']:g} / {m['recall_mid']:g} / {m['recall_high']:g}%", f"{m['too_small']}%",
                 f"{m['too_big']}%", m["picks"], lat]
        if r is top:  # the whole row in bold, the name in green
            lines.append(f'| <span style="color:#1a7f37">**{r["name"]}**</span> (best) | ' + " | ".join(f"**{c}**" for c in cells) + " |")
        else:
            lines.append(f"| **{r['name']}** | " + " | ".join(cells) + " |")
        t = best_thr.get(r["name"])
        if t and "cv" in t:
            c = t["cv"]["metrics"]
            cells = [f"{c['balanced_accuracy']}%", f"{c['accuracy']}%",
                     f"{c['recall_low']:g} / {c['recall_mid']:g} / {c['recall_high']:g}%", f"{c['too_small']}%", f"{c['too_big']}%", c["picks"]]
            if r["name"] == cv_top:  # best honest result with a threshold: highlighted like the best model
                lines.append(f'| ↳ <span style="color:#1a7f37">**threshold chosen by cross-validation (honest)**</span> (tied best) | '
                             f"**5-fold, chosen on the other folds** | **{size_of(r['name'])}** | " + " | ".join(f"**{x}**" for x in cells) + f" | **{lat}** |")
            else:
                lines.append(f"| ↳ *threshold chosen by cross-validation (honest)* | *5-fold, chosen on the other folds* | "
                             f"{size_of(r['name'])} | " + " | ".join(f"*{x}*" for x in cells) + f" | {lat} |")
        if t and (t["rule"], t["threshold"]) != ("As answered", "—"):
            b = t["metrics"]
            rule = f"confidence < {t['threshold']} → high" if t["rule"].startswith("Confidence") else f"cascade {t['threshold']}"
            lines.append(f"| ↳ *best threshold on these labels: {rule}* | *optimistic: picked with the test labels* | {size_of(r['name'])} | "
                         f"*{b['balanced_accuracy']}%* | *{b['accuracy']}%* | *{b['recall_low']:g} / {b['recall_mid']:g} / "
                         f"{b['recall_high']:g}%* | *{b['too_small']}%* | *{b['too_big']}%* | *{b['picks']}* | {lat} |")
        elif t:
            lines.append(f"| ↳ *best threshold on these labels: none beats the plain pick* | — | | | | | | | | |")
    if (HERE / "sizes.md").exists():
        lines += [""] + (HERE / "sizes.md").read_text().rstrip().split("\n")
    if (HERE / "thresholds.md").exists():
        lines += [""] + (HERE / "thresholds.md").read_text().rstrip().split("\n")
    lines += ["", "## ⚙️ Setup per model", "", "| Model | Setup | Notes |", "|---|---|---|"]
    lines += [f"| **{r['name']}** | {r['setup']} | {r['notes']}"
              f"{'' if r['metrics'].get('train_seconds') is None else ' Train ' + str(r['metrics']['train_seconds']) + ' s.'} |"
              for r in recs]
    lines += ["", "## 🧪 How it was tested", "",
              "- **Test set:** `data/llm_tier_selection_600_measured.jsonl`, 600 questions. Each label is the cheapest tier whose "
              "models all answered correctly (LOW: `gpt-4.1-nano` + `Qwen3.5 9B`; MID: `gpt-4.1` + `Gemma 4 26B-A4B`; HIGH: "
              "`gpt-5.4`), graded by a blind judge. The 11 questions no tier solved are labelled HIGH. See `../tier_measure/RESULTS.md`.",
              "- **Fine-tuned models** are trained on `../data/eval_600.json` (authored labels, 200 per tier), the same training data "
              "as in `TIER_SELECTION_RESULTS.md`, and tested once here. The two sets share no questions.",
              "- **Zero-shot decision models** (Jev, d3-flash, CLM, Laya) are asked Jev's low / mid / high question "
              "(`jev/run_jev.py` `LEVEL_QUESTION`), and their top option is the pick.",
              "- **Balanced accuracy** = mean of the three recalls. **Too small** = picked a cheaper tier than measured (answer at "
              "risk); **too big** = picked a dearer tier (money wasted).",
              "- **Latency:** one question at a time on this Mac (M5 Pro), or the API round trip for Jev.",
              "- **Files:** `results/<model>.json` (metrics), `results/predictions/<model>.jsonl` (every pick), "
              "`results/logs/` (console log of every run)."]
    if (HERE / "glossary.md").exists():
        lines += ["", "---", ""] + (HERE / "glossary.md").read_text().rstrip().split("\n")
    DOC.write_text("\n".join(lines) + "\n")


def rescore():
    """Recompute every stored result's metrics from its predictions file, keeping latency and train time."""
    _, labels = test_set()
    for f in RESULTS.glob("*.json"):
        r = json.loads(f.read_text())
        rows = [json.loads(l) for l in (PREDS / f"{r['name']}.jsonl").read_text().splitlines()]
        lat = [x["latency_ms"] for x in rows] if "latency_ms" in rows[0] else None
        m = score(labels, [x["pick"] for x in rows], lat)
        if "train_seconds" in r["metrics"]:
            m["train_seconds"] = r["metrics"]["train_seconds"]
        r["metrics"] = m
        f.write_text(json.dumps(r, indent=2) + "\n")


if __name__ == "__main__":
    if "--rescore" in sys.argv:
        rescore()
    report()
    print(f"rebuilt {DOC.name} from {len(list(RESULTS.glob('*.json')))} results")
