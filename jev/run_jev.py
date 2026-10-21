# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas", "pyarrow", "numpy", "requests", "huggingface_hub"]
# ///
"""Ask TypeSafe Jev (direct API) to pick a tier for every xRouteBench test question.

Same questions, same T1/T2/T3 prompt setups and same scoring as bench.py, so Jev lines up with
Laya, OpenJev and nasiko. Every call is logged with its tokens, cost, latency, retries and
confidence; analyze_jev.py turns the log into jev/results/report_<tag>_<n>_<seed>.md.

Key: TYPESAFE_API_KEY=sk-... in jev/.env (or the repo's .env). Both are gitignored.

    uv run jev/run_jev.py --dry-run                 # print the exact request; no call, no key
    uv run jev/run_jev.py --probe 5                 # 5 real calls: check key, format, cost, latency
    uv run jev/run_jev.py --n 0                     # all 3,729 test questions, run 3 setup
    uv run jev/run_jev.py --n 0 --setup withheader  # run 2 (--setup sizelabels = run 1)
    uv run jev/analyze_jev.py --n 0                 # the report

A run is resumable: answers are appended to a .partial.jsonl as they arrive and a re-run skips them.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

JEV_DIR = Path(__file__).resolve().parent
REPO = JEV_DIR.parent
sys.path.insert(0, str(REPO))
from bench import CHOICE_TO_TIER, LAYA_SETUPS, load_queries, state_for, truth_fields  # noqa: E402

# Jev's default question: options named high / mid / low, described by capability only (no model
# sizes, no task examples, no "multiple choice" wording), so the option text doesn't hint at a format.
LEVEL_QUESTION = {
    "tier": {
        "type": "choice",
        "instructions": "Select the lowest model tier that is sufficient to answer the request correctly. "
        "Choose the cheapest tier that can reliably solve the task. The available tiers are low, mid, and "
        "high, where low is the smallest and least expensive model, mid is a general-purpose model with "
        "stronger capabilities, and high is the largest and most expensive frontier model.",
        "criteria": {
            "high": "Frontier, state-of-the-art models with the strongest reasoning and knowledge capabilities. "
            "Required for highly complex, difficult, ambiguous, or specialized tasks that lower tiers may not "
            "reliably solve.",
            "mid": "General-purpose models with stronger reasoning and knowledge capabilities than low-tier "
            "models. Suitable for moderately complex questions, multi-step reasoning, ambiguity, and tasks "
            "where a small model may be unreliable.",
            "low": "Small, inexpensive models with basic reasoning and knowledge capabilities. Sufficient for "
            "simple, clear, well-defined questions that a small model can reliably answer correctly.",
        },
    }
}
# Three independent yes/no (noul) questions, one per tier; the pick is the cheapest tier whose
# P(yes) reaches the cut-off, else high. Same capability wording as LEVEL_QUESTION.
YESNO_QUESTION = {
    "low": {"type": "noul", "instructions": "A small, inexpensive model with basic reasoning and knowledge "
            "capabilities would answer this request correctly."},
    "mid": {"type": "noul", "instructions": "A general-purpose model with stronger reasoning and knowledge "
            "capabilities than a small model would answer this request correctly."},
    "high": {"type": "noul", "instructions": "A frontier, state-of-the-art model with the strongest reasoning "
             "and knowledge capabilities would answer this request correctly."},
}
YESNO_CUTOFF = 0.5  # the pick stored per call; analyze_yesno.py sweeps other cut-offs from the saved P(yes)


def cascade(p_yes: dict, cut: float) -> str:
    """Cheapest tier whose P(yes) >= cut: T3 (low), then T2 (mid), else T1 (high)."""
    return "T3" if p_yes["T3"] >= cut else "T2" if p_yes["T2"] >= cut else "T1"


# setup -> (question, keep the task's fixed instruction header). '' is the default.
#   yesno       YESNO_QUESTION: P(yes) per tier, cheapest yes wins, header removed
#   ''          high/mid/low options, header removed
#   tiers       bench.py's run 3: T1/T2/T3 options, header removed (same input as Laya and OpenJev)
#   withheader  run 2: T1/T2/T3, full question text
#   sizelabels  run 1: large/medium/small, full question text
SETUPS = {"": (LEVEL_QUESTION, False), "yesno": (YESNO_QUESTION, False), "tiers": LAYA_SETUPS[""],
          "withheader": LAYA_SETUPS["withheader"], "sizelabels": LAYA_SETUPS["sizelabels"]}
LABEL_TO_TIER = {**CHOICE_TO_TIER, "high": "T1", "mid": "T2", "low": "T3"}

URL = "https://api.typesafe.ai/v1/systemone"
PRICE_IN, PRICE_OUT = 0.042, 0.0  # $ per million tokens: TypeSafe list price, output free
TEST_SIZE = 3729
# Response headers worth keeping per call (request ids, rate limits, any server-side timing)
KEEP_HEADERS = ("request-id", "ratelimit", "retry-after", "timing", "processing", "latency", "response-time")


class FatalHTTP(RuntimeError):
    """A 4xx other than 429: retrying won't help (bad key, bad request)."""


def load_key() -> str:
    for env in (JEV_DIR / ".env", REPO / ".env"):
        if env.exists():
            for line in env.read_text().splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))
    return os.environ.get("TYPESAFE_API_KEY", "").strip()


def tag_for(setup: str) -> str:
    return "jev" + (f"-{setup}" if setup else "")


def call(session, url: str, body: dict, timeout: float) -> tuple[dict, dict]:
    """One decision, retrying 429 / 5xx / network errors. Returns (response JSON, timing record)."""
    import requests

    t_start, errors = time.perf_counter(), []
    for attempt in range(8):
        wait = min(30, 2 ** attempt)
        t0 = time.perf_counter()
        try:
            r = session.post(url, json=body, timeout=timeout)
        except requests.RequestException as e:
            errors.append(type(e).__name__)
        else:
            ms = (time.perf_counter() - t0) * 1000
            if r.ok:
                headers = {k.lower(): v for k, v in r.headers.items() if any(s in k.lower() for s in KEEP_HEADERS)}
                return r.json(), {"latency_ms": ms, "total_ms": (time.perf_counter() - t_start) * 1000,
                                  "attempts": attempt + 1, "errors": errors, "headers": headers}
            if r.status_code != 429 and r.status_code < 500:
                raise FatalHTTP(f"HTTP {r.status_code}: {r.text[:500]}")
            errors.append(str(r.status_code))
            wait = max(wait, float(r.headers.get("retry-after") or 0))
        time.sleep(wait)
    raise RuntimeError(f"gave up after 8 attempts: {errors}")


def record(j: dict, timing: dict, price_in: float, price_out: float) -> dict:
    """The fields kept per decision. Probabilities are keyed T1/T2/T3 whatever the option labels.
    A yes/no request keeps P(yes) per tier (p_yes) and picks by cascade at YESNO_CUTOFF."""
    answers = j["answers"]
    if "tier" in answers:
        ans = answers["tier"]
        pick = {"tier": LABEL_TO_TIER[ans["choice"]], "choice": ans["choice"],
                "probabilities": {LABEL_TO_TIER[k]: float(v) for k, v in ans["probabilities"].items()},
                "confidence": ans.get("confidence")}
    else:
        p_yes = {LABEL_TO_TIER[k]: float(v["noul"]) for k, v in answers.items()}
        pick = {"tier": cascade(p_yes, YESNO_CUTOFF), "choice": None, "p_yes": p_yes,
                "probabilities": None, "confidence": None}
    u = j.get("usage") or {}
    tin, tout = int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0)
    billed = u.get("cost")  # use the API's own figure if it ever sends one
    return {**pick, "model": j.get("model"), "request_id": j.get("id"),
            "input_tokens": tin, "output_tokens": tout,
            "cost": float(billed) if billed is not None else tin * price_in / 1e6 + tout * price_out / 1e6,
            "cost_source": "api" if billed is not None else "list price",
            "at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), **timing}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=0, help="test questions (0 = all 3,729; 500 = the 507-question sample)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--setup", choices=list(SETUPS), default="",
                    help="'' = high/mid/low options (default); tiers / withheader / sizelabels = bench.py's runs 3 / 2 / 1")
    ap.add_argument("--model", default="jev-1.13.0", help="pinned version by default; jev-latest moves")
    ap.add_argument("--workers", type=int, default=4, help="parallel requests (latency is measured per call)")
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--url", default=URL)
    ap.add_argument("--price-in", type=float, default=PRICE_IN, help="$ per million input tokens")
    ap.add_argument("--price-out", type=float, default=PRICE_OUT, help="$ per million output tokens")
    ap.add_argument("--probe", type=int, default=0, help="only call this many questions, one at a time, and print each")
    ap.add_argument("--dry-run", action="store_true", help="print the request for the first question and exit")
    ap.add_argument("--data", default=None,
                    help="a question set from make_balanced_set.py (e.g. jev/data/balanced_300.jsonl) instead of --n")
    ap.add_argument("--out", default=str(JEV_DIR / "results"))
    args = ap.parse_args()

    q, suffix = load_queries(args.n, args.seed), f"{args.n}_{args.seed}"
    own = False  # True for our own labelled questions (no benchmark scores)
    if args.data:
        rows_in = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
        suffix = Path(args.data).stem
        if "scores" in rows_in[0]:  # test-split questions listed in the file, in its order
            q = load_queries(0, args.seed).loc[[r["qid"] for r in rows_in]]
        else:
            import pandas as pd

            own = True
            q = pd.DataFrame({"query": [r["question"] for r in rows_in], "task_name": [r["domain"] for r in rows_in]},
                             index=[r["qid"] for r in rows_in])
    question, keep_header = SETUPS[args.setup]
    body_for = lambda query: {"model": args.model, "state": state_for(query, keep_header), "questions": question}  # noqa: E731
    tag, out = tag_for(args.setup), Path(args.out)
    if args.dry_run:
        print(f"POST {args.url}\nAuthorization: Bearer $TYPESAFE_API_KEY\n")
        print(json.dumps(body_for(q.iloc[0]["query"]), indent=2, ensure_ascii=False))
        print(f"\n{len(q)} questions would be sent ({tag}, {suffix}).")
        return

    import requests

    key = load_key()
    if not key:
        raise SystemExit("TYPESAFE_API_KEY not set: put TYPESAFE_API_KEY=sk-... in jev/.env (gitignored)")
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                            "User-Agent": "xroutebench-tier/jev"})
    out.mkdir(parents=True, exist_ok=True)

    if args.probe:
        rows = []
        for qid, row in q.head(args.probe).iterrows():
            j, timing = call(session, args.url, body_for(row["query"]), args.timeout)
            if not rows:
                print("first raw response:\n" + json.dumps(j, indent=2) + "\n")
            d = record(j, timing, args.price_in, args.price_out)
            rows.append(d)
            print(f"{qid}: {d['tier']} conf {d['confidence']} probs {d.get('p_yes') or d['probabilities']} | model {d['model']} | "
                  f"{d['input_tokens']} in / {d['output_tokens']} out tokens, ${d['cost']:.8f} | "
                  f"{d['latency_ms']:.0f} ms, {d['attempts']} attempt(s) | headers {d['headers']}")
        mean_in = sum(d["input_tokens"] for d in rows) / len(rows)
        est = mean_in * TEST_SIZE * args.price_in / 1e6
        print(f"\nmean {mean_in:.0f} input tokens per call -> one full run of {TEST_SIZE:,} questions ~ ${est:.4f} "
              f"(~{TEST_SIZE * mean_in / 1e6:.2f}M tokens); mean latency "
              f"{sum(d['latency_ms'] for d in rows) / len(rows):.0f} ms")
        (out / f"probe_{tag}.jsonl").write_text("".join(json.dumps({"qid": i, **d}) + "\n" for i, d in zip(q.index, rows)))
        return

    stem = f"decisions_{tag}_{suffix}"
    final, partial, raw = out / f"{stem}.jsonl", out / f"{stem}.partial.jsonl", out / f"raw_{tag}_{suffix}.jsonl"
    if final.exists():
        raise SystemExit(f"{final.name} already exists; delete it to ask Jev again")
    done = {}
    if partial.exists():
        lines = partial.read_text().splitlines()
        try:
            json.loads(lines[-1]) if lines else None
        except json.JSONDecodeError:  # killed mid-write: drop the torn last line
            lines = lines[:-1]
            partial.write_text("".join(line + "\n" for line in lines))
        done = {d.pop("qid"): d for d in map(json.loads, lines)}
        print(f"resuming: {len(done)} answers already in {partial.name}")

    todo = [(qid, row["query"]) for qid, row in q.iterrows() if qid not in done]
    print(f"{tag}: {len(todo)} of {len(q)} questions to ask, model {args.model}, {args.workers} workers")

    def one(qid, query):
        j, timing = call(session, args.url, body_for(query), args.timeout)
        return qid, j, record(j, timing, args.price_in, args.price_out)

    started, t0 = datetime.now(timezone.utc), time.perf_counter()
    pool = ThreadPoolExecutor(args.workers)
    try:
        with partial.open("a") as log, raw.open("a") as rawlog:
            for f in as_completed([pool.submit(one, qid, text) for qid, text in todo]):
                qid, j, d = f.result()
                done[qid] = d
                log.write(json.dumps({"qid": qid, **d}) + "\n"); log.flush()
                rawlog.write(json.dumps({"qid": qid, "response": j, "headers": d["headers"]}) + "\n"); rawlog.flush()
                if len(done) % 100 == 0:
                    print(f"  {len(done)}/{len(q)}", flush=True)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    wall = time.perf_counter() - t0

    final.write_text("".join(json.dumps({"qid": i, **done[i], **({} if own else truth_fields(q.loc[i], done[i]["tier"]))}) + "\n"
                             for i in q.index))
    partial.unlink(missing_ok=True)
    info = {"tag": tag, "n": args.n, "seed": args.seed, "data": args.data, "setup": args.setup or "levels", "model_requested": args.model,
            "models_returned": sorted({str(d["model"]) for d in done.values()}), "url": args.url,
            "workers": args.workers, "price_in_per_m": args.price_in, "price_out_per_m": args.price_out,
            "questions": len(q), "asked_this_session": len(todo), "started": started.isoformat(timespec="seconds"),
            "wall_seconds_this_session": round(wall, 1),
            "throughput_per_s": round(len(todo) / wall, 2) if todo and wall else None}
    (out / f"run_{tag}_{suffix}.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"saved {final.name} | {json.dumps(info)}")


if __name__ == "__main__":
    main()
