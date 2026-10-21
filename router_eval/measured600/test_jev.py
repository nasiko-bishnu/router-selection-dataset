# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas", "pyarrow", "numpy", "requests", "huggingface_hub", "laya", "tabulate"]
# ///
"""Jev (TypeSafe API, hosted) asked the low / mid / high question for every question of the measured 600.

Same request as jev/run_jev.py's default setup: LEVEL_QUESTION, state {"request": question}, pinned model
jev-1.13.0. Jev has no fine-tuned version, so this is zero-shot only. The key is read from jev/.env
(TYPESAFE_API_KEY, never printed). Answers are cached in results/cache/jev.jsonl, so a rerun only pays
for what's missing.

    uv run router_eval/measured600/test_jev.py
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "jev"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from harness import save, test_set  # noqa: E402
from run_jev import LEVEL_QUESTION, PRICE_IN, PRICE_OUT, URL, call, load_key, record  # noqa: E402
from bench import state_for  # noqa: E402

MODEL = "jev-1.13.0"
CACHE = HERE / "results" / "cache" / "jev.jsonl"


def main():
    import requests
    queries, _ = test_set()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["query"]: json.loads(l) for l in CACHE.read_text().splitlines()} if CACHE.exists() else {}
    todo = [q for q in dict.fromkeys(queries) if q not in done]
    key = load_key()
    if todo and not key:
        raise SystemExit("TYPESAFE_API_KEY not set in jev/.env")
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "xroutebench-tier/jev"})

    def ask(q):
        j, timing = call(s, URL, {"model": MODEL, "state": state_for(q), "questions": LEVEL_QUESTION}, 60)
        d = record(j, timing, PRICE_IN, PRICE_OUT)
        return {"query": q, "choice": d["choice"], "confidence": d["confidence"], "probabilities": j["answers"]["tier"]["probabilities"],
                "latency_ms": d["latency_ms"], "cost": d["cost"], "model": d["model"]}

    with CACHE.open("a") as f, ThreadPoolExecutor(4) as pool:
        for k, r in enumerate(pool.map(ask, todo), 1):
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            done[r["query"]] = r
            if k % 100 == 0:
                print(f"  jev: {k}/{len(todo)} new answers", flush=True)
    rows = [done[q] for q in queries]
    cost = sum(r["cost"] for r in rows)
    print(f"jev: {len(rows)} answers, models {sorted({r['model'] for r in rows})}, cost ${cost:.4f}", flush=True)
    save("jev-direct", "Decision model, zero-shot (hosted)", f"Jev ({MODEL}) asked low / mid / high, top option",
         f"TypeSafe API, LEVEL_QUESTION, 4 parallel requests; latency is the API round trip. Total cost ${cost:.4f}.",
         [r["choice"] for r in rows], [r["latency_ms"] for r in rows],
         extra=[{"confidence": r["confidence"], "probabilities": r["probabilities"]} for r in rows])


if __name__ == "__main__":
    main()
