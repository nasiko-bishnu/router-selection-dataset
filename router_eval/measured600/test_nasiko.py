# /// script
# requires-python = ">=3.10"
# ///
"""Nasiko's current router (Nasiko-Labs/nasiko main, llm-router) on the measured 600.

main has no difficulty classifier: it classifies the request type with regexes, then Thompson-samples a tier
over (tier, request type) cells, cold-start priors plus whatever feedback it has learned. This runs main's
own Rust code (routing::classify) through nasiko_router_eval.rs, built in a separate nasiko clone, in two modes:
  cold   : no learned cells (a fresh deployment)
  online : the chosen cell learns from the measured label after every pick (best-case, instant feedback)
Thompson sampling is random, so each mode runs with seeds 0-4; the table shows seed 0, the notes the spread.

    uv run router_eval/measured600/test_nasiko.py      # NASIKO_CLONE / CARGO_TARGET_DIR override the defaults
"""
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import TEST, save, score, test_set  # noqa: E402

WORK = Path(os.environ.get("PRTEST_DIR", Path.home() / ".cache" / "router-selection-dataset"))
CLONE = Path(os.environ.get("NASIKO_CLONE", WORK / "nasiko"))
TARGET = Path(os.environ.get("CARGO_TARGET_DIR", WORK / "target"))
SEEDS = range(5)


def sh(*cmd, **kw):
    r = subprocess.run(cmd, cwd=CLONE, text=True, capture_output=True, **kw)
    if r.returncode:
        sys.exit(f"{' '.join(cmd)} failed: {r.stderr[-800:]}")
    return r.stdout.strip()


def main():
    if not CLONE.exists():  # first run: a separate clone, so no nasiko checkout of yours is touched
        WORK.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "https://github.com/Nasiko-Labs/nasiko.git", str(CLONE)], check=True)
    sh("git", "fetch", "-q", "origin", "main")
    sh("git", "checkout", "-q", "-f", "--detach", "origin/main")
    commit = sh("git", "log", "-1", "--format=%h %cd", "--date=short")
    print(f"nasiko main at {commit}", flush=True)
    shutil.copy(HERE / "nasiko_router_eval.rs", CLONE / "llm-router" / "examples" / "nasiko_router_eval.rs")
    sh("cargo", "build", "-q", "--release", "-p", "nasiko-llm-router", "--example", "nasiko_router_eval",
       env={**os.environ, "CARGO_TARGET_DIR": str(TARGET)})
    binary = TARGET / "release" / "examples" / "nasiko_router_eval"
    _, labels = test_set()
    out_dir = HERE / "results" / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    for mode in ("cold", "online"):
        runs = {}
        for seed in SEEDS:
            out = out_dir / f"nasiko_{mode}_seed{seed}.jsonl"
            subprocess.run([str(binary)], env={**os.environ, "EVAL_SET": str(TEST), "OUT": str(out), "SEED": str(seed), "MODE": mode},
                           check=True)
            runs[seed] = [json.loads(l) for l in out.read_text().splitlines()]
        bal = [score(labels, [r["tier"] for r in runs[s]])["balanced_accuracy"] for s in SEEDS]
        acc = [score(labels, [r["tier"] for r in runs[s]])["accuracy"] for s in SEEDS]
        spread = (f"Seeds 0-4: balanced {statistics.mean(bal):.1f}% (sd {statistics.stdev(bal):.1f}), "
                  f"accuracy {statistics.mean(acc):.1f}% (sd {statistics.stdev(acc):.1f}).")
        print(f"nasiko-main-{mode}: {spread}", flush=True)
        how = ("no learned cells, cold-start priors only (a fresh deployment)" if mode == "cold" else
               "after each pick the chosen cell learns from the measured label (1 if the tier is at or above it): "
               "best-case instant feedback")
        rows = runs[0]
        save(f"nasiko-main-{mode}", "Nasiko router (main)", f"regex request type + Thompson-sampled tier, {how}",
             f"main's own routing::classify (Rust, commit {commit}), seed 0 shown. {spread} No difficulty signal: the tier "
             "depends only on the request type and the sampled priors.",
             [r["tier"] for r in rows], [r["latency_us"] / 1000 for r in rows],
             extra=[{"request_type": r["request_type"]} for r in rows])


if __name__ == "__main__":
    main()
