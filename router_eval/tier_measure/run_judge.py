"""Grade the judge pack with Claude, one fresh, tool-less session per batch.

Each batch goes to `claude -p` in the prompt itself: no tools (it cannot open answers/, the key, the web or a
shell), safe mode (no CLAUDE.md, memory, hooks or MCP servers) and no saved session, so batches never see each
other. The reply must hold exactly one grade line per item; a batch whose reply doesn't is retried, and batches
already graded are skipped, so a rerun carries on after a stop or a rate limit.

    python3 router_eval/tier_measure/run_judge.py --model claude-sonnet-5-5 --batches 1     # trial: first batch
    python3 router_eval/tier_measure/run_judge.py --model claude-sonnet-5-5                 # the rest
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import GRADES, PACK  # noqa: E402

RUN_NOTE = """

---

## This run

You have no tools in this session. The batch is in the message, one JSON item per line, instead of a file.
Instead of writing a grades file, reply with **only** the grade lines: one JSON object per line, one line per
item, with exactly the fields `item_id`, `pass`, `score`, `reason`, and nothing before or after them."""


def grade_batch(batch, model, tries=3):
    items = [json.loads(l) for l in batch.read_text().splitlines()]
    want = {i["item_id"] for i in items}
    system = (PACK / "JUDGE_INSTRUCTIONS.md").read_text() + RUN_NOTE
    grades, raws = {}, []
    for attempt in range(1, tries + 1):
        # a retry sends only the items still without a grade, in a fresh session; grades already returned are kept
        todo = "".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items if i["item_id"] not in grades)
        r = subprocess.run(["claude", "-p", "--model", model, "--tools", "", "--safe-mode", "--no-session-persistence",
                            "--system-prompt", system], input=todo, capture_output=True, text=True,
                           cwd=PACK, timeout=1800)
        raw = r.stdout
        raws.append(raw)
        for line in raw.splitlines():
            line = line.strip().strip("`")
            if line.startswith("{"):
                try:
                    g = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if g.get("item_id") in want and g["item_id"] not in grades and isinstance(g.get("pass"), bool):
                    grades[g["item_id"]] = {k: g.get(k) for k in ("item_id", "pass", "score", "reason")}
        raw = "\n\n--- retry ---\n\n".join(raws)
        if set(grades) == want:
            return [grades[i["item_id"]] for i in items], raw
        print(f"  {batch.name} try {attempt}: {len(grades)}/{len(want)} grades (exit {r.returncode}) {r.stderr.strip()[:200]}",
              flush=True)
        if "limit" in (r.stderr + raw).lower():
            raise SystemExit("usage limit reached: rerun later, finished batches are kept")
        time.sleep(10)
    return None, raw


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--batches", type=int, help="grade at most this many (not yet graded) batches")
    a = ap.parse_args()
    (GRADES / "raw").mkdir(parents=True, exist_ok=True)
    todo = [b for b in sorted(PACK.glob("batch_*.jsonl")) if not (GRADES / b.name).exists()][: a.batches]
    print(f"{len(todo)} batches to grade with {a.model}", flush=True)
    for b in todo:
        t0 = time.time()
        grades, raw = grade_batch(b, a.model)
        (GRADES / "raw" / f"{b.stem}.txt").write_text(raw)  # the judge's full reply, kept for review
        if grades is None:
            print(f"{b.name}: FAILED after retries, see grades/raw/{b.stem}.txt", flush=True)
            continue
        (GRADES / b.name).write_text("".join(json.dumps(g, ensure_ascii=False) + "\n" for g in grades))
        n_pass = sum(g["pass"] for g in grades)
        print(f"{b.name}: {n_pass}/{len(grades)} pass, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
