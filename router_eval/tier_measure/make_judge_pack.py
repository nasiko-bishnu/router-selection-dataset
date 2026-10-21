"""Build the blind judge pack from every model's answers.

Each answer becomes an item with a random id; the model's name is removed and the items are shuffled,
so the judge cannot tell which model (or tier) wrote what. The mapping goes to
~/.cache/xroutebench/judge_key.json, outside the repo, so a judge session working here cannot see it. Batches of 50 items go to judge_pack/batch_NN.jsonl, next to the
judging instructions (JUDGE_INSTRUCTIONS.md).

    python3 router_eval/tier_measure/make_judge_pack.py
"""
import json
import random
import secrets
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ANSWERS, HERE, KEY, PACK, candidates  # noqa: E402

BATCH = 50


def main():
    cands = {r["id"]: r for r in candidates()}
    items, key = [], {}
    for f in sorted(ANSWERS.glob("*.jsonl")):
        for a in map(json.loads, f.read_text().splitlines()):
            item_id = secrets.token_hex(5)
            key[item_id] = {"question_id": a["id"], "model": a["model"]}
            c = cands[a["id"]]
            item = {"item_id": item_id, "question": c["query"], "answer": a["answer"]}
            if c.get("reference_answer"):
                item["reference_answer"] = c["reference_answer"]
            items.append(item)
    random.Random(secrets.randbits(32)).shuffle(items)
    if PACK.exists():
        shutil.rmtree(PACK)
    PACK.mkdir(parents=True)
    for n in range(0, len(items), BATCH):
        (PACK / f"batch_{n // BATCH + 1:02d}.jsonl").write_text(
            "".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items[n:n + BATCH]))
    shutil.copy(HERE / "JUDGE_INSTRUCTIONS.md", PACK / "JUDGE_INSTRUCTIONS.md")
    KEY.parent.mkdir(parents=True, exist_ok=True)
    KEY.write_text(json.dumps(key, indent=1) + "\n")
    print(f"{len(items)} items in {len(list(PACK.glob('batch_*.jsonl')))} batches under {PACK.name}/ | key: {KEY.name} (keep it from the judge)")


if __name__ == "__main__":
    main()
