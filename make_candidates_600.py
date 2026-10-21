# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn"]
# ///
"""Candidate pool for tier measurement: our 300 authored questions + 300 real prompts from oasst2.

The oasst2 prompts (OpenAssistant/oasst2, Apache-2.0) were written by many different people, so they
bring real phrasing, varied length and pasted text or code that the authored set lacks. English only.

Filters on oasst2's opening prompts: English, 6-150 words, review quality >= 0.6, not flagged as spam,
PII, inappropriate, hate or sexual content, low toxicity, no role-play or questions about the
assistant itself, no URLs / emails / phone numbers, nothing time-sensitive ("latest", "currently",
years, news), no references to text that isn't included, and no near-duplicates of the authored set.

The 300 oasst2 prompts are sampled with a fixed seed, spread across length bands, prompts with pasted
context (code, quotes, several lines) and subject domains (capped per domain so coding, oasst2's most
common topic, can't dominate). They have label null: tiers are to be measured with real models.

    uv run make_candidates_600.py   # -> llm_tier_selection_candidates_600.jsonl
"""
import gzip
import json
import random
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import make_300_balanced

HERE = Path(__file__).parent
AUTHORED = HERE / "llm_tier_selection_300_balanced.jsonl"
DST = HERE / "llm_tier_selection_candidates_600.jsonl"
OASST_URL = "https://huggingface.co/datasets/OpenAssistant/oasst2/resolve/main/2023-11-05_oasst2_prompts.messages.jsonl.gz"
CACHE = Path.home() / ".cache" / "xroutebench" / "oasst2_prompts.messages.jsonl.gz"
N_OASST, SEED = 300, 0
# Length bands (words) and how many prompts to take from each: more long prompts than the authored set has.
BANDS = [((6, 15), 75), ((16, 30), 90), ((31, 60), 85), ((61, 150), 50)]
MIN_CONTEXT = 80        # prompts with pasted code, quotes or several lines
DOMAIN_CAP = 35         # per subject domain ("other" included)

UNSAFE_OR_SELF = re.compile(r"\b(open ?assistant|chatgpt|you are an? |pretend|role ?play|act as|your name|who are you|"
                            r"are you (sentient|conscious|alive)|jailbreak|uwu|furry|nsfw|sex|porn|kill|suicide|nazi|hitler)\b", re.I)
TIME_SENSITIVE = re.compile(r"\b(latest|current|currently|recent|recently|this year|today|tonight|news|right now|nowadays|"
                            r"trending|stock price|20[12]\d)\b", re.I)
MISSING_TEXT = re.compile(r"\b(the (above|following|attached|previous)|this (article|text|passage|code|image|picture|link|video|pdf))\b", re.I)
PII = re.compile(r"[\w.+-]+@[\w-]+\.\w+|\b\d{3}[-. ]\d{3}[-. ]\d{4}\b|https?://\S+")
CONTEXT = re.compile(r"`|\n|\"[^\"]{15,}\"|'[^']{15,}'")


def load_oasst():
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(OASST_URL, CACHE)
    return [json.loads(l) for l in gzip.open(CACHE, "rt")]


def label_value(r, k):
    v = (r.get("labels") or {}).get(k)
    return v["value"] if v else 0.0


def clean(rows):
    keep, seen = [], set()
    for r in rows:
        t = r["text"].strip()
        n = len(t.split())
        if r.get("lang") != "en" or not 6 <= n <= 150 or label_value(r, "quality") < 0.6:
            continue
        if any(label_value(r, k) > 0 for k in ("spam", "pii", "not_appropriate", "hate_speech", "sexual_content", "lang_mismatch")):
            continue
        if (r.get("detoxify") or {}).get("toxicity", 0) > 0.1:
            continue
        if UNSAFE_OR_SELF.search(t) or TIME_SENSITIVE.search(t) or PII.search(t) or (MISSING_TEXT.search(t) and n < 40):
            continue
        key = re.sub(r"\W+", " ", t.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        keep.append({"query": t, "oasst_message_id": r["message_id"], "words": n})
    return keep


def domain(q):
    try:
        return make_300_balanced.domain_of(q)
    except ValueError:
        return "other"


def main():
    authored = [json.loads(l) for l in AUTHORED.read_text().splitlines() if l.strip()]
    pool = clean(load_oasst())
    # Drop near-duplicates of the authored questions.
    tf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit([r["query"] for r in authored] + [p["query"] for p in pool])
    sim = (tf.transform([p["query"] for p in pool]) @ tf.transform([r["query"] for r in authored]).T).max(axis=1).toarray().ravel()
    pool = [p for p, s in zip(pool, sim) if s < 0.6]
    for p in pool:
        p["topic"], p["context"] = domain(p["query"]), bool(CONTEXT.search(p["query"]))

    rng = random.Random(SEED)
    rng.shuffle(pool)
    chosen, per_domain = [], Counter()
    for (lo, hi), want in BANDS:
        band = [p for p in pool if lo <= p["words"] <= hi]
        # within a band, take context prompts first up to their share, then round-robin across domains
        ctx_share = round(want * MIN_CONTEXT / N_OASST)
        picks = []
        for group in ([p for p in band if p["context"]][:ctx_share * 3], band):
            by_dom = defaultdict(list)
            for p in group:
                by_dom[p["topic"]].append(p)
            limit = ctx_share if group is not band else want
            while len(picks) < limit and any(by_dom.values()):
                for d in sorted(by_dom, key=lambda d: per_domain[d]):
                    if by_dom[d] and len(picks) < limit:
                        p = by_dom[d].pop()
                        if p in picks or per_domain[d] >= DOMAIN_CAP:
                            continue
                        picks.append(p); per_domain[d] += 1
        chosen += picks
    assert len(chosen) == N_OASST, len(chosen)

    rows = [{"query": r["query"], "label": r["label"], "topic": r["topic"], "word_count": r["word_count"], "source": "authored",
             **({"reference_answer": r["reference_answer"]} if "reference_answer" in r else {})} for r in authored]
    rows += [{"query": p["query"], "label": None, "topic": p["topic"], "word_count": p["words"], "source": "oasst2",
              "oasst_message_id": p["oasst_message_id"]} for p in chosen]
    random.Random(SEED).shuffle(rows)
    DST.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"wrote {DST.name}: {len(rows)} candidates ({len(authored)} authored + {len(chosen)} oasst2 from a pool of {len(pool)})")


if __name__ == "__main__":
    main()
