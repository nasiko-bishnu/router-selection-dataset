# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "scikit-learn"]
# ///
"""PR #296 offline `local` backend: hand-rolled lexical features + logistic regression (train_lexical.py, lexical.py).

Features per segment (q = query, x = context): word unigrams, word bigrams, char 3-5 grams of " tok "
(tokens of length >= 2); weight (1 + ln tf) * idf, context scaled 0.5, L2-normalised; 6 dense shape
features appended. Vocabulary keeps features seen in >= 2 seeds (here: >= 2 training questions).
Head LogisticRegression(C=20, max_iter=5000). Only the target changes: 3 tiers.
"""
import math
import re
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression

from common import run

CONTEXT_SCALE, MIN_DF = 0.5, 2
_TOKEN, _CODEISH, _DIGIT = re.compile(r"\w+"), re.compile(r"[{};]|=>|->|::|\bdef\b|\bfn\b"), re.compile(r"\d")


def segment_features(text, p):
    toks = _TOKEN.findall(text.lower())
    out = Counter(f"{p}w:{t}" for t in toks)
    out.update(f"{p}b:{a} {b}" for a, b in zip(toks, toks[1:]))
    for t in toks:
        if len(t) >= 2:
            s = f" {t} "
            out.update(f"{p}c:{s[i:i + n]}" for n in (3, 4, 5) for i in range(len(s) - n + 1))
    return out


def raw_features(q, c):
    f = segment_features(q, "q")
    f.update(segment_features(c, "x"))
    return f


def shape(q, c):
    qw, cw = len(q.split()), len(c.split())
    return [math.log1p(qw) / 5.0, math.log1p(cw) / 6.0, float("```" in c or "```" in q),
            float(bool(_CODEISH.search(c + q))), float(q.strip().endswith("?")), float(bool(_DIGIT.search(q)))]


class Lexical:
    def matrix(self, queries, contexts):
        data, idx, ptr = [], [], [0]
        for q, c in zip(queries, contexts):
            items = [(self.vocab[f], (1 + math.log(tf)) * self.idf[self.vocab[f]] * (CONTEXT_SCALE if f[0] == "x" else 1))
                     for f, tf in raw_features(q, c).items() if f in self.vocab]
            norm = math.sqrt(sum(v * v for _, v in items)) or 1.0
            idx += [i for i, _ in items]
            data += [v / norm for _, v in items]
            ptr.append(len(idx))
        sp = csr_matrix((data, idx, ptr), shape=(len(queries), len(self.vocab)))
        return hstack([sp, csr_matrix(np.array([shape(q, c) for q, c in zip(queries, contexts)]))]).tocsr()

    def fit(self, queries, contexts, labels):
        df = Counter(f for q, c in zip(queries, contexts) for f in set(raw_features(q, c)))
        kept = sorted(f for f, n in df.items() if n >= MIN_DF)
        self.vocab = {f: i for i, f in enumerate(kept)}
        self.idf = [math.log((1 + len(queries)) / (1 + df[f])) + 1.0 for f in kept]
        self.clf = LogisticRegression(C=20.0, max_iter=5000).fit(self.matrix(queries, contexts), labels)

    def predict(self, queries, contexts):
        return list(self.clf.predict(self.matrix(queries, contexts)))


if __name__ == "__main__":
    run(name="pr-296-lexical", pr=296, setup="word 1-2 grams + char 3-5 grams (custom TF-IDF) + shape features + LR (C=20), retrained 3-class",
        make=Lexical,
        notes="Faithful: same featurizer as lexical.py (word uni/bigrams, char 3-5 grams, (1+ln tf)*idf, context x0.5, "
              "L2 norm, 6 shape features), min-df 2, LogisticRegression(C=20). Changed: one 3-class tier LR replaces the "
              "7-way request-type LR and Ridge complexity; document frequency counts questions (our data has no paraphrase "
              "seeds); temperature and fallback threshold dropped.")
