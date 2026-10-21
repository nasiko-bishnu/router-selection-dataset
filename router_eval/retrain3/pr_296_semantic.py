# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["fastembed", "numpy", "scikit-learn"]
# ///
"""PR #296 shipped model: frozen bge-small-en-v1.5 (fastembed) embeddings + shape features + logistic regression.

Feature vector as in llm-router/classifier/model.py: [query embedding, 0.5 * context embedding (zeros when
empty), 6 shape features]; head LogisticRegression(C=4, max_iter=3000). Only the target changes: 3 tiers.
"""
import re

import numpy as np
from fastembed import TextEmbedding
from sklearn.linear_model import LogisticRegression

from common import run

_model, _cache = None, {}


def embed(texts):
    global _model
    if _model is None:
        _model = TextEmbedding("BAAI/bge-small-en-v1.5")
    todo = [t for t in dict.fromkeys(texts) if t not in _cache]
    if todo:
        _cache.update(zip(todo, _model.embed(todo)))
    v = np.array([_cache[t] for t in texts], dtype=np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


def shape_features(query, context):
    qw, cw = len(query.split()), len(context.split())
    return [np.log1p(qw) / 5, np.log1p(cw) / 6, float("```" in context or "```" in query),
            float(bool(re.search(r"[{};]|=>|->|::|\bdef \b|\bfn \b", context + query))),
            float(query.strip().endswith("?")), float(bool(re.search(r"\d", query)))]


def featurize(queries, contexts):
    q = embed(queries)
    has_ctx = np.array([bool(c.strip()) for c in contexts])
    c = np.zeros_like(q)
    if has_ctx.any():
        c[has_ctx] = embed([x for x, h in zip(contexts, has_ctx) if h])
    shape = np.array([shape_features(a, b) for a, b in zip(queries, contexts)], dtype=np.float32)
    return np.hstack([q, 0.5 * c, shape])


class BgeLR:
    def fit(self, queries, contexts, labels):
        self.clf = LogisticRegression(C=4.0, max_iter=3000).fit(featurize(queries, contexts), labels)

    def predict(self, queries, contexts):
        return list(self.clf.predict(featurize(queries, contexts)))


if __name__ == "__main__":
    run(name="pr-296-semantic", pr=296, setup="bge-small-en-v1.5 (fastembed) + shape features + LR (C=4), retrained 3-class",
        make=BgeLR,
        notes="Faithful: same fastembed BAAI/bge-small-en-v1.5 encoder, L2-normalised query + 0.5x context embedding + "
              "6 shape features, LogisticRegression(C=4). Changed: one 3-class tier LR replaces the 7-way request-type LR "
              "and the Ridge complexity regressor. Temperature scaling (does not change argmax) and the 0.9-accuracy "
              "fallback threshold are dropped.")
