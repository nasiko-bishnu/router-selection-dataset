# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["torch", "transformers", "safetensors", "numpy", "scikit-learn"]
# ///
"""PR #292: fine-tuned all-MiniLM-L6-v2 (llm-router/classifier/train_semantic.py), retrained with one 3-class tier head.

Recipe kept: (query, context) sentence pair, max 256 tokens; mean-pool only the query tokens (token_type 0),
L2-normalise; linear head initialised to 10 x normalised class centroids, bias 0; freeze the encoder except
its last two layers; AdamW (encoder lr 2e-5, head lr 2e-3, weight decay 0.01), batch 8, 16 epochs, grad-clip 1,
seed 42, 2 threads, deterministic; epoch 0..16 picked on a held-out validation split, best state kept.
"""
import copy
import os
import random

os.environ["USE_TF"] = "0"
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from transformers import AutoModel, AutoTokenizer

from common import run

BASE = os.environ.get("MINILM_BASE", "sentence-transformers/all-MiniLM-L6-v2")  # the PR's base model
LOCAL = os.path.isdir(BASE)
TIERS = ["low", "mid", "high"]
EPOCHS, SEED = 16, 42


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(BASE, local_files_only=LOCAL, attn_implementation="eager")
        self.tier_head = nn.Linear(self.encoder.config.hidden_size, 3)

    def features(self, input_ids, attention_mask, token_type_ids):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state
        mask = (attention_mask * (token_type_ids == 0)).unsqueeze(-1).float()  # query tokens only
        return nn.functional.normalize((h * mask).sum(1) / mask.sum(1).clamp_min(1), dim=-1)

    def forward(self, **x):
        return self.tier_head(self.features(**x))


class FineTunedMiniLm:
    def encode(self, queries, contexts):
        return self.tok(list(queries), [c or "" for c in contexts], padding=True, truncation="longest_first",
                        max_length=256, return_tensors="pt")

    def fit(self, queries, contexts, labels):
        torch.set_num_threads(2)
        torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
        torch.use_deterministic_algorithms(True)
        # The PR selects the epoch on a separate validation file; we hold out a stratified 20% of the training data.
        tr, va = train_test_split(range(len(queries)), test_size=0.2, stratify=labels, random_state=SEED)
        self.tok = AutoTokenizer.from_pretrained(BASE, local_files_only=LOCAL)
        m = self.model = Model()
        for p in m.encoder.parameters():
            p.requires_grad_(False)
        for layer in m.encoder.encoder.layer[-2:]:
            for p in layer.parameters():
                p.requires_grad_(True)
        x = self.encode([queries[i] for i in tr], [contexts[i] for i in tr])
        xv = self.encode([queries[i] for i in va], [contexts[i] for i in va])
        y = torch.tensor([TIERS.index(labels[i]) for i in tr])
        yv = torch.tensor([TIERS.index(labels[i]) for i in va])
        m.eval()
        with torch.no_grad():
            f = m.features(**x)
            for i in range(3):
                m.tier_head.weight[i] = 10 * nn.functional.normalize(f[y == i].mean(0), dim=0)
            m.tier_head.bias.zero_()
        opt = torch.optim.AdamW([{"params": [p for p in m.encoder.parameters() if p.requires_grad], "lr": 2e-5},
                                 {"params": m.tier_head.parameters(), "lr": 2e-3}], weight_decay=.01)
        best, best_state = float("-inf"), None
        for epoch in range(EPOCHS + 1):
            if epoch:
                m.train()
                for idx in torch.randperm(len(y)).split(8):
                    loss = nn.functional.cross_entropy(m(**{k: v[idx] for k, v in x.items()}), y[idx])
                    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1); opt.step()
            m.eval()
            with torch.no_grad():
                acc = float((m(**xv).argmax(-1) == yv).float().mean())
            if acc > best:  # PR score is acc - 0.05 * complexity MAE; the effort head is gone, so accuracy only
                best, best_state, self.epoch = acc, copy.deepcopy(m.state_dict()), epoch
        m.load_state_dict(best_state); m.eval()
        print(f"  selected epoch {self.epoch} (val acc {best:.3f})", flush=True)

    def predict(self, queries, contexts):
        with torch.no_grad():
            return [TIERS[i] for i in self.model(**self.encode(queries, contexts)).argmax(-1).tolist()]


if __name__ == "__main__":
    run(name="pr-292", pr=292, setup="all-MiniLM-L6-v2 fine-tuned (last 2 layers + tier head), retrained 3-class",
        make=FineTunedMiniLm,
        notes="Faithful: same MiniLM base, query-token mean pooling + L2 norm, centroid-initialised linear head, last two "
              "layers unfrozen, AdamW 2e-5/2e-3 wd 0.01, batch 8, 16 epochs, clip 1, seed 42, best epoch on validation. "
              "Changed: one 3-class tier head replaces the 7-way type head and 5-level effort head (and their ordinal CDF "
              "loss); epoch picked by accuracy on a stratified 20% held out from the training data (the PR had a separate "
              "validation file), so the final model trains on 80%; temperature scaling (no effect on argmax) and the "
              "INT8 ONNX export are skipped: predict() runs the PyTorch model in fp32.")
