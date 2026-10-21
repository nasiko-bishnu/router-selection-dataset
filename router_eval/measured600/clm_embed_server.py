"""Stand-in for `vllm serve Qwen/Qwen3-8B --runner pooling` on an Apple-silicon Mac.

Serves the OpenAI-style /v1/embeddings endpoint clm.Embedder calls, with the same pooling
CLM's head was trained on: Qwen3-8B (bf16) final hidden state of the LAST token (after the
final norm), no special tokens added, L2-normalised. truncate_prompt_tokens keeps the last
N tokens, as vLLM does. Loads the base model only (no LM head: not needed for embeddings).

    QWEN_PATH=<Qwen3-8B snapshot> <clm venv>/bin/python router_eval/measured600/clm_embed_server.py   # :8090
"""
import base64
import os

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from transformers import AutoModel, AutoTokenizer

MODEL = os.environ.get("QWEN_PATH", "Qwen/Qwen3-8B")
BATCH = int(os.environ.get("EMB_BATCH", "8"))
tok = AutoTokenizer.from_pretrained(MODEL)
DTYPE = getattr(torch, os.environ.get("EMB_DTYPE", "bfloat16"))
model = AutoModel.from_pretrained(MODEL, dtype=DTYPE).to("mps").eval()
app = FastAPI()


def embed(texts, truncate=None):
    ids = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    if truncate:
        ids = [x[-truncate:] for x in ids]
    out, n_tokens = [], sum(map(len, ids))
    for i in range(0, len(ids), BATCH):
        chunk = ids[i:i + BATCH]
        width = max(map(len, chunk))
        input_ids = torch.full((len(chunk), width), tok.pad_token_id)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for j, x in enumerate(chunk):  # right padding: causal attention keeps the last real token unaffected
            input_ids[j, :len(x)] = torch.tensor(x)
            mask[j, :len(x)] = 1
        with torch.inference_mode():
            h = model(input_ids=input_ids.to("mps"), attention_mask=mask.to("mps")).last_hidden_state
        last = h[torch.arange(len(chunk)), torch.tensor([len(x) - 1 for x in chunk])].float().cpu().numpy()
        del h
        torch.mps.empty_cache()  # MPS keeps freed blocks cached; on a 24 GB Mac they push the 16 GB model into swap
        out.extend(last / np.linalg.norm(last, axis=1, keepdims=True))
    return out, n_tokens


@app.post("/v1/embeddings")
def embeddings(body: dict):
    texts = body["input"] if isinstance(body["input"], list) else [body["input"]]
    vecs, n = embed(texts, body.get("truncate_prompt_tokens"))
    b64 = body.get("encoding_format") == "base64"
    data = [{"object": "embedding", "index": i,
             "embedding": base64.b64encode(v.astype(np.float32).tobytes()).decode() if b64 else v.tolist()}
            for i, v in enumerate(vecs)]
    return {"object": "list", "data": data, "model": body.get("model"), "usage": {"prompt_tokens": n, "total_tokens": n}}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": "qwen3-8b", "object": "model"}]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8090")), log_level="warning")
