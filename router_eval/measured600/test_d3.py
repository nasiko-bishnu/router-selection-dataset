"""d3-flash (vllm-sr/d3-flash, 9B Qwen3.5 decision model) on the measured 600, zero-shot and fine-tuned.

One pass of the model over the 600 training questions (eval_600) and the 600 test questions asks Jev's
low / mid / high question and keeps the last-token hidden state the readout scores (../d3_flash/run_d3.py,
its own code path). Then:
  zero-shot  = the model's own top option on each test question
  fine-tuned = its readout rows for the three tier codes, trained on the eval_600 features with the recipe in
               ../d3_flash/run_d3.py (D3Readout: AdamW 5e-4, batch 256, up to 20 epochs, 10% validation, patience 5)
Needs the d3 environment (transformers==5.17.0) and the model snapshot; runs on the Mac GPU without the ROCm kernels.

    D3_PATH=<snapshot> <d3 venv>/bin/python router_eval/measured600/test_d3.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "d3_flash"))
sys.path.insert(0, str(HERE))
from harness import save, test_set, train_set  # noqa: E402
from run_d3 import LEVEL_QUESTION, D3Readout, capture_hidden, load_model  # noqa: E402

FEATURES = Path(os.environ.get("D3_M600_FEATURES", Path.home() / ".cache" / "xroutebench" / "d3_m600_features.npz"))
ZEROSHOT = HERE / "results" / "cache" / "d3_zeroshot.jsonl"
KEYS = list(LEVEL_QUESTION["tier"]["criteria"])  # high, mid, low


def capture():
    """Run the model once over both sets; keep hidden states (both) and the zero-shot answers (test)."""
    import torch
    m = load_model()
    kept = capture_hidden(m)
    m.system_one(state={"request": "warm up"}, questions=LEVEL_QUESTION)
    out = {}
    ZEROSHOT.parent.mkdir(parents=True, exist_ok=True)
    with ZEROSHOT.open("w") as zf:
        for name, queries in (("train", train_set()[0]), ("test", test_set()[0])):
            xs = []
            for k, q in enumerate(queries):
                kept.clear()
                t0 = time.perf_counter()
                a = m.system_one(state={"request": q}, questions=LEVEL_QUESTION)["answers"]["tier"]
                ms = (time.perf_counter() - t0) * 1000
                xs.append(kept[-1][0].numpy())
                if name == "test":
                    zf.write(json.dumps({"query": q, "choice": a["choice"], "confidence": a["confidence"],
                                         "probabilities": a["probabilities"], "latency_ms": ms}) + "\n")
                if m.runtime.device.type == "mps" and k % 25 == 0:
                    torch.mps.empty_cache()  # keep PyTorch's GPU cache from pushing the 17 GB model into swap
                    print(f"  d3 {name} {k}/{len(queries)}", flush=True)
            out[f"{name}_x"], out[f"{name}_q"] = np.stack(xs), np.array(queries)
    out["readout"] = m.runtime.readout[:3].float().cpu().numpy()  # rows of codes A, B, C = high, mid, low
    FEATURES.parent.mkdir(parents=True, exist_ok=True)
    np.savez(FEATURES, **out)
    print(f"saved features to {FEATURES}", flush=True)


def main():
    if not FEATURES.exists() or not ZEROSHOT.exists():
        capture()
    queries, _ = test_set()
    z = {r["query"]: r for r in map(json.loads, ZEROSHOT.read_text().splitlines())}
    rows = [z[q] for q in queries]
    save("d3-flash-zeroshot", "Decision model, zero-shot (local)", "d3-flash asked low / mid / high, top option",
         "vllm-sr/d3-flash on the Mac GPU (PyTorch path, no ROCm kernels), Jev's LEVEL_QUESTION; latency includes the 9B backbone.",
         [r["choice"] for r in rows], [r["latency_ms"] for r in rows],
         extra=[{"confidence": r["confidence"], "probabilities": r["probabilities"]} for r in rows])
    d = np.load(FEATURES)
    D3Readout.data = {"x": {q: v for n in ("train", "test") for q, v in zip(d[f"{n}_q"], d[f"{n}_x"])},
                      "w": d["readout"], "keys": KEYS}
    import torch
    m = D3Readout()
    t0 = time.time()
    m.fit(*train_set())
    train_s = time.time() - t0
    picks, lat, extra = [], [], []
    for q in queries:  # the readout only; its probabilities are kept for the threshold tables
        s = time.perf_counter()
        with torch.no_grad():
            p = torch.softmax(m._x([q]) @ m.w.T, dim=1)[0].tolist()  # temperature 1, as in the checkpoint
        lat.append((time.perf_counter() - s) * 1000)
        probs = dict(zip(KEYS, p))
        picks.append(max(probs, key=probs.get))
        extra.append({"confidence": max(p), "probabilities": probs})
    save("d3-flash-finetuned", "Decision model, fine-tuned (local)",
         "d3-flash (backbone frozen), readout rows for the 3 tier codes fine-tuned on eval_600",
         "D3Readout from ../d3_flash/run_d3.py, warm start from the model's own readout; p50 latency is the readout "
         "only (add the backbone time of d3-flash-zeroshot).", picks, lat, extra=extra, train_seconds=train_s)


if __name__ == "__main__":
    main()
