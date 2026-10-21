"""CLM (Contrastive-LM: Qwen3-8B encoder + projection heads) on the measured 600, zero-shot and fine-tuned.

  zero-shot  = CLM's engine asked Jev's low / mid / high question, top option (../clm/run_clm.py zeroshot)
  fine-tuned = both projection heads and the logit scale fine-tuned on eval_600 with CLM's choice/softce recipe
               (../clm/run_clm.py ClmTierHead: AdamW 5e-4, batch 256, up to 20 epochs, 10% validation, patience 5)
Needs the CLM environment, CLM_CKPT (CLM_v0.1-8B.pt) and the Qwen3-8B encoder served at CLM_EMB_URL (on a Mac,
the PyTorch stand-in for vLLM, mps_embed_server.py).

    CLM_CKPT=<.pt> <clm venv>/bin/python router_eval/measured600/test_clm.py
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "clm"))
sys.path.insert(0, str(HERE))
from harness import evaluate_model, save, test_set  # noqa: E402
from run_clm import LEVEL_QUESTION, ClmTierHead, engine  # noqa: E402


def probs(m, qs):  # softmax of CLM's scaled state-option similarities; its argmax is predict()'s pick
    import torch
    import torch.nn.functional as F
    states, _, _ = m._texts(qs)
    with torch.no_grad():
        zs = F.normalize(m.sh(torch.from_numpy(m.emb.embed(states)[0])), dim=-1)
        p = torch.softmax(m.scale * zs @ m.zc.T, -1).tolist()
    return [dict(zip(m.keys, r)) for r in p]


def main():
    if "--finetuned-only" in sys.argv:
        return finetuned()
    e = engine()
    e.answer({"request": "warm up"}, LEVEL_QUESTION)
    queries, _ = test_set()
    picks, lat, extra = [], [], []
    for k, q in enumerate(queries):
        t0 = time.perf_counter()
        a = e.answer({"request": q}, LEVEL_QUESTION)["answers"]["tier"]
        lat.append((time.perf_counter() - t0) * 1000)
        picks.append(a["choice"])
        extra.append({"confidence": a["confidence"], "probabilities": a["probabilities"]})
        if k % 100 == 0:
            print(f"  clm zero-shot {k}/{len(queries)}", flush=True)
    save("clm-zeroshot", "Decision model, zero-shot (local)", "CLM asked low / mid / high, top option",
         "CLM_v0.1-8B heads; Qwen3-8B encoder in PyTorch on the Mac GPU (stand-in for vLLM, same last-token pooling); "
         "latency includes the encoder.", picks, lat, extra=extra)
    finetuned()


def finetuned():
    m = ClmTierHead()
    evaluate_model("clm-finetuned", "Decision model, fine-tuned (local)",
                   "CLM (Qwen3-8B frozen), projection heads fine-tuned on eval_600 for the 3 tier options",
                   "ClmTierHead from ../clm/run_clm.py, warm start from CLM_v0.1-8B; latency includes the encoder call.",
                   m, proba=lambda qs: probs(m, qs))


if __name__ == "__main__":
    main()
