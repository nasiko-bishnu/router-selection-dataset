# /// script
# requires-python = ">=3.10"
# dependencies = ["requests"]
# ///
"""Have one model answer every candidate question; answers are cached so a rerun only fills gaps.

    uv run router_eval/tier_measure/answer.py openai:gpt-4.1-nano --budget 2
    uv run --with mlx-lm router_eval/tier_measure/answer.py mlx:mlx-community/Meta-Llama-3.1-8B-Instruct-4bit

openai:<model>  OpenAI API, key read from OPENAI_API_KEY in the repo's .env (never printed or saved).
mlx:<repo>      a local model on the Mac GPU through mlx-lm (downloads the weights on first use).
Writes answers/<model>.jsonl: id, model, answer, input/output tokens, cost, latency. Stops at --budget dollars.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ANSWERS, ROOT, candidates, slug  # noqa: E402

# $ per million tokens (input, output); reasoning tokens are billed as output.
OPENAI_PRICES = {"gpt-4.1-nano": (0.10, 0.40), "gpt-4.1": (2.00, 8.00), "gpt-4.1-mini": (0.40, 1.60), "gpt-5-nano": (0.05, 0.40),
                 "gpt-5-mini": (0.25, 2.00), "gpt-5.4": (2.50, 15.00), "gpt-5.5": (5.00, 30.00)}
MAX_TOKENS = 2000


def openai_answerer(model):
    import requests
    key = next(l.split("=", 1)[1].strip().strip("\"'") for l in (ROOT / ".env").read_text().splitlines()
               if l.startswith("OPENAI_API_KEY="))
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    reasoning = model.startswith(("gpt-5", "o"))
    price = OPENAI_PRICES.get(model)
    assert price, f"add a price for {model} to OPENAI_PRICES first"

    def ask(q):
        body = {"model": model, "messages": [{"role": "user", "content": q}], "max_completion_tokens": MAX_TOKENS * (4 if reasoning else 1)}
        if not reasoning:
            body["temperature"] = 0
        for attempt in range(5):
            try:
                r = s.post("https://api.openai.com/v1/chat/completions", json=body, timeout=300)
            except (requests.ConnectionError, requests.Timeout):
                time.sleep(5 * 2 ** attempt); continue
            if r.status_code in (429, 500, 502, 503):
                time.sleep(5 * 2 ** attempt); continue
            r.raise_for_status()
            j = r.json(); u = j.get("usage") or {}
            tin, tout = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
            return j["choices"][0]["message"]["content"] or "", tin, tout, tin * price[0] / 1e6 + tout * price[1] / 1e6
        raise RuntimeError("OpenAI kept failing")
    return ask


def mlx_answerer(repo):
    from mlx_lm import generate, load
    model, tok = load(repo)
    gpt_oss = "gpt-oss" in repo  # reasons in an "analysis" channel before the "final" one

    def ask(q):
        extra = {"reasoning_effort": "low"} if gpt_oss else {}  # low effort keeps 600 local answers to hours, not a day
        prompt = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False, **extra)
        raw = generate(model, tok, prompt=prompt, max_tokens=MAX_TOKENS * (2 if gpt_oss else 1), verbose=False)
        text = raw
        if gpt_oss:  # keep only the final answer; no final channel means it ran out of tokens while reasoning
            marker = "<|channel|>final<|message|>"
            text = raw.split(marker)[-1] if marker in raw else ""
            for end in ("<|return|>", "<|end|>"):
                text = text.split(end)[0]
        return text.strip(), len(tok.encode(prompt)), len(tok.encode(raw)), 0.0
    return ask


def mlxvlm_answerer(repo):
    """Models converted with mlx-vlm (e.g. Qwen3.5), used text-only with thinking switched off."""
    from mlx_vlm import generate, load
    model, processor = load(repo)
    tok = getattr(processor, "tokenizer", processor)

    def ask(q):
        prompt = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False,
                                         enable_thinking=False)
        res = generate(model, processor, prompt, max_tokens=MAX_TOKENS, temperature=0.0, verbose=False)
        text = getattr(res, "text", res)
        return text, len(tok.encode(prompt)), len(tok.encode(text)), 0.0
    return ask


def main():
    global MAX_TOKENS  # --max-tokens overrides it for the answerers
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="openai:<model> or mlx:<huggingface repo>")
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--limit", type=int, help="only the first N candidates (a trial)")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help="answer length limit (reasoning models get 4x)")
    ap.add_argument("--redo-capped", type=int, metavar="N",
                    help="re-answer only the saved answers with at least N output tokens (cut off), replacing them")
    a = ap.parse_args()
    MAX_TOKENS = a.max_tokens
    backend, name = a.model.split(":", 1)
    ask = {"openai": openai_answerer, "mlx": mlx_answerer, "mlxvlm": mlxvlm_answerer}[backend](name)
    ANSWERS.mkdir(parents=True, exist_ok=True)
    out = ANSWERS / f"{slug(a.model)}.jsonl"
    saved = [json.loads(l) for l in out.read_text().splitlines()] if out.exists() else []
    if a.redo_capped:  # drop the cut-off answers from the file; they are answered again below
        redo = {s["id"] for s in saved if s["output_tokens"] >= a.redo_capped}
        out.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in saved if s["id"] not in redo))
        print(f"re-answering {len(redo)} cut-off answers with max_tokens={MAX_TOKENS}")
    done = {json.loads(l)["id"] for l in out.read_text().splitlines()} if out.exists() else set()
    rows = [r for r in candidates() if r["id"] not in done][: a.limit]
    spent, t_all = 0.0, time.time()
    with out.open("a") as f:
        for i, r in enumerate(rows, 1):
            if spent >= a.budget:
                print(f"budget of ${a.budget} reached after {i - 1} answers"); break
            t0 = time.time()
            text, tin, tout, cost = ask(r["query"])
            spent += cost
            f.write(json.dumps({"id": r["id"], "model": a.model, "answer": text, "input_tokens": tin, "output_tokens": tout,
                                "cost": round(cost, 6), "latency_s": round(time.time() - t0, 2)}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"  {i}/{len(rows)} answered, ${spent:.3f} spent", flush=True)
    total = len(done) + len(rows)
    print(f"{a.model}: {total} answers in {out.name} | this run ${spent:.3f}, {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
