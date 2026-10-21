# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas", "pyarrow", "numpy", "requests", "huggingface_hub", "laya", "tabulate"]
# ///
"""Tier-selection benchmark on xRouteBench: TypeSafe Jev (OpenRouter), Open-Jev 9B (team-hosted),
Laya and the nasiko llm-router classifier.

Every xRouteBench query was already run on all 18 candidate LLMs, so evaluation is offline:
a selector only picks a tier, and we look up the precomputed performance/tokens of that
tier's model. The only live calls are the tier decisions themselves.

    uv run bench.py --selectors nasiko,laya            # no API key needed
    OPENROUTER_API_KEY=... uv run bench.py             # all selectors
    uv run bench.py --n 0                              # full test split (3729 queries)
    uv run bench.py --selectors laya,laya-multilingual,laya-typed-decisions   # Laya checkpoints side by side
    uv run bench.py --selectors laya-sizelabels,laya-multilingual-withheader  # older prompt setups (see LAYA_SETUPS)
    OPENJEV_API_KEY=... uv run bench.py --n 0 --selectors openjev,openjev-withheader,openjev-sizelabels
    uv run train_laya.py --base multilingual && uv run bench.py --n 0 --selectors laya-tuned-multilingual

Decisions are cached in results/decisions_<selector>_<n>_<seed>.jsonl, so re-runs only
report (delete the file to re-query).
"""

import argparse
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RESULTS = HERE / "results"
DATASET = "ulab-ai/xRouteBench"

# Tier → model. Chosen on the TRAIN split: a clean 671B / 70B / 8B size+cost ladder.
#   cogito-v2-1-671b              train perf 0.747  $0.45 / 1k queries
#   llama-3.3-70b-instruct-turbo  train perf 0.739  $0.24 / 1k queries
#   llama-3-8b-instruct-lite      train perf 0.556  $0.02 / 1k queries
TIER_MODELS = {
    "T1": "cogito-v2-1-671b",
    "T2": "llama-3.3-70b-instruct-turbo",
    "T3": "llama-3-8b-instruct-lite",
}
TIERS = ["T1", "T2", "T3"]

# Same state truncation and same typed question for Jev and Laya, so the only variable is
# the decision model.
MAX_STATE_CHARS = 2000
TIER_QUESTION = {
    "tier": {
        "type": "choice",
        "instructions": "Pick the cheapest LLM tier that will still answer `request` correctly. "
        "T1 is the largest and most expensive, T3 the smallest and cheapest.",
        "criteria": {
            "T1": "the strongest 671B frontier model is needed: hard multi-step math, competition "
            "problems, non-trivial code, specialist expert knowledge",
            "T2": "a 70B model is needed: moderate reasoning over several steps, domain "
            "knowledge, routine coding tasks",
            "T3": "a small 8B model is enough: simple factual recall, common sense, easy "
            "multiple choice, reading comprehension",
        },
    }
}
# The original size-named question (LAYA_RESULTS.md run 1), kept for the prompt comparison.
SIZE_QUESTION = {
    "tier": {
        "type": "choice",
        "instructions": "Pick the cheapest LLM tier that will still answer `request` correctly.",
        "criteria": {
            "large": "the strongest frontier model is needed: hard multi-step math, competition "
            "problems, non-trivial code, specialist expert knowledge",
            "medium": "a 70B model is needed: moderate reasoning over several steps, domain "
            "knowledge, routine coding tasks",
            "small": "a small 8B model is enough: simple factual recall, common sense, easy "
            "multiple choice, reading comprehension",
        },
    }
}
CHOICE_TO_TIER = {"large": "T1", "medium": "T2", "small": "T3", **{t: t for t in TIERS}}

# Laya prompt setups, chosen by selector suffix: laya[-<checkpoint>][-<setup>].
#   (none)       run 3: TIER_QUESTION, fixed instruction header removed (also what Jev gets)
#   withheader   run 2: TIER_QUESTION, full query text
#   sizelabels   run 1: SIZE_QUESTION, full query text
LAYA_SETUPS = {"": (TIER_QUESTION, False), "withheader": (TIER_QUESTION, True), "sizelabels": (SIZE_QUESTION, True)}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def load_queries(n: int, seed: int, split: str = "test") -> pd.DataFrame:
    """One row per query of `split` with perf/cost for each tier's model."""
    from huggingface_hub import hf_hub_download

    cand = pd.read_parquet(
        hf_hub_download(DATASET, "llm_candidates/train.parquet", repo_type="dataset")
    ).set_index("model_name")
    df = pd.read_parquet(
        hf_hub_download(DATASET, f"llmrouter_generic/{split}.parquet", repo_type="dataset"),
        columns=["task_name", "query", "model_name", "performance", "input_tokens", "output_tokens"],
    )
    df = df[df.model_name.isin(TIER_MODELS.values())].copy()
    df["cost"] = (
        df.input_tokens * df.model_name.map(cand.input_price_per_1m)
        + df.output_tokens * df.model_name.map(cand.output_price_per_1m)
    ) / 1e6
    df["qid"] = [hashlib.sha1(f"{t}\0{q}".encode()).hexdigest()[:16] for t, q in zip(df.task_name, df["query"])]
    model_to_tier = {m: t for t, m in TIER_MODELS.items()}
    df["tier"] = df.model_name.map(model_to_tier)

    wide = df.pivot_table(index="qid", columns="tier", values=["performance", "cost"], aggfunc="first")
    wide.columns = [f"{'perf' if a == 'performance' else 'cost'}_{t}" for a, t in wide.columns]
    meta = df.drop_duplicates("qid").set_index("qid")[["task_name", "query"]]
    q = meta.join(wide, how="inner").dropna()

    if n and n < len(q):
        # Stratified by task so small tasks (aime, human_eval) are still represented.
        frac = n / len(q)
        keep = [
            g.sample(min(len(g), max(5, round(len(g) * frac))), random_state=seed).index
            for _, g in q.groupby("task_name")
        ]
        q = q.loc[np.concatenate(keep)]
    return q.sample(frac=1, random_state=seed)  # shuffled: order matters for online learning


def state_for(query: str, keep_header: bool = False) -> dict:
    # Drop the per-task "[System Instruction] …" header: it's identical across a task, so it
    # carries no difficulty signal and its wording ("multiple-choice") biases the pick.
    question = query if keep_header else query.split("[User Query]", 1)[-1].strip()
    return {"request": question[:MAX_STATE_CHARS]}


# --------------------------------------------------------------------------
# Selectors — each returns {qid: {"tier": "T1|T2|T3", "latency_ms": float, "cost": float, ...}}
# --------------------------------------------------------------------------

# Port of nasiko-cloud-rs/oss/llm-router/src/routing/{patterns,classifier}.rs
_CATEGORY_PATTERNS = [
    ("code_generation", [
        r"(?i)\b(write|implement|create|build|generate)\b.{0,40}\b(function|script|code|class|program|api|endpoint|method|module)\b",
        r"(?i)\bwrite (a|an|the|me)\b.*\b(python|javascript|typescript|rust|golang|go|java|c\+\+|sql)\b",
        r"(?i)\bfix (this|the) bug\b",
        r"(?i)\brefactor\b",
        r"(?i)\badd error handling\b",
    ]),
    ("code_understanding", [
        r"(?i)\bexplain (what|how|why)\b",
        r"(?i)\bwhat does (this|that|the) (function|code|script|class) do\b",
        r"(?i)\bhow does (this|that|the) (function|code|script|class) work\b",
        r"(?i)\bwalk me through (this|that) code\b",
        r"(?i)\bwhat is this code doing\b",
    ]),
    ("technical_design", [
        r"(?i)\bhow should i design\b",
        r"(?i)\b(api|system|database|schema) design\b",
        r"(?i)\barchitecture\b",
        r"(?i)\bdesign (a|an|the) (system|api|service|schema|database)\b",
        r"(?i)\btrade-?offs?\b",
    ]),
    ("analytical_reasoning", [
        r"(?i)\bcalculate\b",
        r"(?i)\bprobability\b",
        r"(?i)\bsolve\b",
        r"(?i)\bprove\b",
        r"(?i)\bproof\b",
        r"(?i)\bhow many\b",
        r"(?i)\bwhat'?s the (sum|product|average|result)\b",
        r"[0-9]+\s*[+\-*/]\s*[0-9]+",
    ]),
    ("writing", [
        r"(?i)\bdraft\b",
        r"(?i)\bwrite (an?|the)\b.*\b(email|blog|article|essay|post|letter|story|poem)\b",
        r"(?i)\bcompose\b",
        r"(?i)\brewrite (this|that|the)\b",
        r"(?i)\bmake this sound\b",
    ]),
    ("factual_lookup", [
        r"(?i)\bwhat is (the )?capital of\b",
        r"(?i)^\s*(who|what|when|where) (is|was|are|were)\b",
        r"(?i)\bdefine\b",
        r"(?i)\bhow many\b.*\b(are there|exist)\b",
    ]),
]
_CATEGORY_PATTERNS = [(rt, [re.compile(p) for p in ps]) for rt, ps in _CATEGORY_PATTERNS]
# (tier, quality_tier, cost, strengths) — TIER_ARMS in classifier.rs
_TIER_ARMS = [
    ("T1", 3, 15.0, {"code_generation", "analytical_reasoning", "technical_design"}),
    ("T2", 2, 3.0, {"code_understanding", "writing"}),
    ("T3", 1, 0.8, {"factual_lookup", "general"}),
]
_W_QUALITY, _W_COST, _PRIOR_PSEUDO, _MAX_SAMPLES = 0.7, 0.3, 4.0, 200


def _request_type(text: str) -> str:
    best, best_score = "general", 0
    for rt, pats in _CATEGORY_PATTERNS:
        score = sum(1 for p in pats if p.search(text))
        if score > best_score:
            best, best_score = rt, score
    return best


def _thompson(cells: dict, rt: str, rng: np.random.Generator) -> str:
    lo, hi = min(a[2] for a in _TIER_ARMS), max(a[2] for a in _TIER_ARMS)
    best, best_score = "T1", -np.inf
    for tier, qt, cost, strengths in _TIER_ARMS:
        prior = min(max(0.5 + 0.15 * max(qt - 1, 0) + (0.15 if rt in strengths else 0.0), 0.05), 0.95)
        mean, n = cells.get((tier, rt), (0.0, 0))
        s = mean * n
        q = rng.beta(max(prior * _PRIOR_PSEUDO + s, 1e-6), max((1 - prior) * _PRIOR_PSEUDO + n - s, 1e-6))
        score = _W_QUALITY * q + _W_COST * (1 - (cost - lo) / (hi - lo))
        if score > best_score:
            best, best_score = tier, score
    return best


def select_nasiko(q: pd.DataFrame, seed: int, online: bool) -> dict:
    """The llm-router's regex request-type + Thompson tier pick.

    online=False: cold-start priors only (what a fresh deployment does).
    online=True: after each pick, fold the chosen tier's *true* benchmark performance into
    the cells — a best-case stand-in for the production feedback signal.
    """
    rng = np.random.default_rng(seed)
    cells: dict = {}
    out = {}
    for qid, row in q.iterrows():
        t0 = time.perf_counter()
        rt = _request_type(row["query"])
        tier = _thompson(cells, rt, rng)
        out[qid] = {"tier": tier, "latency_ms": (time.perf_counter() - t0) * 1000, "cost": 0.0, "request_type": rt}
        if online:
            mean, n = cells.get((tier, rt), (0.0, 0))
            n_eff = min(n, _MAX_SAMPLES)
            obs = float(row[f"perf_{tier}"])
            cells[(tier, rt)] = (mean + (obs - mean) / (n_eff + 1), min(n + 1, _MAX_SAMPLES))
    return out


def parse_setup(tag: str, prefix: str) -> tuple[str | None, str]:
    """'<prefix>[-<variant>][-<setup>]' → (variant or None, LAYA_SETUPS key).

    e.g. laya-multilingual-sizelabels → ('multilingual', 'sizelabels'); openjev-withheader → (None, 'withheader').
    """
    rest = tag.removeprefix(prefix).removeprefix("-")
    for setup in filter(None, LAYA_SETUPS):
        if rest == setup or rest.endswith("-" + setup):
            return rest.removesuffix(setup).removesuffix("-") or None, setup
    return rest or None, ""


def route(enough: np.ndarray, cut) -> np.ndarray:
    """Tier index per question: the cheapest tier whose chance of being enough is >= cut (T3, then T2),
    else T1. enough: [questions, 3] in TIERS order. A 1-D array of cuts gives [cuts, questions]."""
    cut = np.asarray(cut, dtype=float)[..., None]
    return np.where(enough[:, 2] >= cut, 2, np.where(enough[:, 1] >= cut, 1, 0))


def tier_enough(probs: np.ndarray) -> np.ndarray:
    """From an ideal-tier model's softmax: T3 is enough with P(T3), T2 with P(T2) + P(T3)."""
    return np.column_stack([np.ones(len(probs)), probs[:, 1] + probs[:, 2], probs[:, 2]])


def per_tier_enough(p_right: np.ndarray) -> np.ndarray:
    """From a per-tier model: T3 is enough with P(T3 right), T2 with P(T2 right). T1 is the fallback."""
    return np.column_stack([np.ones(len(p_right)), p_right[:, 1], p_right[:, 2]])


def routing_curve(rows: list[dict], enough: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Every routing `route` gives as its cut-off slides from all-T3 to all-T1:
    (cuts, gain vs the blind split at each cut, mean cost at each cut, mean cost of each tier).
    rows: [{"scores": {tier: score}, "costs": {tier: cost}}] per question."""
    perf = np.array([[r["scores"][t] for t in TIERS] for r in rows])
    cost = np.array([[r["costs"][t] for t in TIERS] for r in rows])
    p, c = perf.mean(0), cost.mean(0)
    cuts = np.concatenate([[0.0], np.unique(np.quantile(enough[:, 1:], np.linspace(0, 1, 401))), [np.inf]])
    picks = route(enough, cuts)  # [cuts, questions]
    rows_i = np.arange(len(rows))
    pf, cf = perf[rows_i, picks].mean(1), cost[rows_i, picks].mean(1)
    return cuts, (pf - np.interp(cf, c[::-1], p[::-1])) / p[0] * 100, cf, c


def vs_blind_avg(rows: list[dict], enough: np.ndarray) -> float:
    """vs_blind averaged over every budget from always-T3 to always-T1, without argmax.

    enough = each tier's chance of being enough (tier_enough / per_tier_enough). Routing by a
    cut-off over the whole range scores how well the model orders questions by difficulty, so an
    argmax that collapses onto T3 doesn't hide a good ordering, and averaging over the whole curve
    is steadier than the one argmax point."""
    _, gain, cf, c = routing_curve(rows, enough)
    o = np.argsort(cf)
    area = ((gain[o][1:] + gain[o][:-1]) / 2 * np.diff(cf[o])).sum()
    return round(float(area / (c[0] - c[2])), 2)


def laya_items(tok, cfg: dict, texts: list[str], question: dict) -> list[dict]:
    """Tokenise each text exactly as Laya's predict() would (options in criteria order = TIERS)."""
    from laya.agent import Agent
    from laya.common import build_sequence

    qint = Agent._to_internal(question["tier"])
    items = []
    for text in texts:
        ids, markers = build_sequence(tok, {"request": text}, qint, cfg.get("max_len", 512), cfg.get("head_max_len", 192))
        assert len(markers) == len(TIERS), "options truncated"
        items.append({"ids": ids, "markers": markers, "qtype": 0})
    return items


def laya_option_logits(agent, texts: list[str], question: dict) -> np.ndarray:
    """Laya's raw score for each option, [len(texts), 3], without predict()'s softmax across options."""
    import torch
    from laya.common import collate_items

    b = collate_items([laya_items(agent.tok, agent.cfg, texts, question)], agent.tok.pad_token_id)
    device = next(agent.model.parameters()).device
    agent.model.eval()
    with torch.no_grad():
        logits, _ = agent.model(*(b[k].to(device) for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")))
    return logits.float().cpu().numpy()


def select_laya(q: pd.DataFrame, subfolder: str | None, setup: str = "") -> dict:
    os.environ.setdefault("USE_TF", "0")
    import laya
    import torch

    question, keep_header = LAYA_SETUPS[setup]
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  loading Laya ({subfolder or 'english root'}, setup {setup or 'default'}) on {device}…")
    if subfolder and subfolder.startswith("tuned"):  # laya-tuned-<base>: a train_laya.py checkpoint
        agent = laya.load(str(HERE / "checkpoints" / subfolder), device=device)
    else:
        agent = laya.load("convaiinnovations/laya", device=device, subfolder=subfolder)
    agent.predict(state_for("warmup"), question)
    # A train_laya.py --target per-tier checkpoint scores each tier's chance of being right; its
    # validation cut-off picks the tier (predict()'s softmax would make the tiers compete instead).
    fine = agent.cfg.get("fine_tuned", {})
    per_tier = fine.get("target") == "per-tier"

    out = {}
    for i, (qid, row) in enumerate(q.iterrows(), 1):
        t0 = time.perf_counter()
        if per_tier:
            p = 1 / (1 + np.exp(-laya_option_logits(agent, [state_for(row["query"], keep_header)["request"]], question)))
            pick = {"tier": TIERS[int(route(per_tier_enough(p), fine["cutoff"])[0])],
                    "p_right": dict(zip(TIERS, p[0].astype(float).round(4).tolist()))}
        else:
            ans = agent.predict(state_for(row["query"], keep_header), question)["answers"]["tier"]
            pick = {"tier": CHOICE_TO_TIER[ans["choice"]], "probabilities": ans["probabilities"]}
        out[qid] = {**pick, "latency_ms": (time.perf_counter() - t0) * 1000, "cost": 0.0}
        if i % 100 == 0:
            print(f"  laya {i}/{len(q)}")
    return out


def _load_dotenv():
    env = HERE / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def _systemone(q: pd.DataFrame, name: str, url: str, headers: dict, body_extra: dict,
               workers: int, setup: str, partial: Path) -> dict:
    """POST each query to a Jev-style /systemone endpoint.

    Every decision is appended to `partial` as it arrives, and a re-run skips the queries
    already there, so a multi-hour run survives interruptions.
    """
    import requests

    question, keep_header = LAYA_SETUPS[setup]
    done = {}
    if partial.exists():
        lines = partial.read_text().splitlines()
        try:
            json.loads(lines[-1]) if lines else None
        except json.JSONDecodeError:  # killed mid-write: drop the torn last line
            lines = lines[:-1]
            partial.write_text("".join(line + "\n" for line in lines))
        done = {d.pop("qid"): d for d in map(json.loads, lines)}
        print(f"  {name}: resuming, {len(done)} decisions already in {partial.name}")
    session = requests.Session()
    session.headers.update(headers)

    def one(qid, query):
        body = {**body_extra, "state": state_for(query, keep_header), "questions": question}
        for attempt in range(8):
            t0 = time.perf_counter()
            try:
                r = session.post(url, json=body, timeout=120)
            except requests.RequestException as e:
                err = repr(e)
            else:
                if r.ok:
                    j = r.json()
                    ans = j["answers"]["tier"]
                    return qid, {
                        "tier": CHOICE_TO_TIER[ans["choice"]],
                        "probabilities": ans.get("probabilities"),
                        # OpenJev: (top prob - 1/k) / (1 - 1/k), so 0 = uniform, 1 = certain
                        "confidence": ans.get("confidence"),
                        "latency_ms": (time.perf_counter() - t0) * 1000,
                        "server_ms": 1000 * float((j.get("metadata") or {}).get("inference_seconds") or 0.0),
                        "cost": float((j.get("usage") or {}).get("cost") or 0.0),
                    }
                if r.status_code != 429 and r.status_code < 500:
                    raise RuntimeError(f"{name}: HTTP {r.status_code}: {r.text[:500]}")
                err = f"HTTP {r.status_code}"
            time.sleep(min(60, 2**attempt))  # 429 busy / 5xx restarting / network error
        raise RuntimeError(f"{name}: gave up on {qid} after retries ({err})")

    todo = [(qid, row["query"]) for qid, row in q.iterrows() if qid not in done]
    pool = ThreadPoolExecutor(workers)
    try:
        with partial.open("a") as log:
            for f in as_completed([pool.submit(one, qid, text) for qid, text in todo]):
                qid, d = f.result()
                done[qid] = d
                log.write(json.dumps({"qid": qid, **d}) + "\n")
                log.flush()
                if len(done) % 100 == 0:
                    print(f"  {name} {len(done)}/{len(q)}", flush=True)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return {qid: done[qid] for qid in q.index}


def select_jev(q: pd.DataFrame, model: str, workers: int, setup: str, partial: Path) -> dict:
    _load_dotenv()
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise SystemExit("OPENROUTER_API_KEY not set (export it or put it in .env next to bench.py)")
    headers = {"Authorization": f"Bearer {key}", "X-OpenRouter-Title": "xroutebench-tier"}
    return _systemone(q, "jev", "https://openrouter.ai/api/alpha/decisions", headers, {"model": model},
                      workers, setup, partial)


# Open-Jev 9B, hosted by the team (see openjev_guid.md). Plain HTTP and every request is logged
# server-side, so send only public benchmark text. It serves one request at a time.
OPENJEV_URL = "http://104.198.58.212:8080/v1/systemone"


def select_openjev(q: pd.DataFrame, url: str, workers: int, setup: str, partial: Path) -> dict:
    import requests

    _load_dotenv()
    key = os.environ.get("OPENJEV_API_KEY", "").strip()
    if not key:
        raise SystemExit("OPENJEV_API_KEY not set (export it or put it in .env next to bench.py)")
    print(f"  openjev health: {requests.get(url.rsplit('/v1/', 1)[0] + '/health', timeout=10).text[:200]}")
    return _systemone(q, "openjev", url, {"Authorization": f"Bearer {key}"}, {}, workers, setup, partial)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def oracle_tier(row) -> str:
    """Cheapest tier that reaches the best performance any tier gets on this query."""
    best = max(row[f"perf_{t}"] for t in TIERS)
    return next(t for t in reversed(TIERS) if row[f"perf_{t}"] >= best)


# Glossary written at the top of every report_<n>_<seed>.md.
REPORT_TERMS = """## Terms used in this report

| Term | What it means |
|---|---|
| `selector` | The dispatcher being scored: a rule (`always T1`, `random`), the `oracle`, or a model (`nasiko`, `laya-…`, `openjev-…`, `laya-tuned-…`) |
| `always T1` / `always T2` / `always T3` | Send every question to that one tier |
| `oracle` | The perfect dispatcher: knows the answers and picks the ideal tier, the cheapest one that reaches the best score any tier gets |
| `random` | A random tier for each question |
| `… +T1<0.7` | The row above it, with every pick whose top probability is below 0.7 sent to T1 instead (`--escalate-below`) |
| `perf` | Average score of the models picked. For most tasks it's the share of questions answered correctly |
| `quality_kept` | `perf` divided by always-T1's `perf`: accuracy kept compared with always using T1. It can exceed 100% |
| `cost_per_1k` | Dollars to answer 1,000 questions with the models picked (the answering cost, from the dataset's tokens × prices) |
| `cost_saved` | 1 − `cost_per_1k` ÷ always-T1's `cost_per_1k`: money saved compared with always using T1 |
| `oracle_match` | Share of questions where the pick equals the oracle's (the ideal pick rate) |
| `%T1` / `%T2` / `%T3` | Share of questions sent to each tier (the mix) |
| `p50_ms` | Median time per decision, in milliseconds (0 for rules) |
| `selector_$` | What the decisions themselves cost. $0 for local models and rules |
| `n` (per-task table) | Number of questions of that task |
| `oracle %T1` (per-task table) | Share of that task's questions where T1 is the ideal tier |
| `<selector> %T1` (per-task table) | Share of that task's questions the selector sent to T1 |

The fairest single comparison, accuracy against a blind split at the same cost, is worked out in
LAYA_RESULTS.md and OPENJEV_RESULTS.md."""


def truth_fields(row, tier: str) -> dict:
    """What the benchmark recorded for this query, next to a selector's pick.

    ideal_tier = oracle_tier; outcome: ideal | too_small (cheaper than ideal, lost score) |
    too_big (dearer than ideal, same score) | too_big_lost (dearer and still lost score).
    """
    scores = {t: float(row[f"perf_{t}"]) for t in TIERS}
    ideal = oracle_tier(row)
    if tier == ideal:
        outcome = "ideal"
    elif TIERS.index(tier) > TIERS.index(ideal):
        outcome = "too_small"
    else:
        outcome = "too_big" if scores[tier] >= max(scores.values()) else "too_big_lost"
    return {"task": row["task_name"], "ideal_tier": ideal, "outcome": outcome,
            "pick_score": scores[tier], "scores": scores}


def evaluate(q: pd.DataFrame, name: str, decisions: dict) -> dict:
    tiers = q.index.map(lambda i: decisions[i]["tier"])
    perf = np.array([q.at[i, f"perf_{t}"] for i, t in zip(q.index, tiers)])
    cost = np.array([q.at[i, f"cost_{t}"] for i, t in zip(q.index, tiers)])
    lat = np.array([decisions[i].get("latency_ms", 0.0) for i in q.index])
    sel_cost = sum(decisions[i].get("cost", 0.0) for i in q.index)
    mix = pd.Series(tiers).value_counts(normalize=True)
    return {
        "selector": name,
        "perf": perf.mean(),
        "cost_per_1k": cost.mean() * 1000,
        "oracle_match": np.mean([t == o for t, o in zip(tiers, q["oracle"])]),
        **{f"%{t}": mix.get(t, 0.0) for t in TIERS},
        "p50_ms": float(np.median(lat)),
        "selector_$": sel_cost,
        "_tiers": list(tiers),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=500, help="queries to sample (0 = full test split)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selectors", default="nasiko,nasiko_online,laya,jev")
    ap.add_argument("--jev-model", default="~typesafe/jev-latest")
    ap.add_argument("--jev-workers", type=int, default=8)
    ap.add_argument("--openjev-url", default=OPENJEV_URL)
    ap.add_argument("--openjev-workers", type=int, default=2, help="the server handles one request at a time")
    ap.add_argument("--laya-subfolder", default=None, help="e.g. typed-decisions, multilingual")
    ap.add_argument("--escalate-below", type=float, default=None,
                    help="also report each selector that returns probabilities with every pick whose "
                    "top probability is below this sent to T1 (e.g. 0.7)")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    q = load_queries(args.n, args.seed)
    q["oracle"] = q.apply(oracle_tier, axis=1)
    print(f"{len(q)} queries across {q.task_name.nunique()} tasks; tiers = {TIER_MODELS}")

    rows = []
    for t in TIERS:
        rows.append(evaluate(q, f"always {t}", {i: {"tier": t} for i in q.index}))
    rows.append(evaluate(q, "oracle", {i: {"tier": o} for i, o in q["oracle"].items()}))
    rng = np.random.default_rng(args.seed)
    rows.append(evaluate(q, "random", {i: {"tier": rng.choice(TIERS)} for i in q.index}))

    for sel in [s.strip() for s in args.selectors.split(",") if s.strip()]:
        tag = sel if sel != "laya" or not args.laya_subfolder else f"laya-{args.laya_subfolder}"
        cache = RESULTS / f"decisions_{tag}_{args.n}_{args.seed}.jsonl"
        partial = cache.with_suffix(".partial.jsonl")  # API selectors append here as answers arrive
        if cache.exists() and not sel.startswith("nasiko"):
            decisions = {d["qid"]: d for d in map(json.loads, cache.read_text().splitlines())}
            print(f"{tag}: {len(decisions)} cached decisions from {cache.name}")
        else:
            print(f"{tag}: running…")
            if sel == "nasiko":
                decisions = select_nasiko(q, args.seed, online=False)
            elif sel == "nasiko_online":
                decisions = select_nasiko(q, args.seed, online=True)
            elif sel == "laya" or sel.startswith("laya-"):  # e.g. laya-multilingual-sizelabels
                decisions = select_laya(q, *parse_setup(tag, "laya"))
            elif sel == "jev" or sel.startswith("jev-"):
                decisions = select_jev(q, args.jev_model, args.jev_workers, parse_setup(tag, "jev")[1], partial)
            elif sel == "openjev" or sel.startswith("openjev-"):
                decisions = select_openjev(q, args.openjev_url, args.openjev_workers, parse_setup(tag, "openjev")[1], partial)
            else:
                raise SystemExit(f"unknown selector {sel!r}")
            cache.write_text("".join(json.dumps({"qid": k, **v, **truth_fields(q.loc[k], v["tier"])}) + "\n"
                                     for k, v in decisions.items()))
            partial.unlink(missing_ok=True)
        rows.append(evaluate(q, tag, decisions))
        if args.escalate_below is not None and all(d.get("probabilities") for d in decisions.values()):
            thr = args.escalate_below
            escalated = {
                i: {**d, "tier": "T1" if max(d["probabilities"].values()) < thr else d["tier"]}
                for i, d in decisions.items()
            }
            rows.append(evaluate(q, f"{tag} +T1<{thr:g}", escalated))

    t1 = rows[0]
    report = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]).set_index("selector")
    report.insert(1, "quality_kept", report.perf / t1["perf"])
    report.insert(3, "cost_saved", 1 - report.cost_per_1k / t1["cost_per_1k"])

    fmt = report.copy()
    for c in ["quality_kept", "cost_saved", "oracle_match", "%T1", "%T2", "%T3"]:
        fmt[c] = (fmt[c] * 100).map("{:.1f}%".format)
    fmt["perf"] = fmt.perf.map("{:.4f}".format)
    fmt["cost_per_1k"] = fmt.cost_per_1k.map("${:.4f}".format)
    fmt["p50_ms"] = fmt.p50_ms.map("{:.1f}".format)
    fmt["selector_$"] = fmt["selector_$"].map("${:.4f}".format)

    # Per-task: share of queries each selector sent to T1 vs where the oracle needs T1.
    per_task = pd.DataFrame({"n": q.groupby("task_name").size(), "oracle %T1": q.groupby("task_name").oracle.apply(lambda s: (s == "T1").mean())})
    for r in rows[5:]:
        per_task[f"{r['selector']} %T1"] = pd.Series(r["_tiers"], index=q.index).eq("T1").groupby(q.task_name).mean()
    per_task_fmt = per_task.copy()
    for c in per_task.columns[1:]:
        per_task_fmt[c] = (per_task[c] * 100).map("{:.0f}%".format)

    out = [
        f"# xRouteBench tier selection (llmrouter_generic test, n={len(q)}, seed={args.seed})",
        "",
        "Tiers: " + ", ".join(f"{t}={m}" for t, m in TIER_MODELS.items()),
        "",
        REPORT_TERMS,
        "",
        "## Results",
        "",
        fmt.to_markdown(),
        "",
        "## Share of queries sent to T1, per task",
        "",
        per_task_fmt.to_markdown(),
    ]
    text = "\n".join(out)
    print("\n" + text)
    (RESULTS / f"report_{args.n}_{args.seed}.md").write_text(text + "\n")
    per_q = q.drop(columns=["query"]).copy()
    for r in rows[5:]:
        per_q[r["selector"]] = r["_tiers"]
    per_q.to_csv(RESULTS / f"per_query_{args.n}_{args.seed}.csv")


if __name__ == "__main__":
    main()
