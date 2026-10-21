# Tier routing report

> **TL;DR** — No router tested picks the right model tier reliably. The best is a tie at **~48%** balanced accuracy, where chance is 33%: **d3-flash zero-shot** and **Jev with a cross-validated threshold**. Nasiko's current router is at chance. Models fine-tuned on the older `eval_600` training set do worse: its tiers were written by hand and overstate difficulty, while the test tiers here were measured with real LLMs.

## 📁 Data
 
- **Test set:** 600 questions (300 written by us + 300 from OpenAssistant). Their tiers were **measured with real LLMs, not written by hand**: LOW **376** / MID **140** / HIGH **84**.
- **How the tiers were measured:** a question's tier is the cheapest one whose models all answered correctly. A blind judge (Claude Sonnet 5.5) graded the answers.

| Tier | Models |
|---|---|
| **LOW** | `gpt-4.1-nano` + `Qwen3.5 9B` |
| **MID** | `gpt-4.1` + `Gemma 4 26B-A4B` |
| **HIGH** | `gpt-5.4` |

- **Training data (fine-tuned models only):** `eval_600`, an older set for tier selection

## 📊 Results

| Router | Bal. % | Acc. % | Recall % L/M/H | Precision % L/M/H | Under % | Over % | Router $/1k | p50 | Size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **d3-flash zero-shot**\* (tie) | **48.3** | 50.5 | 46/85/14 | 87/31/75 | 15.2 | 34.3 | 0 (GPU 17 GB) | 682 ms | 8.4 B |
| **Jev + threshold**† (tie) | **48.6** | 54.0 | 59/54/33 | 84/38/20 | 14.3 | 31.7 | 0.022 (API) | 299 ms | n/p |
| Jev zero-shot | 45.6 | **59.5** | 70/65/1 | 79/35/100 | 22.0 | **18.5** | 0.022 (API) | 299 ms | n/p |
| d3-flash fine-tuned | 43.6 | 35.8 | 29/36/66 | 86/20/25 | 6.5 | 57.7 | 0 (GPU 17 GB) | ~0.7 s | 8.4 B |
| CLM fine-tuned | 41.6 | 32.5 | 26/25/74 | 84/20/20 | **5.5** | 62.0 | 0 (GPU 16 GB) | 114 ms | 8.2 B |
| Laya fine-tuned‡ | 38.6 | 29.5 | 15/62/38 | 79/22/24 | 9.5 | 61.0 | 0 (GPU 1 GB) | 46 ms | 421 M |
| PR-292 | 37.9 | 29.0 | 21/32/61 | 73/20/19 | 8.3 | 62.7 | 0 (CPU) | 2.8 ms | 23 M |
| PR-296 semantic | 37.0 | 28.5 | 20/35/56 | 74/21/18 | 9.2 | 62.3 | 0 (CPU) | 4.1 ms | 33 M |
| PR-296 lexical | 35.0 | 25.3 | 16/28/61 | 77/19/16 | 7.7 | 67.0 | 0 (CPU) | 0.2 ms | < 1 M |
| Nasiko main | 33.4 | 42.2 | 50/38/12 | 62/21/22 | 25.0 | 32.8 | 0 (CPU) | < 0.1 ms | rules |
| *Always LOW* | 33.3 | 62.7 | 100/0/0 | 63/—/— | 37.3 | 0 | — | — | — |
| *Always HIGH* | 33.3 | 14.0 | 0/0/100 | —/—/14 | 0 | 86.0 | — | — | — |
| ***Perfect routing*** | *100* | *100* | *100/100/100* | *100/100/100* | *0* | *0* | — | — | — |

**Columns:**
- **Bal.:** balanced accuracy, the average recall over the three tiers; 33.3% is chance.

  ```
  Bal. = (Recall LOW + Recall MID + Recall HIGH) / 3
  Recall of a tier = correct picks of that tier / questions that really are that tier
  ```

  **Example (d3-flash zero-shot):** recall is 172/376 = 45.7% for LOW, 119/140 = 85.0% for MID and 12/84 = 14.3% for HIGH.
  So Bal. = (45.7 + 85.0 + 14.3) / 3 = **48.3%**.

- **Recall:** of the questions that really are that tier, the share routed there.
- **Precision:** of the questions routed to a tier, the share that really belong there.
- **Under:** sent to a cheaper tier than measured, so the answer is at risk.
- **Over:** sent to a dearer tier than needed, so money is wasted.
- **Router $/1k:** what the **decision model itself** costs to make 1,000 tier decisions; the LLM that then answers isn't counted.
  - **Jev** is a paid API: $0.013 for our 600 calls, about **$0.022 per 1,000**.
  - **Every other router is self-hosted,** with no per-call fee. Its cost is the hardware it needs, shown in brackets: a GPU with about 17 GB (d3-flash), 16 GB (CLM) or 1 GB (Laya) of memory, or just a CPU (PR models, Nasiko).
- **p50:** median time to pick a tier for one question. It was measured on a Mac (M5 Pro), except Jev, which is the API round trip. d3-flash and CLM ran without their fast GPU serving stacks, so they'd be quicker in production.
- **Size:** parameters; n/p = not published.

‡ The best of the three Laya fine-tunes (typed-decisions). Laya fine-tunes vary from run to run on the Mac GPU.

† **Threshold (Jev):**
- **The rule:** if Jev's confidence in its pick is below **0.7**, send the question to HIGH instead.
- **How it was chosen:** by 5-fold cross-validation, so no question is scored with a threshold picked using its own label. Four of the 5 folds chose 0.6–0.7; the fifth chose a cascade at 0.9.
- **Effect:** it raises Jev from 45.6% to 48.6%. Tuned on these same labels it would reach 51.1%, but that number is optimistic.

\* **d3-flash is shown without a threshold, because none helped it.**
- **Every threshold tried scored the same or lower:** confidence < 0.3–0.95 → HIGH, and cascade cut-offs 0.3–0.9.
- **The best of them** (confidence < 0.3 → HIGH) also scores 48.3%. It gets there by sending 399 of 600 questions to HIGH: over 61.8%.
- **Chosen by cross-validation,** a threshold drops it to 45.0%.
- **So its plain pick is its best result.**

## 🔍 Conclusions

1. **No router is good enough yet.** The best two are statistically tied: Jev is ahead by only 0.3 points, 95% interval −4.3 to +5.0.

2. **Nasiko's router has no difficulty signal.** It picks by regex request type plus random sampling.


