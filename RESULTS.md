# Tier classifiers on the measured 600 set

> **TL;DR** — 16 classifier runs tested on 600 questions whose tiers were **measured** with real models (LOW 376 / MID 140 / HIGH 84). The set is unbalanced, so models are ranked by **balanced accuracy** (mean recall over the three tiers; chance = 33.3%). Always answering LOW gets 62.7% plain accuracy but only 33.3% balanced. Best balanced accuracy: **48.3%**. Terms are explained in the [Glossary](#-glossary) at the end.

*Last updated: 21 Oct 2026 15:16*

## 🔍 Findings

> [!IMPORTANT]
> **No classifier routes the measured tiers well.** The best balanced accuracy is <span style="color:#9a6700">**48.3%**</span> (d3-flash zero-shot), against 33.3% for chance. On the authored 300 set, the same fine-tuned d3-flash and CLM scored 76.7% and 73.7%.

**1. Zero-shot decision models do best.**
- **d3-flash zero-shot (48.3%)** and **Jev (45.6%)** tell LOW from MID reasonably well.
- But they almost never pick HIGH: recall on HIGH is **14.3%** for d3-flash and **1.2%** for Jev.

**2. Fine-tuning on the authored labels doesn't transfer.** Every model trained on `eval_600` (authored labels) over-picks HIGH. For example:

| Model | Picks HIGH (84 are really HIGH) | Too big |
|---|---:|---:|
| `d3-flash-finetuned` | 222 | 57.7% |
| `clm-finetuned` | 310 | 62.0% |
| `pr-296-lexical` | 317 | 67.0% |
| `laya-english-finetuned` | 237 | 63.5% |

- Most questions an author would call MID or HIGH are solved by today's LOW models. A model trained to copy the authored labels sends them up a tier or two.
- For d3-flash, fine-tuning **lowers** balanced accuracy (48.3% → 43.6%). For CLM it raises balanced accuracy (38.9% → 41.6%), but plain accuracy falls from 51.3% to 32.5%.

**3. The retrained PR classifiers sit near chance (35.0–37.9%).** They see only the question text, so they can only copy the authored labels' surface patterns.

**4. Nasiko's current router (main) is chance: 33.4% cold, 33.8% online.** It has no difficulty signal at all.
- **How it picks:** a regex request type, then a Thompson-sampled tier from cold-start priors. The tier depends only on the request type and random draws.
- **Online feedback:** even with best-case, instant feedback from the measured labels, it doesn't improve. Feedback counts any tier at or above the label as right, so it drifts to MID for almost everything (506 of 600 picks). MID is "right" for 86% of questions and cheaper than HIGH.
- **Measurement:** main's own Rust `routing::classify`, commit `c6ebd4b3`, seeds 0–4 (spread under 1 point balanced).

**5. Zero-shot Laya is chance.** All three bases pick MID for almost every question.
- **Fine-tuned,** the best is **38.6%** (typed-decisions).
- **Laya's fine-tunes don't repeat exactly on the Mac GPU,** even with the same seed. A rerun changed English from 37.9% to 37.7% (479 of 600 picks the same) and typed-decisions from 35.0% to 38.6% (427 the same).
- **Every other run reproduced exactly:** all zero-shot runs, Laya multilingual's fine-tune, fine-tuned d3-flash and CLM, and the PR models.

**6. Best model: a tie between d3-flash zero-shot and Jev with a threshold.**
- **As answered,** d3-flash zero-shot leads: 48.3% against Jev's 45.6%.
- **With a threshold chosen honestly by 5-fold cross-validation,** Jev reaches **48.6%**. The rule and threshold are picked on 4 folds and applied to the 5th. Jev mostly picks confidence < 0.6–0.7 → high.
- **A bootstrap test can't separate them.** The difference is +0.3 points with a 95% interval from −4.3 to +5.0, and Jev comes out ahead in 54% of resamples, about a coin flip.
- **Each score is uncertain by itself too:** d3-flash's 95% interval is 44.7–51.8%, and Jev's is 44.1–53.4%.
- **d3-flash gains nothing from thresholds:** chosen by cross-validation, it drops to 45.0%.

**7. Thresholds help only a little.** Escalating unsure picks to HIGH, or using a cascade cut-off, gains at most about 5 points (see the threshold tables below):
- **Jev:** best is 51.1% on these labels (confidence < 0.7 → high), 48.6% chosen honestly, against 45.6% as answered.
- **Zero-shot Laya:** a threshold lifts English and typed-decisions from chance to about 37%. They pick MID for everything, so escalating unsure picks to HIGH is all they can gain.
- **d3-flash fine-tuned:** best is 46.1% (cascade 0.3).
- **d3-flash zero-shot and CLM zero-shot:** no threshold beats the plain pick by more than 0.5 points.
- **The thresholds that won on the authored 300 set don't transfer.** Jev's 0.95, which scored 66.3% there, gets 46.5% here. d3-flash's 0.5 gets 38.2%, below its plain 48.3%.

> [!TIP]
> **The training labels are the bottleneck, not the models.** The next step is to train on **measured** labels: for example, d3-flash's readout or CLM's heads with cross-validation on this 600 set, or new measured training data. Then test again.

> [!NOTE]
> - **Latency:**
>   - d3-flash runs on the Mac GPU without its ROCm kernels.
>   - CLM's encoder runs in PyTorch, not vLLM.
>   - `d3-flash-finetuned` latency counts the readout only. The 9B backbone pass adds about 0.7 s (see `d3-flash-zeroshot`).
> - **Jev** cost `$0.013` for 600 calls.
> - **CLM's setup** reproduces the earlier session's numbers exactly: same commit `d5f9ef0`, and the same check output.

## 📊 Results

> [!TIP]
> **Best model: `d3-flash-zeroshot`**: 48.3% balanced accuracy (50.5% accuracy, recall low / mid / high 45.7 / 85 / 14.3%), as answered.
> **Best with a threshold chosen by cross-validation: `jev-direct`**: 48.6% balanced (54.0% accuracy). The two are a statistical tie (see Findings). Both are highlighted.

| Model | Kind | Params | **Balanced acc.** | Accuracy | Recall low / mid / high | Too small | Too big | Picks low / mid / high | p50 latency |
|---|---|---:|---:|---:|---|---:|---:|---|---:|
| *Always LOW (baseline)* | — | — | 33.3% | 62.7% | 100 / 0 / 0% | 37.3% | 0.0% | 600/0/0 | — |
| <span style="color:#1a7f37">**d3-flash-zeroshot**</span> (best) | **Decision model, zero-shot (local)** | **8.39 B** | **48.3%** | **50.5%** | **45.7 / 85 / 14.3%** | **15.2%** | **34.3%** | **198/386/16** | **681.9 ms** |
| ↳ *threshold chosen by cross-validation (honest)* | *5-fold, chosen on the other folds* | 8.39 B | *45.0%* | *40.0%* | *32.7 / 55.7 / 46.4%* | *9.8%* | *50.2%* | *140/277/183* | 681.9 ms |
| ↳ *best threshold on these labels: none beats the plain pick* | — | | | | | | | | |
| **jev-direct** | Decision model, zero-shot (hosted) | not published | 45.6% | 59.5% | 70.5 / 65 / 1.2% | 22.0% | 18.5% | 336/263/1 | 299 ms |
| ↳ <span style="color:#1a7f37">**threshold chosen by cross-validation (honest)**</span> (tied best) | **5-fold, chosen on the other folds** | **not published** | **48.6%** | **54.0%** | **58.8 / 53.6 / 33.3%** | **14.3%** | **31.7%** | **264/195/141** | **299 ms** |
| ↳ *best threshold on these labels: confidence < 0.7 → high* | *optimistic: picked with the test labels* | not published | *51.1%* | *54.8%* | *59.8 / 45.7 / 47.6%* | *13.0%* | *32.2%* | *272/150/178* | 299 ms |
| **d3-flash-finetuned** | Decision model, fine-tuned (local) | 8.39 B | 43.6% | 35.8% | 29 / 36.4 / 65.5% | 6.5% | 57.7% | 127/251/222 | <0.1 ms |
| ↳ *threshold chosen by cross-validation (honest)* | *5-fold, chosen on the other folds* | 8.39 B | *44.1%* | *43.0%* | *39.9 / 54.3 / 38.1%* | *11.2%* | *45.8%* | *175/298/127* | <0.1 ms |
| ↳ *best threshold on these labels: cascade 0.3* | *optimistic: picked with the test labels* | 8.39 B | *46.1%* | *46.0%* | *43.4 / 59.3 / 35.7%* | *11.7%* | *42.3%* | *190/305/105* | <0.1 ms |
| **clm-finetuned** | Decision model, fine-tuned (local) | 8.19 B | 41.6% | 32.5% | 26.1 / 25 / 73.8% | 5.5% | 62.0% | 117/173/310 | 114.3 ms |
| ↳ *threshold chosen by cross-validation (honest)* | *5-fold, chosen on the other folds* | 8.19 B | *41.4%* | *36.5%* | *32.7 / 34.3 / 57.1%* | *9.2%* | *54.3%* | *152/222/226* | 114.3 ms |
| ↳ *best threshold on these labels: cascade 0.3* | *optimistic: picked with the test labels* | 8.19 B | *42.6%* | *38.5%* | *35.1 / 37.9 / 54.8%* | *9.7%* | *51.8%* | *162/230/208* | 114.3 ms |
| **clm-zeroshot** | Decision model, zero-shot (local) | 8.19 B | 38.9% | 51.3% | 68.1 / 20 / 28.6% | 24.3% | 24.3% | 388/106/106 | 115.7 ms |
| ↳ *threshold chosen by cross-validation (honest)* | *5-fold, chosen on the other folds* | 8.19 B | *36.9%* | *39.3%* | *47.3 / 8.6 / 54.8%* | *17.3%* | *43.3%* | *275/48/277* | 115.7 ms |
| ↳ *best threshold on these labels: confidence < 0.3 → high* | *optimistic: picked with the test labels* | 8.19 B | *39.4%* | *40.8%* | *48.4 / 7.9 / 61.9%* | *15.8%* | *43.3%* | *273/39/288* | 115.7 ms |
| **laya-typed-decisions-finetuned** | Decision model, fine-tuned (local) | 421 M | 38.6% | 29.5% | 15.4 / 62.1 / 38.1% | 9.5% | 61.0% | 73/395/132 | 46.2 ms |
| ↳ *threshold chosen by cross-validation (honest)* | *5-fold, chosen on the other folds* | 421 M | *42.0%* | *27.3%* | *11.2 / 45.7 / 69%* | *5.2%* | *67.5%* | *53/294/253* | 46.2 ms |
| ↳ *best threshold on these labels: confidence < 0.6 → high* | *optimistic: picked with the test labels* | 421 M | *42.0%* | *27.3%* | *11.2 / 45.7 / 69%* | *5.2%* | *67.5%* | *53/294/253* | 46.2 ms |
| **pr-292** | PR, retrained | 22.7 M | 37.9% | 29.0% | 20.7 / 32.1 / 60.7% | 8.3% | 62.7% | 107/230/263 | 2.8 ms |
| ↳ *threshold chosen by cross-validation (honest)* | *5-fold, chosen on the other folds* | 22.7 M | *37.1%* | *30.8%* | *25 / 32.9 / 53.6%* | *10.0%* | *59.2%* | *130/235/235* | 2.8 ms |
| ↳ *best threshold on these labels: cascade 0.3* | *optimistic: picked with the test labels* | 22.7 M | *38.9%* | *32.7%* | *26.6 / 36.4 / 53.6%* | *10.0%* | *57.3%* | *136/240/224* | 2.8 ms |
| **laya-english-finetuned** | Decision model, fine-tuned (local) | 421 M | 37.7% | 28.7% | 19.9 / 33.6 / 59.5% | 7.8% | 63.5% | 101/262/237 | 44.6 ms |
| ↳ *threshold chosen by cross-validation (honest)* | *5-fold, chosen on the other folds* | 421 M | *37.0%* | *31.3%* | *22.3 / 52.9 / 35.7%* | *11.7%* | *57.0%* | *119/338/143* | 44.6 ms |
| ↳ *best threshold on these labels: cascade 0.4* | *optimistic: picked with the test labels* | 421 M | *38.9%* | *31.5%* | *21.3 / 51.4 / 44%* | *10.0%* | *58.5%* | *107/327/166* | 44.6 ms |
| **pr-296-semantic** | PR, retrained | 33.4 M | 37.0% | 28.5% | 19.9 / 35 / 56% | 9.2% | 62.3% | 101/235/264 | 4.1 ms |
| ↳ *threshold chosen by cross-validation (honest)* | *5-fold, chosen on the other folds* | 33.4 M | *36.4%* | *30.2%* | *21.5 / 47.1 / 40.5%* | *11.2%* | *58.7%* | *106/301/193* | 4.1 ms |
| ↳ *best threshold on these labels: cascade 0.4* | *optimistic: picked with the test labels* | 33.4 M | *38.2%* | *31.7%* | *22.1 / 52.1 / 40.5%* | *11.3%* | *57.0%* | *109/318/173* | 4.1 ms |
| **pr-296-lexical** | PR, retrained | < 1 M (no network) | 35.0% | 25.3% | 16.5 / 27.9 / 60.7% | 7.7% | 67.0% | 81/202/317 | 0.2 ms |
| ↳ *threshold chosen by cross-validation (honest)* | *5-fold, chosen on the other folds* | < 1 M (no network) | *37.2%* | *31.5%* | *23.1 / 49.3 / 39.3%* | *11.5%* | *57.0%* | *115/303/182* | 0.2 ms |
| ↳ *best threshold on these labels: cascade 0.3* | *optimistic: picked with the test labels* | < 1 M (no network) | *37.2%* | *31.5%* | *23.1 / 49.3 / 39.3%* | *11.5%* | *57.0%* | *115/303/182* | 0.2 ms |
| **nasiko-main-online** | Nasiko router (main) | 0 (rules) | 33.8% | 28.7% | 13.6 / 84.3 / 3.6% | 16.7% | 54.7% | 81/506/13 | <0.1 ms |
| **nasiko-main-cold** | Nasiko router (main) | 0 (rules) | 33.4% | 42.2% | 50.5 / 37.9 / 11.9% | 25.0% | 32.8% | 305/249/46 | <0.1 ms |
| **laya-english-zeroshot** | Decision model, zero-shot (local) | 421 M | 33.3% | 23.3% | 0 / 100 / 0% | 14.0% | 62.7% | 0/600/0 | 45 ms |
| ↳ *threshold chosen by cross-validation (honest)* | *5-fold, chosen on the other folds* | 421 M | *37.3%* | *18.7%* | *0 / 32.1 / 79.8%* | *2.8%* | *78.5%* | *0/179/421* | 45 ms |
| ↳ *best threshold on these labels: confidence < 0.8 → high* | *optimistic: picked with the test labels* | 421 M | *37.3%* | *18.7%* | *0 / 32.1 / 79.8%* | *2.8%* | *78.5%* | *0/179/421* | 45 ms |
| **laya-multilingual-finetuned** | Decision model, fine-tuned (local) | 322 M | 33.1% | 25.3% | 15.7 / 40.7 / 42.9% | 11.8% | 62.8% | 95/280/225 | 19.6 ms |
| ↳ *threshold chosen by cross-validation (honest)* | *5-fold, chosen on the other folds* | 322 M | *30.0%* | *20.3%* | *10.1 / 30 / 50%* | *10.2%* | *69.5%* | *66/214/320* | 19.6 ms |
| ↳ *best threshold on these labels: cascade 0.9* | *optimistic: picked with the test labels* | 322 M | *34.0%* | *14.7%* | *0 / 4.3 / 97.6%* | *0.3%* | *85.0%* | *0/18/582* | 19.6 ms |
| **laya-typed-decisions-zeroshot** | Decision model, zero-shot (local) | 421 M | 32.9% | 23.0% | 0 / 98.6 / 0% | 14.0% | 63.0% | 0/597/3 | 44.3 ms |
| ↳ *threshold chosen by cross-validation (honest)* | *5-fold, chosen on the other folds* | 421 M | *37.5%* | *21.7%* | *0 / 63.6 / 48.8%* | *7.2%* | *71.2%* | *0/395/205* | 44.3 ms |
| ↳ *best threshold on these labels: confidence < 0.6 → high* | *optimistic: picked with the test labels* | 421 M | *37.5%* | *21.7%* | *0 / 63.6 / 48.8%* | *7.2%* | *71.2%* | *0/395/205* | 44.3 ms |
| **laya-multilingual-zeroshot** | Decision model, zero-shot (local) | 322 M | 31.4% | 23.0% | 3.5 / 87.1 / 3.6% | 14.5% | 62.5% | 25/536/39 | 20.1 ms |
| ↳ *threshold chosen by cross-validation (honest)* | *5-fold, chosen on the other folds* | 322 M | *31.8%* | *18.2%* | *2.4 / 39.3 / 53.6%* | *7.0%* | *74.8%* | *16/251/333* | 20.1 ms |
| ↳ *best threshold on these labels: confidence < 0.95 → high* | *optimistic: picked with the test labels* | 322 M | *33.4%* | *14.5%* | *0 / 5 / 95.2%* | *0.7%* | *84.8%* | *0/41/559* | 20.1 ms |

## 📏 Model sizes

Parameter counts are read from the weight files on disk, and disk sizes are measured. Sizes that aren't published are marked so.

### Tier classifiers tested here

| Model | What's inside | **Parameters** | Precision | Disk | Runs on |
|---|---|---:|---|---:|---|
| **d3-flash** | Qwen3.5-based decision model | **8.39 B** | bf16 | 16.8 GB | Mac GPU, ~16 GB RAM |
| **CLM** | Qwen3-8B encoder + 2 projection heads | **8.19 B** + small heads | bf16 | 16.4 GB + 0.08 GB | Mac GPU, ~16 GB RAM |
| **Jev** | Hosted by TypeSafe | **not published** | — | — | TypeSafe API |
| **Laya English** | ModernBERT-large encoder (28 layers, width 1024) + decision head | **421 M** | fp16 | 0.80 GB | Mac GPU |
| **Laya typed-decisions** | Same architecture as English | **421 M** | fp16 | 0.80 GB | Mac GPU |
| **Laya multilingual** | A smaller multilingual encoder | **322 M** | fp16 | 0.61 GB | Mac GPU |
| **PR-296 semantic** | bge-small-en-v1.5 + logistic regression | **33.4 M** (model card) + LR | fp32 (ONNX) | ~0.13 GB | CPU |
| **PR-292** | all-MiniLM-L6-v2 + tier head | **22.7 M** | fp32 | 0.09 GB | CPU |
| **PR-296 lexical** | N-gram features + logistic regression | thousands of weights, no neural network | — | < 1 MB | CPU |
| **Nasiko main** | Regexes + Thompson sampling over 3 tiers | **0**: rules and a random draw | — | — | CPU (Rust) |

The fine-tuned versions are the same size: d3-flash and CLM retrain only small parts (readout, heads), and Laya retrains its top 8 layers in place.

### Models used to measure the labels

| Model | Role | **Parameters** | Disk / access |
|---|---|---:|---|
| `gpt-4.1-nano` | LOW, OpenAI | not published | API |
| **`Qwen3.5 9B`** | LOW, open-source | ~9 B | 5.98 GB (MLX 4-bit) |
| `gpt-4.1` | MID, OpenAI | not published | API |
| **`Gemma 4 26B-A4B`** | MID, open-source | **26 B total, ~4 B active** per token (mixture-of-experts) | ~15.3 GB (MLX 4-bit) |
| `gpt-5.4` | HIGH, OpenAI | not published | API |
| Claude Sonnet 5.5 | Blind judge | not published | Claude plan |

> [!NOTE]
> - **The best classifiers are the big ones.** d3-flash (8.4 B) and Jev (size unknown) lead at 46–48% balanced accuracy. But they're 20–400× larger than Laya or the PR models, and still well short of good routing.
> - **Deciding can cost as much as answering:** d3-flash is about the size of `Qwen3.5 9B`, one of the LOW answer models.
> - **The PR models are tiny (23–33 M) and fast** (a few ms), but sit near chance on measured labels.

## 🎚️ Results at different thresholds

Every model that gives tier probabilities is thresholded: all runs except Nasiko's router, which only samples a tier. Where a model has no confidence of its own, its top probability is used.

### Best per model

| Model | As answered | **Cross-validated threshold (honest)** | Best on these labels (optimistic) | Best rule and threshold | 300-set pick (honest) |
|---|---:|---:|---:|---|---:|
| **d3-flash-zeroshot** | 48.3% | 45.0% | 48.3% | As answered, — | 38.2% |
| **jev-direct** | 45.6% | **48.6%** | 51.1% | Confidence < X → high, 0.7 | 46.5% |
| **d3-flash-finetuned** | 43.6% | 44.1% | 46.1% | Cascade, 0.3 | — |
| **clm-finetuned** | 41.6% | 41.4% | 42.6% | Cascade, 0.3 | — |
| **clm-zeroshot** | 38.9% | 36.9% | 39.4% | Confidence < X → high, 0.3 | 37.4% |
| **laya-typed-decisions-finetuned** | 38.6% | 42.0% | 42.0% | Confidence < X → high, 0.6 | — |
| **pr-292** | 37.9% | 37.1% | 38.9% | Cascade, 0.3 | — |
| **laya-english-finetuned** | 37.7% | 37.0% | 38.9% | Cascade, 0.4 | — |
| **pr-296-semantic** | 37.0% | 36.4% | 38.2% | Cascade, 0.4 | — |
| **pr-296-lexical** | 35.0% | 37.2% | 37.2% | Cascade, 0.3 | — |
| **laya-english-zeroshot** | 33.3% | 37.3% | 37.3% | Confidence < X → high, 0.8 | — |
| **laya-multilingual-finetuned** | 33.1% | 30.0% | 34.0% | Cascade, 0.9 | — |
| **laya-typed-decisions-zeroshot** | 32.9% | 37.5% | 37.5% | Confidence < X → high, 0.6 | — |
| **laya-multilingual-zeroshot** | 31.4% | 31.8% | 33.4% | Confidence < X → high, 0.95 | — |

**Rules:**
- **Confidence < X → high:** keep the model's pick unless its confidence is below X; then send the question to high.
- **Cascade, cut-off C:** low if P(low) ≥ C, else mid if P(low) + P(mid) ≥ C, else high: the cheapest tier whose chance of being enough reaches C.

> [!WARNING]
> Every row is scored on the same 600 questions, so the **best** row is chosen with the test labels and is optimistic. Two honest checks: **cross-validated** (5 stratified folds; for each fold the rule and threshold, or the plain answer, are chosen on the other 4 folds and applied to this one) and **300-set pick** (the threshold that was best on the authored 300 set).

### d3-flash-zeroshot

*Confidence: d3-flash's own confidence (not its top probability).*

| Rule | Threshold | **Balanced acc.** | Accuracy | Recall low / mid / high | Too small | Too big | Picks low / mid / high |
|---|---|---:|---:|---|---:|---:|---|
| As answered | — | **48.3%** | 50.5% | 45.7 / 85 / 14.3% | 15.2% | 34.3% | 198/386/16 |
| Confidence < X → high | 0.3 | **48.3%** | 36.3% | 28.7 / 22.1 / 94% | 1.8% | 61.8% | 114/87/399 |
| Confidence < X → high | 0.4 | 41.2% | 27.3% | 19.9 / 3.6 / 100% | 0.5% | 72.2% | 78/18/504 |
| Confidence < X → high | 0.5 (300-set pick) | 38.2% | 23.2% | 14.6 / 0 / 100% | 0.2% | 76.7% | 56/0/544 |
| Confidence < X → high | 0.6 | 36.4% | 19.8% | 9.3 / 0 / 100% | 0.2% | 80.0% | 36/0/564 |
| Confidence < X → high | 0.7 | 34.8% | 16.8% | 4.5 / 0 / 100% | 0.0% | 83.2% | 17/0/583 |
| Confidence < X → high | 0.8 | 33.8% | 14.8% | 1.3 / 0 / 100% | 0.0% | 85.2% | 5/0/595 |
| Confidence < X → high | 0.9 | 33.3% | 14.0% | 0 / 0 / 100% | 0.0% | 86.0% | 0/0/600 |
| Confidence < X → high | 0.95 | 33.3% | 14.0% | 0 / 0 / 100% | 0.0% | 86.0% | 0/0/600 |
| Cascade | 0.3 | 45.0% | 56.2% | 63 / 70.7 / 1.2% | 20.7% | 23.2% | 296/303/1 |
| Cascade | 0.4 | 47.7% | 54.2% | 54.3 / 82.9 / 6% | 17.2% | 28.7% | 237/358/5 |
| Cascade | 0.5 | 47.7% | 49.0% | 42.8 / 87.1 / 13.1% | 14.8% | 36.2% | 183/403/14 |
| Cascade | 0.6 | 46.6% | 42.8% | 31.1 / 87.1 / 21.4% | 12.3% | 44.8% | 128/438/34 |
| Cascade | 0.7 | 47.4% | 37.7% | 22.3 / 73.6 / 46.4% | 8.3% | 54.0% | 89/423/88 |
| Cascade | 0.8 | 44.8% | 31.3% | 15.4 / 53.6 / 65.5% | 5.0% | 63.7% | 59/373/168 |
| Cascade | 0.9 | 38.0% | 20.7% | 5.6 / 21.4 / 86.9% | 1.8% | 77.5% | 21/253/326 |

**Cross-validated (honest):** 45.0% balanced, 40.0% accuracy, recall 32.7 / 55.7 / 46.4%. Chosen per fold: As answered, Cascade 0.7, Confidence < X → high 0.3, Confidence < X → high 0.3, As answered.

### jev-direct

*Confidence: Jev's own confidence.*

| Rule | Threshold | **Balanced acc.** | Accuracy | Recall low / mid / high | Too small | Too big | Picks low / mid / high |
|---|---|---:|---:|---|---:|---:|---|
| As answered | — | 45.6% | 59.5% | 70.5 / 65 / 1.2% | 22.0% | 18.5% | 336/263/1 |
| Confidence < X → high | 0.3 | 45.4% | 58.8% | 69.7 / 62.9 / 3.6% | 21.3% | 19.8% | 330/252/18 |
| Confidence < X → high | 0.4 | 46.7% | 57.8% | 67.6 / 57.1 / 15.5% | 19.2% | 23.0% | 315/226/59 |
| Confidence < X → high | 0.5 | 49.7% | 57.7% | 65.2 / 54.3 / 29.8% | 16.7% | 25.7% | 301/204/95 |
| Confidence < X → high | 0.6 | 50.6% | 56.2% | 61.7 / 52.1 / 38.1% | 14.7% | 29.2% | 283/181/136 |
| Confidence < X → high | 0.7 | **51.1%** | 54.8% | 59.8 / 45.7 / 47.6% | 13.0% | 32.2% | 272/150/178 |
| Confidence < X → high | 0.8 | 50.1% | 51.7% | 56.6 / 32.9 / 60.7% | 9.2% | 39.2% | 245/110/245 |
| Confidence < X → high | 0.9 | 46.8% | 45.8% | 50.5 / 17.1 / 72.6% | 6.2% | 48.0% | 212/61/327 |
| Confidence < X → high | 0.95 (300-set pick) | 46.5% | 42.0% | 44.4 / 9.3 / 85.7% | 3.7% | 54.3% | 184/31/385 |
| Cascade | 0.3 | 44.6% | 62.0% | 78.5 / 54.3 / 1.2% | 24.5% | 13.5% | 392/207/1 |
| Cascade | 0.4 | 44.4% | 60.5% | 75 / 57.1 / 1.2% | 23.8% | 15.7% | 369/230/1 |
| Cascade | 0.5 | 45.3% | 59.3% | 70.5 / 64.3 / 1.2% | 22.2% | 18.5% | 337/262/1 |
| Cascade | 0.6 | 46.6% | 58.8% | 67.6 / 68.6 / 3.6% | 20.8% | 20.3% | 315/281/4 |
| Cascade | 0.7 | 47.8% | 57.7% | 63.6 / 71.4 / 8.3% | 19.3% | 23.0% | 293/298/9 |
| Cascade | 0.8 | 48.0% | 56.2% | 59.8 / 73.6 / 10.7% | 18.2% | 25.7% | 272/313/15 |
| Cascade | 0.9 | 50.4% | 55.2% | 54.5 / 80 / 16.7% | 14.7% | 30.2% | 233/339/28 |

**Cross-validated (honest):** 48.6% balanced, 54.0% accuracy, recall 58.8 / 53.6 / 33.3%. Chosen per fold: Confidence < X → high 0.7, Confidence < X → high 0.7, Cascade 0.9, Confidence < X → high 0.6, Confidence < X → high 0.7.

### d3-flash-finetuned

*Confidence: top probability.*

| Rule | Threshold | **Balanced acc.** | Accuracy | Recall low / mid / high | Too small | Too big | Picks low / mid / high |
|---|---|---:|---:|---|---:|---:|---|
| As answered | — | 43.6% | 35.8% | 29 / 36.4 / 65.5% | 6.5% | 57.7% | 127/251/222 |
| Confidence < X → high | 0.3 | 43.6% | 35.8% | 29 / 36.4 / 65.5% | 6.5% | 57.7% | 127/251/222 |
| Confidence < X → high | 0.4 | 44.4% | 36.2% | 29 / 36.4 / 67.9% | 6.2% | 57.7% | 126/250/224 |
| Confidence < X → high | 0.5 | 44.5% | 34.5% | 26.3 / 32.1 / 75% | 5.0% | 60.5% | 113/216/271 |
| Confidence < X → high | 0.6 | 39.1% | 27.5% | 21 / 9.3 / 86.9% | 2.3% | 70.2% | 86/108/406 |
| Confidence < X → high | 0.7 | 37.3% | 23.2% | 15.2 / 1.4 / 95.2% | 1.0% | 75.8% | 62/26/512 |
| Confidence < X → high | 0.8 | 36.0% | 19.5% | 9 / 0 / 98.8% | 0.3% | 80.2% | 36/1/563 |
| Confidence < X → high | 0.9 | 33.7% | 14.7% | 1.1 / 0 / 100% | 0.0% | 85.3% | 4/0/596 |
| Confidence < X → high | 0.95 | 33.3% | 14.0% | 0 / 0 / 100% | 0.0% | 86.0% | 0/0/600 |
| Cascade | 0.3 | **46.1%** | 46.0% | 43.4 / 59.3 / 35.7% | 11.7% | 42.3% | 190/305/105 |
| Cascade | 0.4 | 44.4% | 40.0% | 34.3 / 50 / 48.8% | 9.3% | 50.7% | 149/294/157 |
| Cascade | 0.5 | 41.1% | 33.7% | 26.3 / 38.6 / 58.3% | 7.3% | 59.0% | 113/277/210 |
| Cascade | 0.6 | 43.0% | 31.7% | 21 / 36.4 / 71.4% | 4.5% | 63.8% | 86/266/248 |
| Cascade | 0.7 | 40.5% | 26.7% | 15.2 / 24.3 / 82.1% | 2.8% | 70.5% | 62/239/299 |
| Cascade | 0.8 | 37.7% | 21.8% | 9 / 17.1 / 86.9% | 2.0% | 76.2% | 36/205/359 |
| Cascade | 0.9 | 34.8% | 16.0% | 1.1 / 9.3 / 94% | 0.8% | 83.2% | 4/169/427 |

**Cross-validated (honest):** 44.1% balanced, 43.0% accuracy, recall 39.9 / 54.3 / 38.1%. Chosen per fold: Cascade 0.3, Confidence < X → high 0.4, Cascade 0.3, Cascade 0.3, Cascade 0.3.

### clm-finetuned

*Confidence: top probability.*

| Rule | Threshold | **Balanced acc.** | Accuracy | Recall low / mid / high | Too small | Too big | Picks low / mid / high |
|---|---|---:|---:|---|---:|---:|---|
| As answered | — | 41.6% | 32.5% | 26.1 / 25 / 73.8% | 5.5% | 62.0% | 117/173/310 |
| Confidence < X → high | 0.3 | 41.6% | 32.5% | 26.1 / 25 / 73.8% | 5.5% | 62.0% | 117/173/310 |
| Confidence < X → high | 0.4 | 40.9% | 31.5% | 24.7 / 24.3 / 73.8% | 5.3% | 63.2% | 111/169/320 |
| Confidence < X → high | 0.5 | 38.7% | 27.5% | 19.4 / 19.3 / 77.4% | 4.7% | 67.8% | 90/137/373 |
| Confidence < X → high | 0.6 | 35.6% | 22.2% | 13 / 9.3 / 84.5% | 3.2% | 74.7% | 62/70/468 |
| Confidence < X → high | 0.7 | 34.1% | 18.2% | 7.2 / 3.6 / 91.7% | 1.8% | 80.0% | 36/34/530 |
| Confidence < X → high | 0.8 | 33.9% | 16.5% | 4.5 / 0.7 / 96.4% | 0.8% | 82.7% | 22/7/571 |
| Confidence < X → high | 0.9 | 33.3% | 14.5% | 1.1 / 0 / 98.8% | 0.3% | 85.2% | 6/0/594 |
| Confidence < X → high | 0.95 | 33.6% | 14.5% | 0.8 / 0 / 100% | 0.2% | 85.3% | 4/0/596 |
| Cascade | 0.3 | **42.6%** | 38.5% | 35.1 / 37.9 / 54.8% | 9.7% | 51.8% | 162/230/208 |
| Cascade | 0.4 | 41.3% | 34.3% | 27.7 / 37.9 / 58.3% | 7.5% | 58.2% | 122/236/242 |
| Cascade | 0.5 | 39.8% | 29.2% | 19.4 / 32.1 / 67.9% | 6.0% | 64.8% | 90/234/276 |
| Cascade | 0.6 | 40.3% | 26.0% | 13 / 29.3 / 78.6% | 4.0% | 70.0% | 62/221/317 |
| Cascade | 0.7 | 37.7% | 21.3% | 7.2 / 21.4 / 84.5% | 2.8% | 75.8% | 36/189/375 |
| Cascade | 0.8 | 36.5% | 19.0% | 4.5 / 15.7 / 89.3% | 1.8% | 79.2% | 22/158/420 |
| Cascade | 0.9 | 35.8% | 16.5% | 1.1 / 10 / 96.4% | 0.7% | 82.8% | 6/108/486 |

**Cross-validated (honest):** 41.4% balanced, 36.5% accuracy, recall 32.7 / 34.3 / 57.1%. Chosen per fold: As answered, Cascade 0.3, Cascade 0.3, Cascade 0.3, Cascade 0.3.

### clm-zeroshot

*Confidence: CLM's own confidence.*

| Rule | Threshold | **Balanced acc.** | Accuracy | Recall low / mid / high | Too small | Too big | Picks low / mid / high |
|---|---|---:|---:|---|---:|---:|---|
| As answered | — | 38.9% | 51.3% | 68.1 / 20 / 28.6% | 24.3% | 24.3% | 388/106/106 |
| Confidence < X → high | 0.3 | **39.4%** | 40.8% | 48.4 / 7.9 / 61.9% | 15.8% | 43.3% | 273/39/288 |
| Confidence < X → high | 0.4 | 38.6% | 33.2% | 34 / 4.3 / 77.4% | 11.2% | 55.7% | 194/16/390 |
| Confidence < X → high | 0.5 (300-set pick) | 37.4% | 27.5% | 23.7 / 2.9 / 85.7% | 7.0% | 65.5% | 131/8/461 |
| Confidence < X → high | 0.6 | 35.4% | 21.3% | 13.3 / 0 / 92.9% | 3.8% | 74.8% | 73/3/524 |
| Confidence < X → high | 0.7 | 33.4% | 17.0% | 6.1 / 0 / 94% | 2.5% | 80.5% | 38/1/561 |
| Confidence < X → high | 0.8 | 34.0% | 15.2% | 1.9 / 0 / 100% | 0.7% | 84.2% | 11/0/589 |
| Confidence < X → high | 0.9 | 33.3% | 14.0% | 0 / 0 / 100% | 0.0% | 86.0% | 0/0/600 |
| Confidence < X → high | 0.95 | 33.3% | 14.0% | 0 / 0 / 100% | 0.0% | 86.0% | 0/0/600 |
| Cascade | 0.3 | 36.8% | 58.7% | 84.6 / 22.1 / 3.6% | 31.3% | 10.0% | 484/104/12 |
| Cascade | 0.4 | 38.8% | 54.0% | 71.3 / 32.1 / 13.1% | 26.8% | 19.2% | 405/160/35 |
| Cascade | 0.5 | 36.4% | 45.5% | 54.8 / 37.9 / 16.7% | 23.7% | 30.8% | 312/228/60 |
| Cascade | 0.6 | 36.9% | 36.7% | 34 / 49.3 / 27.4% | 18.2% | 45.2% | 194/304/102 |
| Cascade | 0.7 | 37.7% | 29.5% | 18.1 / 52.1 / 42.9% | 12.2% | 58.3% | 100/328/172 |
| Cascade | 0.8 | 37.0% | 22.7% | 6.1 / 44.3 / 60.7% | 7.2% | 70.2% | 38/282/280 |
| Cascade | 0.9 | 36.6% | 18.2% | 0.3 / 28.6 / 81% | 3.2% | 78.7% | 4/170/426 |

**Cross-validated (honest):** 36.9% balanced, 39.3% accuracy, recall 47.3 / 8.6 / 54.8%. Chosen per fold: Confidence < X → high 0.4, Confidence < X → high 0.3, As answered, Confidence < X → high 0.3, Confidence < X → high 0.3.

### laya-typed-decisions-finetuned

*Confidence: top probability.*

| Rule | Threshold | **Balanced acc.** | Accuracy | Recall low / mid / high | Too small | Too big | Picks low / mid / high |
|---|---|---:|---:|---|---:|---:|---|
| As answered | — | 38.6% | 29.5% | 15.4 / 62.1 / 38.1% | 9.5% | 61.0% | 73/395/132 |
| Confidence < X → high | 0.3 | 38.6% | 29.5% | 15.4 / 62.1 / 38.1% | 9.5% | 61.0% | 73/395/132 |
| Confidence < X → high | 0.4 | 38.2% | 29.2% | 15.2 / 61.4 / 38.1% | 9.5% | 61.3% | 72/394/134 |
| Confidence < X → high | 0.5 | 38.9% | 28.7% | 14.1 / 58.6 / 44% | 8.7% | 62.7% | 67/371/162 |
| Confidence < X → high | 0.6 | **42.0%** | 27.3% | 11.2 / 45.7 / 69% | 5.2% | 67.5% | 53/294/253 |
| Confidence < X → high | 0.7 | 39.3% | 23.8% | 9.8 / 27.1 / 81% | 3.5% | 72.7% | 45/202/353 |
| Confidence < X → high | 0.8 | 37.2% | 20.2% | 7.2 / 11.4 / 92.9% | 1.7% | 78.2% | 34/91/475 |
| Confidence < X → high | 0.9 | 34.3% | 16.7% | 4.3 / 2.1 / 96.4% | 1.0% | 82.3% | 22/11/567 |
| Confidence < X → high | 0.95 | 33.2% | 15.5% | 3.2 / 0 / 96.4% | 0.7% | 83.8% | 16/1/583 |
| Cascade | 0.3 | 37.6% | 33.5% | 20.7 / 81.4 / 10.7% | 14.2% | 52.3% | 103/458/39 |
| Cascade | 0.4 | 38.1% | 31.3% | 17 / 75.7 / 21.4% | 11.8% | 56.8% | 80/444/76 |
| Cascade | 0.5 | 38.5% | 29.2% | 14.1 / 65.7 / 35.7% | 9.8% | 61.0% | 67/409/124 |
| Cascade | 0.6 | 38.2% | 26.7% | 11.2 / 55.7 / 47.6% | 8.2% | 65.2% | 53/378/169 |
| Cascade | 0.7 | 34.6% | 22.7% | 9.8 / 35.7 / 58.3% | 6.7% | 70.7% | 45/313/242 |
| Cascade | 0.8 | 33.7% | 20.0% | 7.2 / 25 / 69% | 5.0% | 75.0% | 34/247/319 |
| Cascade | 0.9 | 33.2% | 17.0% | 4.3 / 10.7 / 84.5% | 2.7% | 80.3% | 22/129/449 |

**Cross-validated (honest):** 42.0% balanced, 27.3% accuracy, recall 11.2 / 45.7 / 69%. Chosen per fold: Confidence < X → high 0.6, Confidence < X → high 0.6, Confidence < X → high 0.6, Confidence < X → high 0.6, Confidence < X → high 0.6.

### pr-292

*Confidence: top probability.*

| Rule | Threshold | **Balanced acc.** | Accuracy | Recall low / mid / high | Too small | Too big | Picks low / mid / high |
|---|---|---:|---:|---|---:|---:|---|
| As answered | — | 37.9% | 29.0% | 20.7 / 32.1 / 60.7% | 8.3% | 62.7% | 107/230/263 |
| Confidence < X → high | 0.3 | 37.9% | 29.0% | 20.7 / 32.1 / 60.7% | 8.3% | 62.7% | 107/230/263 |
| Confidence < X → high | 0.4 | 37.9% | 29.0% | 20.7 / 32.1 / 60.7% | 8.3% | 62.7% | 107/229/264 |
| Confidence < X → high | 0.5 | 38.4% | 29.2% | 20.7 / 31.4 / 63.1% | 8.0% | 62.8% | 107/224/269 |
| Confidence < X → high | 0.6 | 37.6% | 27.0% | 17.8 / 27.1 / 67.9% | 7.3% | 65.7% | 95/202/303 |
| Confidence < X → high | 0.7 | 37.6% | 25.5% | 15.7 / 22.1 / 75% | 6.3% | 68.2% | 86/168/346 |
| Confidence < X → high | 0.8 | 36.9% | 23.8% | 13.8 / 17.1 / 79.8% | 5.2% | 71.0% | 73/141/386 |
| Confidence < X → high | 0.9 | 35.2% | 20.2% | 9.3 / 9.3 / 86.9% | 3.0% | 76.8% | 48/95/457 |
| Confidence < X → high | 0.95 | 34.7% | 18.3% | 6.1 / 8.6 / 89.3% | 2.3% | 79.3% | 32/69/499 |
| Cascade | 0.3 | **38.9%** | 32.7% | 26.6 / 36.4 / 53.6% | 10.0% | 57.3% | 136/240/224 |
| Cascade | 0.4 | 38.8% | 30.5% | 22.3 / 35.7 / 58.3% | 8.8% | 60.7% | 115/243/242 |
| Cascade | 0.5 | 37.9% | 29.0% | 20.7 / 32.1 / 60.7% | 8.3% | 62.7% | 107/231/262 |
| Cascade | 0.6 | 37.1% | 27.0% | 17.8 / 29.3 / 64.3% | 7.8% | 65.2% | 95/232/273 |
| Cascade | 0.7 | 36.8% | 25.5% | 15.7 / 25.7 / 69% | 7.2% | 67.3% | 86/219/295 |
| Cascade | 0.8 | 36.1% | 24.2% | 13.8 / 24.3 / 70.2% | 6.5% | 69.3% | 73/213/314 |
| Cascade | 0.9 | 36.0% | 21.7% | 9.3 / 21.4 / 77.4% | 4.3% | 74.0% | 48/197/355 |

**Cross-validated (honest):** 37.1% balanced, 30.8% accuracy, recall 25 / 32.9 / 53.6%. Chosen per fold: Cascade 0.4, Confidence < X → high 0.5, Cascade 0.3, Cascade 0.3, Cascade 0.3.

### laya-english-finetuned

*Confidence: top probability.*

| Rule | Threshold | **Balanced acc.** | Accuracy | Recall low / mid / high | Too small | Too big | Picks low / mid / high |
|---|---|---:|---:|---|---:|---:|---|
| As answered | — | 37.7% | 28.7% | 19.9 / 33.6 / 59.5% | 7.8% | 63.5% | 101/262/237 |
| Confidence < X → high | 0.3 | 37.7% | 28.7% | 19.9 / 33.6 / 59.5% | 7.8% | 63.5% | 101/262/237 |
| Confidence < X → high | 0.4 | 37.7% | 28.3% | 19.4 / 32.9 / 60.7% | 7.7% | 64.0% | 99/259/242 |
| Confidence < X → high | 0.5 | 38.1% | 27.2% | 16.8 / 32.1 / 65.5% | 6.7% | 66.2% | 85/235/280 |
| Confidence < X → high | 0.6 | 35.9% | 22.5% | 11.7 / 18.6 / 77.4% | 4.7% | 72.8% | 62/136/402 |
| Confidence < X → high | 0.7 | 35.1% | 18.8% | 7.2 / 6.4 / 91.7% | 2.3% | 78.8% | 39/48/513 |
| Confidence < X → high | 0.8 | 33.0% | 14.8% | 1.6 / 2.1 / 95.2% | 0.7% | 84.5% | 10/14/576 |
| Confidence < X → high | 0.9 | 33.3% | 14.0% | 0 / 0 / 100% | 0.0% | 86.0% | 0/1/599 |
| Confidence < X → high | 0.95 | 33.3% | 14.0% | 0 / 0 / 100% | 0.0% | 86.0% | 0/0/600 |
| Cascade | 0.3 | 38.8% | 34.8% | 26.6 / 60 / 29.8% | 12.7% | 52.5% | 136/349/115 |
| Cascade | 0.4 | **38.9%** | 31.5% | 21.3 / 51.4 / 44% | 10.0% | 58.5% | 107/327/166 |
| Cascade | 0.5 | 35.8% | 26.7% | 16.8 / 37.1 / 53.6% | 8.3% | 65.0% | 85/291/224 |
| Cascade | 0.6 | 34.7% | 23.0% | 11.7 / 29.3 / 63.1% | 6.7% | 70.3% | 62/256/282 |
| Cascade | 0.7 | 33.1% | 19.0% | 7.2 / 17.1 / 75% | 4.7% | 76.3% | 39/198/363 |
| Cascade | 0.8 | 33.2% | 16.2% | 1.6 / 15.7 / 82.1% | 2.5% | 81.3% | 10/161/429 |
| Cascade | 0.9 | 32.1% | 14.3% | 0 / 9.3 / 86.9% | 1.8% | 83.8% | 0/96/504 |

**Cross-validated (honest):** 37.0% balanced, 31.3% accuracy, recall 22.3 / 52.9 / 35.7%. Chosen per fold: Cascade 0.4, Cascade 0.3, Cascade 0.3, Cascade 0.4, Cascade 0.4.

### pr-296-semantic

*Confidence: top probability.*

| Rule | Threshold | **Balanced acc.** | Accuracy | Recall low / mid / high | Too small | Too big | Picks low / mid / high |
|---|---|---:|---:|---|---:|---:|---|
| As answered | — | 37.0% | 28.5% | 19.9 / 35 / 56% | 9.2% | 62.3% | 101/235/264 |
| Confidence < X → high | 0.3 | 37.0% | 28.5% | 19.9 / 35 / 56% | 9.2% | 62.3% | 101/235/264 |
| Confidence < X → high | 0.4 | 37.0% | 28.3% | 19.9 / 32.9 / 58.3% | 8.5% | 63.2% | 99/221/280 |
| Confidence < X → high | 0.5 | 36.3% | 25.3% | 15.7 / 26.4 / 66.7% | 6.7% | 68.0% | 77/161/362 |
| Confidence < X → high | 0.6 | 34.3% | 20.8% | 10.6 / 13.6 / 78.6% | 3.8% | 75.3% | 51/89/460 |
| Confidence < X → high | 0.7 | 32.3% | 16.8% | 5.6 / 5.7 / 85.7% | 2.7% | 80.5% | 31/42/527 |
| Confidence < X → high | 0.8 | 33.0% | 15.2% | 2.4 / 1.4 / 95.2% | 0.8% | 84.0% | 11/11/578 |
| Confidence < X → high | 0.9 | 33.4% | 14.5% | 0.8 / 0.7 / 98.8% | 0.3% | 85.2% | 5/3/592 |
| Confidence < X → high | 0.95 | 33.4% | 14.2% | 0.3 / 0 / 100% | 0.0% | 85.8% | 1/1/598 |
| Cascade | 0.3 | 36.7% | 34.7% | 29 / 55 / 26.2% | 14.8% | 50.5% | 148/341/111 |
| Cascade | 0.4 | **38.2%** | 31.7% | 22.1 / 52.1 / 40.5% | 11.3% | 57.0% | 109/318/173 |
| Cascade | 0.5 | 37.5% | 27.5% | 15.7 / 44.3 / 52.4% | 8.7% | 63.8% | 77/292/231 |
| Cascade | 0.6 | 36.6% | 24.3% | 10.6 / 40.7 / 58.3% | 6.7% | 69.0% | 51/263/286 |
| Cascade | 0.7 | 35.1% | 20.3% | 5.6 / 30.7 / 69% | 5.0% | 74.7% | 31/218/351 |
| Cascade | 0.8 | 36.4% | 18.5% | 2.4 / 22.1 / 84.5% | 2.3% | 79.2% | 11/164/425 |
| Cascade | 0.9 | 34.6% | 15.8% | 0.8 / 10 / 92.9% | 1.2% | 83.0% | 5/85/510 |

**Cross-validated (honest):** 36.4% balanced, 30.2% accuracy, recall 21.5 / 47.1 / 40.5%. Chosen per fold: Cascade 0.4, Confidence < X → high 0.4, Cascade 0.4, Cascade 0.4, Cascade 0.4.

### pr-296-lexical

*Confidence: top probability.*

| Rule | Threshold | **Balanced acc.** | Accuracy | Recall low / mid / high | Too small | Too big | Picks low / mid / high |
|---|---|---:|---:|---|---:|---:|---|
| As answered | — | 35.0% | 25.3% | 16.5 / 27.9 / 60.7% | 7.7% | 67.0% | 81/202/317 |
| Confidence < X → high | 0.3 | 35.0% | 25.3% | 16.5 / 27.9 / 60.7% | 7.7% | 67.0% | 81/202/317 |
| Confidence < X → high | 0.4 | 34.7% | 25.0% | 16.2 / 27.1 / 60.7% | 7.7% | 67.3% | 80/199/321 |
| Confidence < X → high | 0.5 | 33.4% | 21.7% | 10.9 / 25 / 64.3% | 6.2% | 72.2% | 52/175/373 |
| Confidence < X → high | 0.6 | 32.7% | 18.8% | 7.2 / 17.1 / 73.8% | 4.7% | 76.5% | 36/124/440 |
| Confidence < X → high | 0.7 | 32.5% | 16.3% | 3.2 / 12.1 / 82.1% | 3.3% | 80.3% | 18/86/496 |
| Confidence < X → high | 0.8 | 32.3% | 15.0% | 2.1 / 4.3 / 90.5% | 2.0% | 83.0% | 13/42/545 |
| Confidence < X → high | 0.9 | 33.6% | 14.5% | 0.5 / 1.4 / 98.8% | 0.5% | 85.0% | 4/11/585 |
| Confidence < X → high | 0.95 | 33.6% | 14.2% | 0 / 0.7 / 100% | 0.2% | 85.7% | 1/2/597 |
| Cascade | 0.3 | **37.2%** | 31.5% | 23.1 / 49.3 / 39.3% | 11.5% | 57.0% | 115/303/182 |
| Cascade | 0.4 | 35.7% | 27.0% | 17 / 40 / 50% | 9.2% | 63.8% | 84/273/243 |
| Cascade | 0.5 | 34.8% | 23.3% | 10.9 / 36.4 / 57.1% | 7.2% | 69.5% | 52/258/290 |
| Cascade | 0.6 | 32.9% | 19.8% | 7.2 / 27.1 / 64.3% | 6.0% | 74.2% | 36/217/347 |
| Cascade | 0.7 | 30.7% | 16.7% | 3.2 / 23.6 / 65.5% | 5.7% | 77.7% | 18/189/393 |
| Cascade | 0.8 | 31.5% | 15.7% | 2.1 / 15 / 77.4% | 3.8% | 80.5% | 13/128/459 |
| Cascade | 0.9 | 33.9% | 15.5% | 0.5 / 10.7 / 90.5% | 1.7% | 82.8% | 4/74/522 |

**Cross-validated (honest):** 37.2% balanced, 31.5% accuracy, recall 23.1 / 49.3 / 39.3%. Chosen per fold: Cascade 0.3, Cascade 0.3, Cascade 0.3, Cascade 0.3, Cascade 0.3.

### laya-english-zeroshot

*Confidence: top probability.*

| Rule | Threshold | **Balanced acc.** | Accuracy | Recall low / mid / high | Too small | Too big | Picks low / mid / high |
|---|---|---:|---:|---|---:|---:|---|
| As answered | — | 33.3% | 23.3% | 0 / 100 / 0% | 14.0% | 62.7% | 0/600/0 |
| Confidence < X → high | 0.3 | 33.3% | 23.3% | 0 / 100 / 0% | 14.0% | 62.7% | 0/600/0 |
| Confidence < X → high | 0.4 | 33.3% | 23.3% | 0 / 100 / 0% | 14.0% | 62.7% | 0/600/0 |
| Confidence < X → high | 0.5 | 33.5% | 23.3% | 0 / 99.3 / 1.2% | 13.8% | 62.8% | 0/598/2 |
| Confidence < X → high | 0.6 | 33.7% | 23.2% | 0 / 96.4 / 4.8% | 13.3% | 63.5% | 0/587/13 |
| Confidence < X → high | 0.7 | 34.4% | 22.2% | 0 / 82.9 / 20.2% | 11.2% | 66.7% | 0/534/66 |
| Confidence < X → high | 0.8 | **37.3%** | 18.7% | 0 / 32.1 / 79.8% | 2.8% | 78.5% | 0/179/421 |
| Confidence < X → high | 0.9 | 33.3% | 14.0% | 0 / 0 / 100% | 0.0% | 86.0% | 0/0/600 |
| Confidence < X → high | 0.95 | 33.3% | 14.0% | 0 / 0 / 100% | 0.0% | 86.0% | 0/0/600 |
| Cascade | 0.3 | 33.3% | 23.3% | 0 / 100 / 0% | 14.0% | 62.7% | 2/598/0 |
| Cascade | 0.4 | 33.3% | 23.3% | 0 / 100 / 0% | 14.0% | 62.7% | 1/599/0 |
| Cascade | 0.5 | 33.3% | 23.3% | 0 / 100 / 0% | 14.0% | 62.7% | 0/600/0 |
| Cascade | 0.6 | 32.9% | 23.0% | 0 / 98.6 / 0% | 14.0% | 63.0% | 0/598/2 |
| Cascade | 0.7 | 32.8% | 22.8% | 0 / 97.1 / 1.2% | 13.8% | 63.3% | 0/591/9 |
| Cascade | 0.8 | 31.5% | 20.8% | 0 / 81.4 / 13.1% | 12.2% | 67.0% | 0/546/54 |
| Cascade | 0.9 | 33.1% | 14.8% | 0 / 10 / 89.3% | 1.5% | 83.7% | 0/95/505 |

**Cross-validated (honest):** 37.3% balanced, 18.7% accuracy, recall 0 / 32.1 / 79.8%. Chosen per fold: Confidence < X → high 0.8, Confidence < X → high 0.8, Confidence < X → high 0.8, Confidence < X → high 0.8, Confidence < X → high 0.8.

### laya-multilingual-finetuned

*Confidence: top probability.*

| Rule | Threshold | **Balanced acc.** | Accuracy | Recall low / mid / high | Too small | Too big | Picks low / mid / high |
|---|---|---:|---:|---|---:|---:|---|
| As answered | — | 33.1% | 25.3% | 15.7 / 40.7 / 42.9% | 11.8% | 62.8% | 95/280/225 |
| Confidence < X → high | 0.3 | 33.1% | 25.3% | 15.7 / 40.7 / 42.9% | 11.8% | 62.8% | 95/280/225 |
| Confidence < X → high | 0.4 | 31.9% | 22.5% | 12.5 / 32.1 / 51.2% | 9.7% | 67.8% | 75/244/281 |
| Confidence < X → high | 0.5 | 32.2% | 17.5% | 7.2 / 5 / 84.5% | 4.2% | 78.3% | 47/66/487 |
| Confidence < X → high | 0.6 | 32.4% | 15.7% | 4.3 / 0 / 92.9% | 2.7% | 81.7% | 31/13/556 |
| Confidence < X → high | 0.7 | 33.4% | 15.3% | 2.7 / 0 / 97.6% | 1.5% | 83.2% | 19/5/576 |
| Confidence < X → high | 0.8 | 33.3% | 14.5% | 1.1 / 0 / 98.8% | 0.5% | 85.0% | 7/4/589 |
| Confidence < X → high | 0.9 | 33.3% | 14.0% | 0 / 0 / 100% | 0.0% | 86.0% | 0/1/599 |
| Confidence < X → high | 0.95 | 33.3% | 14.0% | 0 / 0 / 100% | 0.0% | 86.0% | 0/0/600 |
| Cascade | 0.3 | 33.4% | 32.7% | 23.7 / 76.4 / 0% | 19.5% | 47.8% | 142/455/3 |
| Cascade | 0.4 | 32.8% | 27.7% | 12.5 / 83.6 / 2.4% | 16.5% | 55.8% | 75/510/15 |
| Cascade | 0.5 | 32.9% | 24.2% | 7.2 / 73.6 / 17.9% | 13.5% | 62.3% | 47/469/84 |
| Cascade | 0.6 | 31.9% | 19.7% | 4.3 / 45 / 46.4% | 9.2% | 71.2% | 31/329/240 |
| Cascade | 0.7 | 30.0% | 15.2% | 2.7 / 13.6 / 73.8% | 4.8% | 80.0% | 19/138/443 |
| Cascade | 0.8 | 33.2% | 15.0% | 1.1 / 5.7 / 92.9% | 1.3% | 83.7% | 7/42/551 |
| Cascade | 0.9 | **34.0%** | 14.7% | 0 / 4.3 / 97.6% | 0.3% | 85.0% | 0/18/582 |

**Cross-validated (honest):** 30.0% balanced, 20.3% accuracy, recall 10.1 / 30 / 50%. Chosen per fold: Cascade 0.9, Cascade 0.9, As answered, Cascade 0.3, As answered.

### laya-typed-decisions-zeroshot

*Confidence: top probability.*

| Rule | Threshold | **Balanced acc.** | Accuracy | Recall low / mid / high | Too small | Too big | Picks low / mid / high |
|---|---|---:|---:|---|---:|---:|---|
| As answered | — | 32.9% | 23.0% | 0 / 98.6 / 0% | 14.0% | 63.0% | 0/597/3 |
| Confidence < X → high | 0.3 | 32.9% | 23.0% | 0 / 98.6 / 0% | 14.0% | 63.0% | 0/597/3 |
| Confidence < X → high | 0.4 | 32.9% | 23.0% | 0 / 98.6 / 0% | 14.0% | 63.0% | 0/597/3 |
| Confidence < X → high | 0.5 | 32.9% | 22.7% | 0 / 95 / 3.6% | 13.5% | 63.8% | 0/582/18 |
| Confidence < X → high | 0.6 | **37.5%** | 21.7% | 0 / 63.6 / 48.8% | 7.2% | 71.2% | 0/395/205 |
| Confidence < X → high | 0.7 | 34.3% | 14.7% | 0 / 2.9 / 100% | 0.0% | 85.3% | 0/21/579 |
| Confidence < X → high | 0.8 | 33.3% | 14.0% | 0 / 0 / 100% | 0.0% | 86.0% | 0/0/600 |
| Confidence < X → high | 0.9 | 33.3% | 14.0% | 0 / 0 / 100% | 0.0% | 86.0% | 0/0/600 |
| Confidence < X → high | 0.95 | 33.3% | 14.0% | 0 / 0 / 100% | 0.0% | 86.0% | 0/0/600 |
| Cascade | 0.3 | 33.4% | 23.5% | 0.3 / 100 / 0% | 14.0% | 62.5% | 2/598/0 |
| Cascade | 0.4 | 33.3% | 23.3% | 0 / 100 / 0% | 14.0% | 62.7% | 0/600/0 |
| Cascade | 0.5 | 33.1% | 23.2% | 0 / 99.3 / 0% | 14.0% | 62.8% | 0/599/1 |
| Cascade | 0.6 | 32.9% | 22.8% | 0 / 96.4 / 2.4% | 13.7% | 63.5% | 0/591/9 |
| Cascade | 0.7 | 36.0% | 22.5% | 0 / 79.3 / 28.6% | 10.0% | 67.5% | 0/503/97 |
| Cascade | 0.8 | 34.4% | 15.3% | 0 / 9.3 / 94% | 0.8% | 83.8% | 0/90/510 |
| Cascade | 0.9 | 33.3% | 14.0% | 0 / 0 / 100% | 0.0% | 86.0% | 0/0/600 |

**Cross-validated (honest):** 37.5% balanced, 21.7% accuracy, recall 0 / 63.6 / 48.8%. Chosen per fold: Confidence < X → high 0.6, Confidence < X → high 0.6, Confidence < X → high 0.6, Confidence < X → high 0.6, Confidence < X → high 0.6.

### laya-multilingual-zeroshot

*Confidence: top probability.*

| Rule | Threshold | **Balanced acc.** | Accuracy | Recall low / mid / high | Too small | Too big | Picks low / mid / high |
|---|---|---:|---:|---|---:|---:|---|
| As answered | — | 31.4% | 23.0% | 3.5 / 87.1 / 3.6% | 14.5% | 62.5% | 25/536/39 |
| Confidence < X → high | 0.3 | 31.4% | 23.0% | 3.5 / 87.1 / 3.6% | 14.5% | 62.5% | 25/536/39 |
| Confidence < X → high | 0.4 | 31.5% | 22.8% | 3.2 / 86.4 / 4.8% | 14.2% | 63.0% | 22/534/44 |
| Confidence < X → high | 0.5 | 31.1% | 21.8% | 2.4 / 81.4 / 9.5% | 12.8% | 65.3% | 15/502/83 |
| Confidence < X → high | 0.6 | 28.0% | 18.0% | 0.8 / 62.9 / 20.2% | 11.2% | 70.8% | 6/426/168 |
| Confidence < X → high | 0.7 | 29.8% | 16.3% | 0 / 40.7 / 48.8% | 7.2% | 76.5% | 0/313/287 |
| Confidence < X → high | 0.8 | 31.6% | 15.3% | 0 / 22.1 / 72.6% | 3.8% | 80.8% | 0/192/408 |
| Confidence < X → high | 0.9 | 33.0% | 15.0% | 0 / 12.1 / 86.9% | 1.8% | 83.2% | 0/92/508 |
| Confidence < X → high | 0.95 | **33.4%** | 14.5% | 0 / 5 / 95.2% | 0.7% | 84.8% | 0/41/559 |
| Cascade | 0.3 | 31.5% | 25.2% | 8.8 / 82.1 / 3.6% | 17.0% | 57.8% | 68/519/13 |
| Cascade | 0.4 | 33.2% | 24.5% | 4 / 92.1 / 3.6% | 14.3% | 61.2% | 27/551/22 |
| Cascade | 0.5 | 32.9% | 23.7% | 2.4 / 92.9 / 3.6% | 13.7% | 62.7% | 15/556/29 |
| Cascade | 0.6 | 31.9% | 22.3% | 0.8 / 91.4 / 3.6% | 13.5% | 64.2% | 6/552/42 |
| Cascade | 0.7 | 30.2% | 20.8% | 0 / 87.1 / 3.6% | 13.5% | 65.7% | 0/541/59 |
| Cascade | 0.8 | 31.7% | 20.0% | 0 / 71.4 / 23.8% | 10.7% | 69.3% | 0/464/136 |
| Cascade | 0.9 | 28.3% | 15.2% | 0 / 35 / 50% | 7.0% | 77.8% | 0/269/331 |

**Cross-validated (honest):** 31.8% balanced, 18.2% accuracy, recall 2.4 / 39.3 / 53.6%. Chosen per fold: Confidence < X → high 0.9, Cascade 0.4, Confidence < X → high 0.95, Confidence < X → high 0.95, Cascade 0.4.

## ⚙️ Setup per model

| Model | Setup | Notes |
|---|---|---|
| **d3-flash-zeroshot** | d3-flash asked low / mid / high, top option | vllm-sr/d3-flash on the Mac GPU (PyTorch path, no ROCm kernels), Jev's LEVEL_QUESTION; latency includes the 9B backbone. |
| **jev-direct** | Jev (jev-1.13.0) asked low / mid / high, top option | TypeSafe API, LEVEL_QUESTION, 4 parallel requests; latency is the API round trip. Total cost $0.0131. |
| **d3-flash-finetuned** | d3-flash (backbone frozen), readout rows for the 3 tier codes fine-tuned on eval_600 | Readout training warm-starts from the model's own readout; p50 latency is the readout only (add the backbone time of d3-flash-zeroshot). Train 0.6 s. |
| **clm-finetuned** | CLM (Qwen3-8B frozen), projection heads fine-tuned on eval_600 for the 3 tier options | Heads warm-started from CLM_v0.1-8B; latency includes the encoder call. Train 65.5 s. |
| **clm-zeroshot** | CLM asked low / mid / high, top option | CLM_v0.1-8B heads; Qwen3-8B encoder in PyTorch on the Mac GPU (stand-in for vLLM, same last-token pooling); latency includes the encoder. |
| **laya-typed-decisions-finetuned** | Laya typed-decisions fine-tuned on eval_600 (options low / mid / high) | Plain cross-entropy, AdamW, top 8 encoder layers, up to 20 epochs, best epoch by loss on 10% of eval_600 held out. Chosen epoch: 6. Train 841.3 s. |
| **pr-292** | all-MiniLM-L6-v2 fine-tuned (last 2 layers + tier head), 3-class | PR #292's model and recipe (16 epochs, best epoch on a 20% hold-out of eval_600). Train 32.1 s. |
| **laya-english-finetuned** | Laya english fine-tuned on eval_600 (options low / mid / high) | Plain cross-entropy, AdamW, top 8 encoder layers, up to 20 epochs, best epoch by loss on 10% of eval_600 held out. Chosen epoch: 6. Train 839.1 s. |
| **pr-296-semantic** | bge-small-en-v1.5 (fastembed) + shape features + LR (C=4), 3-class | PR #296's semantic model, trained on eval_600. Train 1.8 s. |
| **pr-296-lexical** | word 1-2 grams + char 3-5 grams + shape features + LR (C=20), 3-class | PR #296's lexical model, trained on eval_600. Train 0.2 s. |
| **nasiko-main-online** | regex request type + Thompson-sampled tier, after each pick the chosen cell learns from the measured label (1 if the tier is at or above it): best-case instant feedback | main's own routing::classify (Rust, commit c6ebd4b3 2026-10-07), seed 0 shown. Seeds 0-4: balanced 33.7% (sd 0.8), accuracy 28.6% (sd 3.2). No difficulty signal: the tier depends only on the request type and the sampled priors. |
| **nasiko-main-cold** | regex request type + Thompson-sampled tier, no learned cells, cold-start priors only (a fresh deployment) | main's own routing::classify (Rust, commit c6ebd4b3 2026-10-07), seed 0 shown. Seeds 0-4: balanced 32.7% (sd 1.0), accuracy 41.6% (sd 1.1). No difficulty signal: the tier depends only on the request type and the sampled priors. |
| **laya-english-zeroshot** | Laya english asked low / mid / high, top option | The base model as published, no training. |
| **laya-multilingual-finetuned** | Laya multilingual fine-tuned on eval_600 (options low / mid / high) | Plain cross-entropy, AdamW, top 8 encoder layers, up to 20 epochs, best epoch by loss on 10% of eval_600 held out. Chosen epoch: 2. Train 367.9 s. |
| **laya-typed-decisions-zeroshot** | Laya typed-decisions asked low / mid / high, top option | The base model as published, no training. |
| **laya-multilingual-zeroshot** | Laya multilingual asked low / mid / high, top option | The base model as published, no training. |

## 🧪 How it was tested

- **Test set:** `data/llm_tier_selection_600_measured.jsonl`, 600 questions. Each label is the cheapest tier whose models all answered correctly (LOW: `gpt-4.1-nano` + `Qwen3.5 9B`; MID: `gpt-4.1` + `Gemma 4 26B-A4B`; HIGH: `gpt-5.4`), graded by a blind judge. The 11 questions no tier solved are labelled HIGH.
- **Fine-tuned models** are trained on `eval_600` (authored labels, 200 per tier; not included here) and tested once here. The two sets share no questions.
- **Zero-shot decision models** (Jev, d3-flash, CLM, Laya) are asked Jev's low / mid / high question (`LEVEL_QUESTION`), and their top option is the pick.
- **Balanced accuracy** = mean of the three recalls. **Too small** = picked a cheaper tier than measured (answer at risk); **too big** = picked a dearer tier (money wasted).
- **Latency:** one question at a time on this Mac (M5 Pro), or the API round trip for Jev.
- **Files:** `results/<model>.json` (metrics), `results/predictions/<model>.jsonl` (every pick), `results/logs/` (console log of every run).

---

## 📖 Glossary

### Tiers and labels

| Term | Meaning |
|---|---|
| **Tier** | Which size of model a question is sent to: **LOW** (small, cheap), **MID** (general-purpose), **HIGH** (frontier, most expensive) |
| **Measured label** | The cheapest tier whose models all answered the question correctly, as graded by a blind judge. LOW = `gpt-4.1-nano` + `Qwen3.5 9B` both pass; MID = `gpt-4.1` + `Gemma 4 26B-A4B` both pass; HIGH = `gpt-5.4` passes |
| **Authored label** | A tier a person guessed by reading the question, as in `eval_600` and the 300 set. Not checked against real model answers |
| **NONE** | No tier answered correctly, not even HIGH. These 11 questions are labelled HIGH in this set |
| **`eval_600`** | The 600 questions with authored labels (200 per tier) that every fine-tuned model was trained on. No overlap with the test set |

### Scores

| Term | Meaning |
|---|---|
| **Accuracy** | Share of the 600 questions where the picked tier equals the measured tier. Dominated by LOW, which is 63% of the set |
| **Balanced accuracy** | The average of the three recalls, so every tier counts equally however many questions it has. **33.3% = chance.** The main ranking here |
| **Recall (low / mid / high)** | Of the questions that really are that tier, the share the model picked correctly. Example: recall high 14% = 12 of the 84 HIGH questions caught |
| **Too small** | Share of questions sent to a **cheaper** tier than measured: the answer is at risk of being wrong |
| **Too big** | Share of questions sent to a **dearer** tier than measured: money wasted, the answer is still fine |
| **Picks low / mid / high** | How many of the 600 questions the model sent to each tier. The truth is 376 / 140 / 84 |
| **p50 latency** | Median time to classify one question (half are faster, half slower) |
| **Always LOW (baseline)** | A "router" that sends everything to LOW: 62.7% accuracy but 33.3% balanced. Shows why plain accuracy misleads on this set |

### How models decide

| Term | Meaning |
|---|---|
| **Decision model** | A model built to answer structured questions about a text, such as Jev, d3-flash, CLM and Laya. It is given the question plus the low / mid / high instructions and scores the three options |
| **`LEVEL_QUESTION`** | The exact instructions and tier descriptions every decision model is asked: "Select the lowest model tier that is sufficient…" |
| **Zero-shot** | The model as published, with **no training on our data**; it judges from the instructions alone |
| **Fine-tuned** | The model after training part of it on `eval_600`'s questions and labels, so its picks match those examples |
| **Readout (d3-flash)** | d3-flash's last small layer that turns its internal state into scores for the options. Fine-tuning retrains only this; the 9B model stays frozen |
| **Projection heads (CLM)** | CLM's two small layers that map the question and each option into a shared space; the closest option wins. Fine-tuning retrains only these |
| **Encoder** | The large model that turns text into numbers (an embedding): Qwen3-8B for CLM, MiniLM for PR-292, bge-small for PR-296 |
| **Embedding** | A list of numbers representing a text's meaning; similar texts get similar numbers |
| **Logistic regression (LR)** | A simple classifier that learns a weight per feature and picks the most likely tier (PR-296) |
| **N-grams** | Word pairs or character snippets used as features, e.g. "chance of" or "rai" (PR-296 lexical) |
| **Shape features** | Simple text facts such as length, digits or code symbols (PR-296) |

### Thresholds

| Term | Meaning |
|---|---|
| **Confidence** | How sure the model says it is of its pick. Each model defines it its own way (d3-flash's is not its top probability) |
| **Probabilities** | The model's chance for each tier, adding up to 1, e.g. low 0.91 / mid 0.09 / high 0.00 |
| **Confidence < X → high** | Keep the model's pick unless its confidence is below X; then send the question to HIGH to be safe |
| **Cascade, cut-off C** | Pick LOW if P(low) ≥ C, else MID if P(low) + P(mid) ≥ C, else HIGH: the cheapest tier likely enough |
| **300-set pick** | The threshold that was best on the authored 300 set, reused here unchanged: an honest test, because it wasn't chosen with these labels |
| **Optimistic** | A "best" threshold chosen by looking at the same test labels it is scored on, so it overstates real performance |

### Nasiko router

| Term | Meaning |
|---|---|
| **Nasiko main** | The current router in Nasiko-Labs/nasiko (`llm-router`, commit `c6ebd4b3`), run through its own Rust code |
| **Request type** | Nasiko's regex guess of the kind of work: code generation, factual lookup, writing, … |
| **Thompson sampling** | Picking a tier by a weighted random draw that leans towards tiers believed good and cheap for that request type |
| **Cold start** | The router with nothing learned yet: only its built-in priors (a fresh deployment) |
| **Online** | The router learning after every pick from feedback (here, best-case instant feedback from the measured labels) |
| **Seed** | The starting value of the random number generator; different seeds give different random draws. 5 seeds were run to check the spread |
