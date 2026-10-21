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

- Most questions an author would call MID or HIGH are solved by today's LOW models (see `../tier_measure/RESULTS.md`). A model trained to copy the authored labels sends them up a tier or two.
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
