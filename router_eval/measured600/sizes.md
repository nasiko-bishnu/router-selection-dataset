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
