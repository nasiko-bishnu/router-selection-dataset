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
| **`LEVEL_QUESTION`** | The exact instructions and tier descriptions every decision model is asked, from `jev/run_jev.py`: "Select the lowest model tier that is sufficient…" |
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
