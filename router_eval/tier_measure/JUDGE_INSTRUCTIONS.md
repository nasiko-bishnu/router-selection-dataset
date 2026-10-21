# Judge instructions

> **TL;DR** — Grade each answer in `batch_NN.jsonl` as **pass** or **fail**. Write one line per item to `../grades/batch_NN.jsonl` with `item_id`, `pass`, `score` and `reason`. Judge the answers only, never their source.

## ⚙️ Rules for the judge session

- **Only these files.** Read only the files in this `judge_pack/` folder.
- **No key, no history.** Do **not** open the `answers/` folder, or anything that says which model wrote an answer. Don't use any earlier conversation about this project.
- **Independent items.** Grade each item on its own. Items are shuffled, and there's no pattern to find.
- **No other help.** Don't run code, search the web, or ask another model. Your own judgement is the measurement.

## 🧪 How to grade

Each line in a batch file is one item:

```json
{"item_id": "a1b2c3d4e5", "question": "...", "answer": "...", "reference_answer": "... (only on some items)"}
```

Judge the answer on its **substance**, not its style or length:

| Situation | Grade |
|---|---|
| Facts, numbers and reasoning are right, and every part of the request is done | **pass** |
| Minor slips that don't change the outcome | **pass** |
| Wrong final result, a required part missing, or a serious factual or logical error | **fail** |
| Open-ended request (design, analysis, writing): a competent expert would accept it as sound and meeting the stated constraints | **pass** |
| Open-ended request that is vague, generic or ignores a stated constraint | **fail** |
| A `reference_answer` is given and the answer's final result disagrees with it | **fail** |
| The answer refuses a reasonable request, or is empty or cut off before the answer | **fail** |

> [!IMPORTANT]
> - **Be strict and consistent:** the same standard for every item, however long or confident the answer looks.
> - **When genuinely unsure,** give `"pass": false` with a `score` between 0.4 and 0.6, and say why in the reason.

## 📁 Output

For every `batch_NN.jsonl`, write `../grades/batch_NN.jsonl` with **one line per item**, in any order:

```json
{"item_id": "a1b2c3d4e5", "pass": true, "score": 0.9, "reason": "Correct total and both requested steps shown."}
```

| Field | Meaning |
|---|---|
| `item_id` | Copied exactly from the batch |
| `pass` | `true` or `false` |
| `score` | 0 to 1: how fully and correctly the request is met |
| `reason` | One short sentence |

Every item in a batch needs a grade, so a batch's grade file has as many lines as the batch itself.
