"""Make a length-balanced copy of llm_tier_selection_150_combined.jsonl.

The original gives its tier away through length: without the appended answer-style suffixes, LOW
questions are 5-16 words (median 12) and HIGH 18-27 (median 21), so length alone picks the right tier
75% of the time. This copy:
  - removes the two boilerplate suffixes from every question (they were a giveaway of their own),
  - rewrites 25 LOW questions to be wordier (added everyday context, same easy task),
  - rewrites 25 HIGH questions to be terser (same hard core, fewer words),
  - rewords the other 25 HIGH questions in plain language (same hard core), so long technical words
    no longer give the tier away through character count,
  - restyles 8 MID questions (voice and openings only, same task),
  - replaces 22 questions with new ones of another type in the same tier (hard math in HIGH, easy
    design and planning in LOW, expert facts and light design in MID), so question type stops
    giving the tier away,
  - a final pass spreads words and question marks that leaned by tier (HIGH "data / model / test",
    MID "explain / compare") across all tiers, restores #22's four causes, swaps the well-known Pell
    equation for x^2 - 13y^2 = 1, and adds reference answers (numeric ones verified by computation),
  - leaves every other question unchanged.
Rewrites spread first-person framing, semicolons and openings ("What...", "How...", "Explain...")
across all three tiers so none of them points to one tier, and avoid cue words such as "simple",
"basic", "short" or "just". Rewritten rows carry "edit": "longer" / "shorter" and "original_query".
The rewritten labels are a judgement call and should be re-checked against real model results.

    used by make_300_balanced.py; run directly it only prints a summary
"""
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "llm_tier_selection_150_combined.jsonl"
SRC_COMMIT = "b9fe5c3"  # the original file lives in git history from this commit on


def read_original():
    """The original 150 rows, from disk if present, else from git history."""
    text = SRC.read_text() if SRC.exists() else subprocess.run(
        ["git", "-C", str(HERE), "show", f"{SRC_COMMIT}:{SRC.name}"], capture_output=True, text=True, check=True).stdout
    return [json.loads(l) for l in text.splitlines() if l.strip()]
SUFFIXES = ["Keep the answer concise and explain your reasoning briefly.",
            "Answer directly, show necessary steps when relevant, and avoid unrelated details."]

# Row index in the source file -> rewritten question.
LONGER = {  # LOW: same easy task, more words
    5: "A student scored 4, 7, 9, and 12 on four spelling quizzes this month. What is the mean score, with the arithmetic shown?",
    6: "For a science worksheet for young children, please name the three states of matter that are commonly taught in school.",
    9: "For a class presentation next week, explain the difference between renewable and non-renewable energy, giving one everyday example of each.",
    10: "A primary school teacher is preparing flashcards and needs a one-sentence definition of an ecosystem that a child could understand.",
    12: "Someone who recently opened their first bank account is confused. Explain the difference between a debit card and a credit card.",
    26: "A coworker sent over a file ending in .csv, and the recipient is unsure what it is. What does that extension usually indicate?",
    36: "My grandmother asked me this while setting up her new laptop: what is the purpose of an operating system on a computer?",
    47: "In two sentences that a worried parent could follow easily, summarize what a vaccine is for and how it helps.",
    58: "Prices at my local grocery store seem higher every year. In everyday economics, what does the term 'inflation' mean?",
    60: "Planning a safari holiday with my family next year, I wondered: what is Kenya's capital, and on which continent is Kenya located?",
    61: "While visiting a museum gift shop I saw a famous portrait print and wondered: who painted the Mona Lisa?",
    64: "A school garden keeps losing topsoil after heavy rain each spring. What are two common causes of soil erosion in general?",
    66: "Before a trip to Paris next week, how would you translate 'The train leaves at six o'clock' into French?",
    76: "For a biology homework sheet about blood, what is the function of red blood cells in the human body?",
    83: "Ahead of a work talk next Tuesday, how can a presentation slide be made easier for the audience to read? Give two ways.",
    88: "I am learning JavaScript from an online course. Write a function that checks whether a given number is even and returns true or false.",
    89: "Looking through a backyard telescope with my kids last night, we wondered which planet is known for its prominent ring system.",
    92: "How would you cook a vegetable stir-fry from a typical fridge? Give three ingredients commonly used in one.",
    99: "My daughter's school is buying lab equipment and she asked me: what is the main purpose of a microscope in science?",
    104: "A restaurant bill comes to 250 rupees and the diner wants to leave an 18% tip. Calculate 18% of 250 and show the arithmetic.",
    113: "A movie marathon lasts 2.4 hours in total. Convert that duration into minutes and show the calculation you used.",
    117: "Last month my laptop died and I lost some photos. Explain why regular backups are useful for personal files like these.",
    126: "Geography students often mix these up, so explain in plain terms the difference between weather and climate.",
    132: "A running app reports a route of 3.5 kilometers. How many meters is that? Show the multiplication you used.",
    145: "Five friends each told me how many books they read this year: 3, 8, 10, 14, and 21. What is the median?",
}
SHORTER = {  # HIGH: same hard core, fewer words
    0: "Retrieved passages each look right but disagree. When should the system trust one, cite both, or refuse?",
    1: "How would you tell population shift from intervention-induced feedback when a clinical model degrades?",
    8: "Design multi-region storage giving low-latency reads, strict financial correctness, and survival of regional outages.",
    11: "Estimate a school program's causal effect without randomization, despite baseline differences and students switching schools.",
    13: "Make long-running AI jobs fault-tolerant despite rate limits, partial failures, and non-idempotent external side effects.",
    22: "What would distinguish covariate shift from concept drift when a model fails in a new region?",
    27: "Prices changed alongside customer mix and marketing spend. How would you credibly estimate price elasticity?",
    28: "Fuse satellite, sensor, and simulation data at different resolutions while propagating uncertainty honestly.",
    33: "How should you split data to test whether a graph neural network generalizes to unseen organizations?",
    39: "Which differential-privacy parameters suit analytics on a small dataset, and how do you avoid misleading guarantees?",
    40: "Split a monolith into services when transactions cross modules and feature work cannot pause.",
    41: "Our p99 latency rises after every release while average latency and CPU stay flat; why?",
    43: "Analyze conflicting clauses in a cross-border software contract: governing law, jurisdiction, liability, and data duties.",
    56: "How would you evaluate a coding agent whose passing solutions may hide security flaws or regressions?",
    72: "Our clicks rose but long-term retention fell. Design an experiment separating engagement from real user value.",
    77: "Design permissions and prompt-injection defenses for an agent that can read files and run tools.",
    91: "Overall accuracy improved but a small subgroup got worse. What release policy makes this trade-off defensible?",
    95: "Why does difference-in-differences break under spillovers to neighboring regions, and what design fixes it?",
    103: "Set a decision policy when false-negative costs vary by customer and human review capacity is limited.",
    109: "I see rising tail latency in a concurrent Rust service: is it scheduler contention, allocator pressure, or lock convoys?",
    116: "My skilled teams adopted AI first and chose their timing. Is our measured productivity gain causal?",
    123: "An optimizer's cheap schedule violates rare constraints in production. Design verification and fallback around it.",
    131: "Detect benchmark contamination in a language model without its training data, even when questions are paraphrased.",
    147: "How should hourly reports handle events arriving up to 48 hours late without unbounded state?",
    149: "My model's calibration drifted after a policy change but ranking metrics held; diagnose and recalibrate without leakage.",
}

PLAINER = {  # HIGH: same hard core in plain words, so long technical vocabulary stops giving the tier away
    90: "Answers need facts from many long documents. Plan how to split the work, track the evidence, and test it.",
    115: "Propose checks that show whether a test measures real reasoning rather than memory or format tricks.",
    110: "Compare pooling data with training where data cannot leave each site and uneven sites drop out.",
    146: "Build a stock portfolio when return guesses are noisy, rules are messy, and big losses matter more than averages.",
    130: "Some sensors lie on purpose and links drop often. Track the true state of the system anyway.",
    140: "A compiler speed-up makes code faster but rarely gives wrong results. Find the bug systematically.",
    87: "Share out scarce aid in a disaster when needs are unclear, goals clash, and the facts keep changing.",
    14: "Version product prices so old invoices stay right and edits can run at once. Plan the tables, rollout, and undo.",
    20: "Judge how a new tax hits poor families, revenue, work choices, and regional gaps, and which parts rest on guesses.",
    93: "Plan how to handle private records used to train models, so you can trace, delete, limit, and audit every use.",
    138: "Does a three-tier model router cut cost without hurting answers? Plan the tests, metrics, and error checks.",
    78: "An async Rust server keeps its memory after traffic drops. Find where it goes and prove it.",
    55: "In a trial, some people skip treatment, some drop out, and effects differ. Say what you can estimate and how.",
    101: "A hospital triage model moves to new patient groups. List what to check before and after it goes live.",
    121: "Many customers upload untrusted archives and plugins to one service. Keep each one safely contained.",
    94: "Treatment data has gaps and hidden causes. Estimate the true effect and check how fragile it is.",
    111: "Wind, sun, and demand are uncertain. Decide which power plants run, weighing reliability, cost, emissions, and backup.",
    50: "A safety problem is so rare that a normal A/B test would need millions of users. Test a fix anyway.",
    30: "Two event feeds send duplicates and late records. Make each business action happen exactly once.",
    135: "For support chats, should you use a small custom model or a big general one? Weigh speed, handoffs, privacy, and mix.",
    134: "Prove or refute: a connected graph with an even number of odd-degree vertices always has an Euler circuit.",
    127: "Build a fair test for AI agents that use tools, when the tools change and tasks have several right answers.",
    100: "Keep approximate betweenness scores for a huge, changing graph in little memory. What fails, and when?",
    129: "Pick an optimizer for a bumpy objective with costly gradients and hard limits, and say how you would check it.",
    122: "Sensor data has gaps and a drifting baseline. Find rare anomalies without leaking future data into the model.",
}
REPLACED = {  # new questions of a different type, same tier, so question type stops giving the tier away
    8: "Find all solutions of x^2 - 2y^2 = 1 in positive integers with x below 1,000, and prove none are missing.",
    77: "Among 39 coins, one is fake and either heavier or lighter. What is the fewest balance weighings that always finds it?",
    72: "Count the ways to place 8 non-attacking rooks on a chessboard with none on the main diagonal.",
    40: "In what precise sense does 1 + 2 + 3 + ... equal -1/12, and in what sense is it false?",
    121: "How many squares, including tilted ones, have all four corners on the points of a 9 by 9 grid?",
    93: "What is the last nonzero digit of 100 factorial? Show a method that would also work for 1000 factorial.",
    87: "How many trailing zeros does 1000 factorial have when written in base 12? Justify each step.",
    14: "Why does a mutation in 21-hydroxylase raise androgen levels in congenital adrenal hyperplasia? Trace the pathway.",
    89: "Design a weekly chore chart that splits five household chores fairly among three roommates.",
    61: "Plan a packing list for a two-day beach trip with a toddler, grouped into clothes, food, and toys.",
    60: "Compare tea and coffee for someone who wants less caffeine in the evening, and suggest one option.",
    16: "A houseplant wilted after a week without water. What most likely caused it, and what should the owner do?",
    76: "Propose a name and a one-line slogan for a neighborhood book swap held every Saturday morning.",
    126: "Our team meetings keep running late. Suggest two rules that would help them finish on time.",
    139: "Sketch a three-step morning routine that helps two kids get ready for school on time.",
    58: "Why might a phone battery run out faster in very cold weather? Give one plain reason.",
    105: "Which design pattern would decouple a payment module from several shipping providers? Name it and explain why.",
    136: "What legal doctrine lets a court refuse to enforce a contract that was grossly unfair when signed?",
    73: "Why does a lithium-ion battery lose capacity faster when kept fully charged and hot? Explain the mechanism.",
    46: "Compare SSD and HDD storage for a database server, giving one workload that suits each.",
    120: "What causes the seasons on Earth, and why is distance from the Sun not the main reason?",
    133: "Outline test cases for a login form covering valid input, wrong passwords, and a locked account.",
}
FIX = {  # final pass: spreads tier-leaning words and question marks across tiers, restores #22, harder Pell equation
    1: "Tell population shift apart from feedback caused by the treatment itself when a clinical predictor degrades.",
    22: "When a predictor fails in a new region, separate covariate shift, label shift, concept drift, and pipeline bugs.",
    33: "Choose splits that reveal whether a graph neural network truly generalizes to organizations it never saw.",
    56: "Judge a coding agent whose passing solutions may still hide security flaws, slowdowns, or messy code.",
    91: "Overall accuracy rose but one small subgroup got worse. Write a release rule that makes this trade-off defensible.",
    147: "Hourly reports must absorb events arriving up to 48 hours late. Handle revisions without letting state grow forever.",
    138: "Check whether a three-tier router really cuts cost without hurting answers. Name the baselines, metrics, and error checks.",
    13: "Make long-running jobs that call outside APIs survive rate limits, partial failures, and side effects that cannot repeat.",
    28: "Fuse satellite, ground-sensor, and simulation readings at different resolutions while carrying uncertainty through honestly.",
    43: "Analyze conflicting clauses in a cross-border software contract: governing law, jurisdiction, liability, and privacy duties.",
    50: "A safety problem is so rare that a normal A/B experiment would need millions of users. Evaluate a fix anyway.",
    90: "Answers need facts from many long documents. Plan how to split the work, track the evidence, and check the final answer.",
    94: "Treatment records have gaps and hidden causes. Estimate the true effect and check how fragile it is.",
    101: "A hospital triage predictor moves to new patient groups. List what to check before and after it goes live.",
    110: "Compare pooling records with training where records cannot leave each site and uneven sites drop out.",
    115: "Propose checks that show whether a benchmark measures real reasoning rather than memory or format tricks.",
    116: "My skilled teams adopted the new assistant first and chose their timing. Is our measured productivity gain causal?",
    122: "Sensor readings have gaps and a drifting baseline. Find rare anomalies without leaking future readings into training.",
    127: "Build a fair benchmark for agents that use tools, when the tools change and tasks have several right answers.",
    131: "Detect benchmark contamination in a chatbot without its training corpus, even when questions are paraphrased.",
    135: "For support chats, should you pick a small custom system or a big general one? Weigh speed, handoffs, privacy, and mix.",
    149: "Our calibration drifted after a policy change but ranking metrics held; diagnose it and recalibrate without leakage.",
    55: "In a trial, some people skip treatment, some drop out, and effects differ. Explain what you can estimate and how.",
    129: "Compare optimizers for a bumpy objective with costly gradients and hard limits, and say how you would check the choice.",
    8: "Find the smallest positive integer solution of x^2 - 13y^2 = 1, and explain why no smaller one exists.",
    2: "How does a database transaction prevent partial updates when a payment operation fails halfway?",
    17: "Solar or wind power for a remote site with variable weather and limited battery storage: which fits better, and why?",
    42: "How can a browser cache make a site faster, and what problem can stale cached content cause?",
    44: "When does caching beat database indexing, and when is it the other way round? Give one trade-off each.",
    48: "A CSV contains duplicate customer IDs with different timestamps. Keep only the latest record for each, safely.",
    51: "REST or GraphQL for a mobile app that needs different fields on different screens: which suits it, and why?",
    53: "A dataset has several missing ages. Should you fill them with the mean or the median for a skewed distribution?",
    69: "When is a LEFT JOIN preferable to an INNER JOIN? Show it with a small library example.",
    107: "A web API returns intermittent 429 errors. What does that status mean, and how should the client respond?",
    137: "Using a classifier that flags spam among 100 emails, show what precision, recall, and F1 each measure.",
    117: "Last month my laptop died and I lost some photos. Explain why regular backups of personal data are useful.",
    34: "Write a Python function that returns the largest number in a list, plus one test for an empty list.",
    143: "Write a SQL query to count the rows in a data table called orders.",
    88: "I am learning JavaScript from an online course. Write a function that checks whether a number is even, with one test.",
    97: "Compare one advantage and one disadvantage of solar panels for a home.",
}
REFERENCE_ANSWERS = {  # numeric ones verified by computation
    8: "x = 649, y = 180 (fundamental solution, from the continued fraction of sqrt(13))",
    77: "4 weighings: n weighings can find the fake and tell heavier/lighter among at most (3^n - 3)/2 coins, and (3^4 - 3)/2 = 39",
    72: "14,833 (the number of derangements of 8)",
    121: "540 squares (sum over side vectors (a, b) of the placements; axis-aligned 204 plus tilted 336)",
    93: "4",
    87: "497 (1000! has 994 factors of 2 and 498 of 3; 12 = 2^2 * 3, so min(floor(994/2), 498) = 497)",
    40: "The series diverges, so it has no sum in the ordinary sense; -1/12 is the value of the analytically continued Riemann zeta function at -1 (also given by zeta and Ramanujan summation), not a limit of partial sums",
    14: "21-hydroxylase deficiency blocks cortisol (and aldosterone) synthesis; low cortisol removes negative feedback, ACTH rises, and the accumulated precursors (17-hydroxyprogesterone) are shunted into adrenal androgen synthesis",
}
RESTYLE = {  # MID: same task and tier, varied voice and openings (first person, semicolons, fewer "Explain...")
    7: "I need a low-downtime migration plan for a small Flask service moving from one virtual machine to containers.",
    37: "I am deciding whether to lease or buy a laptop. Compare costs over three years, including resale value and maintenance.",
    73: "My recipe for four people uses 300 g of rice. How should I scale the ingredients for seven people?",
    125: "I want a budget-conscious vegetarian day of meals that provides roughly 90 grams of protein; please plan it.",
    136: "My home uses 12 kWh daily. Calculate my 30-day consumption and the bill at ₹7.50 per kWh.",
    32: "Why can a random train-test split give misleading results for time-series forecasting?",
    96: "Why does binary search require sorted data, and what is its time complexity?",
    142: "How can class imbalance make accuracy misleading? Use a confusion-matrix example.",
    15: "What is the difference between correlation and causation, and when does confusing them lead to a false claim?",
    59: "What separates supervised from unsupervised learning? Give one realistic use case for each.",
    80: "How do precision and recall change when a fraud detector raises its decision threshold?",
    106: "How does public-key cryptography differ from shared-secret encryption, at a high level?",
}


def strip_suffixes(q):
    for s in SUFFIXES:
        q = q.replace(s, "")
    return re.sub(r"\s+", " ", q).strip()


def build():
    """The 150 original questions with every rewrite applied (used by make_300_balanced.py)."""
    rows = read_original()
    for i in LONGER:
        assert rows[i]["label"] == "LOW", (i, rows[i]["label"])
    for i in SHORTER:
        assert rows[i]["label"] == "HIGH", (i, rows[i]["label"])
    for i in RESTYLE:
        assert rows[i]["label"] == "MID", (i, rows[i]["label"])
    expected = {i: ("LOW" if i in LONGER or rows[i]["label"] == "LOW" else rows[i]["label"]) for i in REPLACED}
    for i in REPLACED:  # a replacement keeps the tier of the question it replaces
        assert rows[i]["label"] == expected[i], (i, rows[i]["label"])
    for i in PLAINER:
        assert rows[i]["label"] == "HIGH" and i not in SHORTER, (i, rows[i]["label"])
    out = []
    for i, r in enumerate(rows):
        r = dict(r)
        base = strip_suffixes(r["query"])
        edits = [(REPLACED, "replaced"), (LONGER, "longer"), (SHORTER, "shorter"), (PLAINER, "plainer"), (RESTYLE, "restyled")]
        hit = next(((d[i], name) for d, name in edits if i in d), None)
        if hit:
            r["original_query"] = r["query"]
            r["query"], r["edit"] = hit
        else:
            r["query"] = base
        if i in FIX:  # the final pass overrides any earlier rewrite
            r.setdefault("original_query", rows[i]["query"])
            r["query"] = FIX[i]
            r.setdefault("edit", "restyled")
        if i in REFERENCE_ANSWERS:
            r["reference_answer"] = REFERENCE_ANSWERS[i]
        r["word_count"] = len(r["query"].split())
        out.append(r)
    return out


if __name__ == "__main__":
    rows = build()
    print(f"{len(rows)} rows: {len(LONGER)} LOW made longer, {len(SHORTER)} HIGH made shorter, {len(PLAINER)} HIGH made plainer, "
          f"{len(RESTYLE)} MID restyled, {len(REPLACED)} replaced, {len(FIX)} in the final pass, "
          f"{len(REFERENCE_ANSWERS)} reference answers (written out by make_300_balanced.py)")
