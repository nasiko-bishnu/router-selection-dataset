"""Grow the balanced 150-question set to 300: 50 new questions per tier on 50 real-world topics.

Every new topic gets one LOW, one MID and one HIGH question, so the topic says nothing about the tier.
The new questions follow the rules the balanced 150 set was fixed to:
  - similar length in every tier (about 14-21 words),
  - openings, first-person framing ("my", "our", "I") and question marks mixed across tiers,
  - question types mixed across tiers (easy planning in LOW, terse questions in HIGH, arithmetic in MID),
  - none of the cue words the validator looks for ("simple", "basic", "short", "just", "hard", ...).
Tiers are relative to model size: LOW = a small (~8B) model answers reliably, MID = needs a ~70B
general model, HIGH = needs a frontier model. They are judgements until checked against real results.

The merged set is shuffled with a fixed seed so the tiers are not in blocks.

    python3 make_300_balanced.py   # -> llm_tier_selection_300_balanced.jsonl (the 150 rewrites come from make_150_balanced.py)
"""
import json
import random
from pathlib import Path

import make_150_balanced

HERE = Path(__file__).parent
DST = HERE / "llm_tier_selection_300_balanced.jsonl"
FIELDS = ("query", "label", "topic", "word_count", "reference_answer")  # what the dataset keeps

# topic: (LOW, MID, HIGH)
NEW = {
    "personal_finance": (
        "My monthly take-home pay is $3,000 and my rent is $1,050. What share of my income goes to rent?",
        "Build a monthly budget for a $3,000 income covering rent, groceries, transport, savings, and a yearly car bill.",
        "My freelance income swings 40% month to month. Set a savings rule that never misses quarterly taxes or rent."),
    "taxes": (
        "A friend is filing taxes for the first time. What is the difference between a deduction and a credit?",
        "Someone earned $8,000 freelancing and $50,000 in salary. Which records should they keep for filing, and why?",
        "A remote worker lived in two US states and abroad this year. Which places can tax which income?"),
    "loans": (
        "In plain words for a first-time buyer, what does the interest rate on a car loan mean?",
        "Compare 15-year and 30-year mortgages on $300,000 at 6%: monthly payment and total interest for each.",
        "Should I prepay my 4% mortgage or invest instead, given taxes, inflation risk, job insecurity, and a ten-year horizon?"),
    "investing": (
        "My bank advisor keeps calling index funds diversified. What does that word mean for my savings?",
        "An investment grew from $10,000 to $16,000 over five years. What was its average yearly growth rate?",
        "Why can a portfolio with a higher average return still leave a retiree worse off over thirty years?"),
    "insurance": (
        "New homeowners often ask this: name two things a standard home insurance policy usually covers.",
        "A health plan has a $1,500 deductible and 20% coinsurance. What do I pay on a $6,000 bill?",
        "Price crop insurance for farmers whose drought losses hit the whole region in the same years."),
    "cooking": (
        "For a picnic tomorrow, how long should eggs boil to be hard, and how should they be cooled?",
        "My cake recipe serves eight from a 9-inch pan. What pan size and baking changes make it serve twelve?",
        "My sourdough keeps collapsing at 2,500 meters altitude. Adjust hydration, yeast activity, and baking to fix it."),
    "nutrition": (
        "Which common foods are good sources of vitamin C for children getting over a cold?",
        "Plan a day of meals around 2,000 calories for someone avoiding dairy, with at least 25 grams of fiber.",
        "A marathon runner has celiac disease and low iron. Plan meals that fix both without nutrients blocking each other."),
    "fitness": (
        "After a 30-minute evening dog walk, which three stretches help the legs recover?",
        "Design a four-week plan that takes me from walking to running 5 km, with rest days.",
        "Two studies disagree on whether interval training beats steady cardio for weight loss. Reconcile their results."),
    "medicine": (
        "It was very hot at the football match today. What are common signs of dehydration in adults?",
        "A child has a 39°C fever and a rash that doesn't fade under a pressed glass. What should a parent do?",
        "An older patient on warfarin starts a new antibiotic and gets nosebleeds. Work through the interaction and next steps."),
    "pharmacology": (
        "My doctor said to finish the antibiotics even if I feel better. Why does that matter?",
        "Ibuprofen or paracetamol for a headache in someone with a past stomach ulcer: which is safer, and why?",
        "Why does grapefruit juice raise blood levels of some drugs but not others? Use enzyme and transporter details."),
    "mental_health": (
        "Work has been stressful all week. Give two calming things to do in the hour before sleep.",
        "My colleague seems withdrawn and keeps missing deadlines. How can I raise it kindly without overstepping?",
        "A school wants to screen every student for depression. Weigh false positives, consent, stigma, and follow-up capacity."),
    "parenting": (
        "Our toddler fights brushing teeth every night. What are some gentle ways to make it a habit?",
        "Write a fair screen-time agreement for siblings aged 8 and 13 who have different school schedules.",
        "We are divorced parents in different countries who disagree on our teen's school. Propose a process courts would accept."),
    "education": (
        "For a second-grade spelling quiz on Friday, write three words that include the 'ch' sound.",
        "Turn 'students will understand photosynthesis' into a measurable lesson objective, and add one way to assess it.",
        "Test scores rose after a new curriculum, but tutoring grew and enrollment shifted. Can the curriculum take credit?"),
    "career": (
        "After a job interview yesterday afternoon, write a two-line message thanking the interviewer for their time.",
        "Rewrite this resume bullet to show impact: 'Responsible for handling customer complaints in the store.'",
        "I have a startup offer with equity and a stable job with a pension. How do I compare them with unknown exit odds?"),
    "workplace": (
        "What is the purpose of a probation period for employees starting a new job?",
        "Plan an on-call rotation for six engineers in three time zones so nobody works two weekends in a row.",
        "Design pay bands that close a gender pay gap without cutting anyone's salary or exceeding the budget."),
    "tenant_law": (
        "In general terms, can a landlord usually enter a rented flat without giving the tenant any notice?",
        "My landlord kept my whole deposit for 'cleaning'. What steps should I take, and what evidence helps?",
        "A tenant sublet on Airbnb against the lease and a guest was injured in a rent-controlled building. Who is liable?"),
    "contracts": (
        "My new freelance client keeps saying we should work 'in good faith'. What does that phrase mean?",
        "Spot the risky clauses in a freelance contract with unlimited revisions and payment 'upon client satisfaction'.",
        "Draft a force-majeure clause for a supply deal covering pandemics and sanctions but not ordinary price rises."),
    "travel": (
        "My parents are flying abroad for the first time. What documents do they usually need at the airport?",
        "I want a five-day rail trip from Rome to Paris under €600, with two museum days and no night trains.",
        "Plan two weeks across four countries for my wheelchair-using father, covering visas, step-free transport, and medicines."),
    "commuting": (
        "For a daily 10 km commute to the office, which is usually cheaper over a month, a bus pass or taxis?",
        "A bus leaves every 12 minutes and a train every 20. Both left at 8:00; when do they next leave together?",
        "Redesign a city bus network to cut average travel time without dropping riders who depend on night routes."),
    "cars": (
        "A check-engine light comes on during a drive and stays on. What does that usually mean?",
        "My brakes squeal only on mornings after rain and stop after a few minutes. What is likely happening?",
        "Our electric delivery fleet wants less battery wear. Plan charging that balances heat, depth of discharge, and deadlines."),
    "home_repair": (
        "Before fixing a leaking kitchen tap this weekend, how should the water supply be turned off safely?",
        "Black mold keeps returning on our bathroom ceiling every winter. Find the likely cause and a lasting fix.",
        "Cracks in my old house widen in clay soil during droughts. Should I underpin, improve drainage, or monitor?"),
    "gardening": (
        "Which vegetables grow well in large pots on a sunny balcony for a first-time gardener?",
        "My tomato leaves turn yellow from the bottom up while fruit is forming. What are likely causes and fixes?",
        "Plan a five-year rotation for a small farm that rebuilds soil, controls pests without sprays, and keeps income steady."),
    "agriculture": (
        "On a visit to my uncle's farm, he mentioned rotating crops. Why do farmers rotate crops between seasons?",
        "A 12-hectare farm needs 40% legumes and the rest split evenly between wheat and maize. How many hectares each?",
        "Groundwater is falling under a farming valley. Design water rules that farmers, towns, and wildlife could all accept."),
    "weather": (
        "Thunderstorms are forecast for this afternoon. What should I pack for a morning day hike in the hills?",
        "What does a 70% chance of rain in a forecast actually mean, and what does it not mean?",
        "What evidence and methods would make it scientifically defensible to blame one heatwave on climate change?"),
    "energy_bills": (
        "Our electricity bill doubled this summer because of air conditioning. Give three ways to bring it down.",
        "A heater uses 1.5 kW for 4 hours a day. At $0.20 per kWh, what does a month cost?",
        "Should we buy solar panels, a battery, or both, with time-of-use prices, export limits, and a ten-year payback goal?"),
    "environment": (
        "Our building recently started a recycling program. Which household items can usually go in the recycling bin?",
        "Paper or plastic shopping bags: which has the lower environmental impact once you count how often each is reused?",
        "A city debates banning gas stoves. Weigh indoor air, emissions, grid load, renter costs, and fairness before deciding."),
    "shopping": (
        "At the supermarket, is a 1 kg bag of rice for $3 cheaper per kilogram than 500 g for $1.60?",
        "For three items at the same price, is 'buy two, get the third half price' better than 25% off everything?",
        "Design a returns policy for my online clothing shop that cuts fraud and waste without driving off honest customers."),
    "small_business": (
        "My sister is opening a small bakery that sells sourdough and coffee. Suggest three names for it.",
        "A café sells 200 coffees a day at $3.50, costing $1.20 each, and rent is $4,000 monthly. What is monthly profit?",
        "Our family business must choose between a second shop and selling online. Compare cash flow, risk, and timing."),
    "marketing": (
        "Write a catchy one-line ad for our town's weekend farmers' market that runs every Saturday morning.",
        "Plan an email campaign to win back customers who haven't ordered in six months, with subject lines and timing.",
        "Our ad spending rose during the busy holiday season. How can we tell how many extra sales the ads caused?"),
    "logistics": (
        "An online order email included a tracking number. What is it for, and how is it used?",
        "Plan routes for three vans covering twelve delivery stops so each van works roughly the same number of hours.",
        "Port strikes and rising freight costs hit a global supplier. Reshape inventory and sourcing to keep delivery promises."),
    "real_estate": (
        "My cousin is moving to a new city. What is the difference between renting and leasing an apartment?",
        "Should I buy a $250,000 flat or keep renting at $1,200 a month over five years, counting fees and repairs?",
        "Value an office building whose main tenant may leave in two years while remote work keeps rising."),
    "sports": (
        "My son starts soccer practice next week. How many players does each team have on the field?",
        "Explain soccer's offside rule using an example where an attacker is exactly level with the last defender.",
        "Build a fair ranking for chess clubs whose players rarely meet across regions and play different numbers of games."),
    "music": (
        "Before a first classical concert tonight, which instruments usually make up a string quartet?",
        "Why does a melody sound sadder when moved from a major key to its relative minor? Give an example.",
        "Two hit songs share a four-chord progression and similar melodies. Is that likely copyright infringement, and why?"),
    "books_film": (
        "At bedtime tonight, summarize the story of Cinderella in three sentences for a four-year-old.",
        "Compare the narrators of 'The Great Gatsby' and 'To Kill a Mockingbird' and how each shapes the story.",
        "Can an unreliable narrator still be morally trustworthy? Argue it with two contrasting novels as evidence."),
    "history": (
        "My grandfather served in the army during the war. In which year did World War II end in Europe?",
        "Why did the fall of Constantinople in 1453 matter for European trade routes? Give two effects.",
        "Did the printing press cause the Reformation? Weigh the strongest evidence for and against that claim."),
    "geography": (
        "On a world map at school, which ocean lies between Africa and Australia's west coast?",
        "Why are so many major cities built on rivers or coasts? Give three reasons with real examples.",
        "How did the Aswan High Dam change the Nile's floods, and who gained or lost from that change?"),
    "languages": (
        "I am visiting Madrid next month. How do I say 'Where is the nearest pharmacy?' in Spanish?",
        "When does Spanish use 'ser' versus 'estar'? Give two sentences where switching them changes the meaning.",
        "Translate an English liability clause into Japanese while keeping both its legal force and its politeness level."),
    "email_writing": (
        "Write a two-sentence email telling my team the office will be closed next Monday for repairs.",
        "Rewrite this so it stays firm but polite: 'Your service is useless and I want my money back now.'",
        "Write a public apology after a customer data leak that admits fault without creating extra legal exposure."),
    "public_speaking": (
        "Before a class presentation on Friday, what are three tips for good eye contact with the audience?",
        "Structure a five-minute wedding toast that blends one funny story with a sincere message to the couple.",
        "Prepare a town-hall briefing on a factory closure, with answers to the three hardest questions residents will ask."),
    "smartphones": (
        "My grandmother has a new Android phone. How does she take a screenshot of a message?",
        "My phone battery drains fast even when idle. List likely causes and how to check each in settings.",
        "Should doctors photograph patients on personal phones? Weigh privacy law, security controls, and clinical convenience."),
    "scams": (
        "I received an odd email asking me to confirm my password. What are warning signs of phishing?",
        "A 'bank' caller asks me to read out a code they texted me. What is happening, and what should I do?",
        "Design account-takeover checks for a payments app that do not block honest travelers logging in from new devices."),
    "programming": (
        "I am new to Python. Write one line of code that prints the numbers from 1 to 10.",
        "Write a Python function that merges overlapping time intervals, turning [1,3] and [2,6] into [1,6].",
        "Design a lock-free queue that stays correct despite the ABA problem, and explain why it is safe."),
    "spreadsheets": (
        "In a household budget spreadsheet, which formula adds up the values in cells A1 to A10?",
        "Write a spreadsheet formula that finds a product's price by its code on another sheet.",
        "Two sales spreadsheets disagree by 3% after currency conversion, refunds, and time-zone cutoffs. Reconcile them step by step."),
    "wifi_networking": (
        "Our neighbor asked to use our Wi-Fi. What does setting a Wi-Fi password protect us from?",
        "Our home Wi-Fi is fast near the router but weak upstairs. Give three practical fixes and their rough costs.",
        "Our pages load slowly for one region only in the evenings. Trace the cause across DNS, the CDN, and peering."),
    "surveys": (
        "In a school survey, 30 out of 120 parents said yes. What percentage of parents said yes?",
        "Why can a company's average salary be far above its typical salary? Show it with a small example.",
        "An online survey skews young and urban. Weight the responses so the estimates represent the whole country."),
    "government": (
        "My city is electing a new mayor this year. What does a mayor usually do in local government?",
        "Compare first-past-the-post and proportional voting using three parties that win 40%, 35%, and 25% of votes.",
        "Design a redistricting process that resists gerrymandering yet keeps communities together, with measurable fairness tests."),
    "public_health": (
        "Flu season is starting at my children's school. Why does regular handwashing help so much?",
        "A town of 20,000 people had 400 flu cases this week. What is the attack rate per 1,000 people?",
        "Allocate scarce vaccines across regions with different infection rates and different levels of trust in health officials."),
    "astronomy": (
        "My daughter asked why the Moon looks different each night this month. What should I tell her?",
        "Why do astronauts on the space station feel weightless when gravity there is still about 90% of Earth's?",
        "How do astronomers read an exoplanet's atmosphere from starlight, and what could fake a detection?"),
    "household_chemistry": (
        "Why should bleach never be mixed with ammonia-based cleaners when cleaning the bathroom at home?",
        "Why does baking soda absorb fridge smells, and when would vinegar work better for cleaning?",
        "Why do lithium battery fires restart after being put out? Plan safe storage for our warehouse of e-bikes."),
    "pets": (
        "We are adopting a puppy next week. Which common foods are poisonous to dogs?",
        "My cat stopped using the litter box after we moved house. What are likely reasons and fixes?",
        "A shelter has an outbreak of a contagious dog virus. Plan intake, isolation, and adoptions without overcrowding."),
}


# Balancing pass after the first validation: spreads question marks, first person, and words that leaned
# by tier ("usually", "weigh", "without", openings "Two..." / "Before...") across tiers.
FIX = {
    ("workplace", 0): "What is a probation period for employees starting a new job meant to achieve?",
    ("wifi_networking", 0): "A neighbor asked to share a Wi-Fi network. What does setting a password protect against?",
    ("commuting", 0): "Should someone with a daily 10 km commute buy a monthly bus pass or take taxis?",
    ("pharmacology", 1): "Should someone with a past stomach ulcer take ibuprofen or paracetamol for a headache, and why?",
    ("smartphones", 2): "Is it acceptable for doctors to photograph patients on personal phones, given privacy law and security controls?",
    ("taxes", 0): "A friend is filing taxes for the first time. Explain the difference between a deduction and a credit.",
    ("loans", 0): "Explain in plain words what the interest rate on a car loan means for a first-time buyer.",
    ("cooking", 0): "For a picnic tomorrow, give the boiling time and the cooling method for eggs with firm yolks.",
    ("nutrition", 0): "List common foods that are good sources of vitamin C for children getting over a cold.",
    ("medicine", 0): "It was very hot at the football match today. List common signs of dehydration in adults.",
    ("tenant_law", 0): "In general terms, may a landlord enter a rented flat without giving the tenant any notice?",
    ("travel", 0): "My parents are flying abroad for the first time. List the documents they will need at the airport.",
    ("cars", 0): "A check-engine light comes on during a drive and stays on. Explain what that typically means.",
    ("home_repair", 0): "To fix a leaking kitchen tap this weekend, explain how to turn off the water supply safely.",
    ("environment", 0): "A building recently started a recycling program. Name household items that belong in the recycling bin.",
    ("music", 0): "Name the instruments that make up a string quartet, for someone attending a first classical concert tonight.",
    ("public_speaking", 0): "Give three tips for good eye contact with the audience during a class presentation on Friday.",
    ("government", 0): "A city is electing a new mayor this year. Explain what a mayor does in local government.",
    ("scams", 0): "An odd email asks the reader to confirm a password. List warning signs that it is phishing.",
    ("cars", 1): "My brakes usually squeal only on mornings after rain and stop after a few minutes. What is happening?",
    ("fitness", 1): "Design a four-week plan that takes me from walking to running 5 km without injury, with rest days.",
    ("fitness", 2): "Studies disagree on whether interval training beats steady cardio for weight loss. Why, and which result holds up?",
    ("music", 2): "Is it copyright infringement when two hit songs share a four-chord progression and similar melodies?",
    ("spreadsheets", 2): "Why might our two sales spreadsheets disagree by 3% after currency conversion, refunds, and time-zone cutoffs?",
    ("mental_health", 2): "Should a school screen every student for depression, given false positives, consent, stigma, and limited follow-up?",
    ("environment", 2): "Should our city ban gas stoves, given indoor air, emissions, grid load, renter costs, and fairness?",
    ("workplace", 2): "How can our pay bands close a gender pay gap while nobody's salary is cut and the budget holds?",
    ("shopping", 2): "How should my online clothing shop handle returns to cut fraud and waste while keeping honest customers?",
    ("email_writing", 2): "How do you write a public apology after a customer data leak that admits fault but limits legal exposure?",
    ("commuting", 2): "How would you redesign a city bus network to cut travel time while protecting riders who rely on night routes?",
    ("pets", 2): "A shelter has an outbreak of a contagious dog virus. How should intake, isolation, and adoptions change?",
    ("travel", 2): "Plan two weeks in four countries for my father in a wheelchair: visas, step-free transport, and medicines.",
    ("government", 2): "How would you draw voting districts that resist gerrymandering and keep communities together? Name fairness tests.",
    ("public_health", 2): "How should scarce vaccines be split across regions with different infection rates and trust in officials?",
    ("gardening", 2): "Plan a five-year farm rotation that rebuilds soil, controls pests with no sprays, and keeps income steady.",
}


# ---------------------------------------------------------------------------------------------------
# Pass 3: even out question types and domains across tiers, and add format variety (pasted context,
# longer prompts, other languages) to every tier. 25 replacements per tier; each keeps its tier.
# old question (exact text) -> (new question, domain)
REPLACE_V3 = {
    # ---------------- LOW ----------------
    "Our toddler fights brushing teeth every night. What are some gentle ways to make it a habit?": ("Plan a one-day city walk for a visitor who likes parks, a bakery stop, and being back by 5 pm.", "travel_transport"),
    "Work has been stressful all week. Give two calming things to do in the hour before sleep.": ("Outline a weekend schedule for two friends running a lemonade stand on Saturday and Sunday afternoons.", "business_work"),
    "Give three practical ways a household can reduce food waste without buying special equipment.": ("Create a weekly reminder plan to back up phone photos to a cloud folder.", "security_tech"),
    "A school garden keeps losing topsoil after heavy rain each spring. What are two common causes of soil erosion in general?": ("Design a three-question quiz that checks whether a student knows what a chatbot is.", "ai_ml"),
    "A building recently started a recycling program. Name household items that belong in the recycling bin.": ("Structure a thank-you card for a teacher with an opening line, one memory, and a closing wish.", "language_writing"),
    "Write a Python expression that sorts a list of numbers in ascending order.": ("Create a daily water reminder plan for an office worker who often forgets to drink during meetings.", "health_medicine"),
    "Write a SQL query that returns every column and row from a table named employees.": ("Set up three savings jars for a ten-year-old's pocket money and say what each jar is for.", "money_finance"),
    "What does a compiler do, and how is its role different from a text editor.": ("Plan a family movie night for four with snacks, a start time, and a film everyone can watch.", "culture_history"),
    "What is the purpose of an HTML link element on a webpage.": ("Outline what to bring to a city office when renewing a driver's license.", "law_government"),
    "Draft a polite message asking a colleague to reschedule tomorrow's meeting to Friday.": ("A class of 25 students has 15 girls. What fraction of the class are boys?", "math"),
    "Translate this into Spanish: 'Please send me the updated report tomorrow morning.'.": ("A car uses 6 liters of fuel per 100 km. How much fuel does a 250 km trip need?", "travel_transport"),
    "Summarize the water cycle in four sentences, including evaporation and precipitation.": ("A recipe needs 200 g of flour for 10 cookies. How much flour makes 25 cookies?", "food_fitness"),
    "For a second-grade spelling quiz on Friday, write three words that include the 'ch' sound.": ("Which is safer for an online account, a long passphrase or a short password with symbols?", "security_tech"),
    "Explain what a password manager does and why it can be useful.": ("Should a family of four take the train or drive for a 120 km day trip with luggage?", "travel_transport"),
    "Explain what a database primary key does and why duplicate values are usually disallowed.": ("Which is better for checking today's weather, a chatbot or a weather app, and why?", "ai_ml"),
    "A city is electing a new mayor this year. Explain what a mayor does in local government.": ("This error appeared on my screen: 'File not found: report.docx'. What does it mean, and what should I check?", "software_coding"),
    "Explain in plain words what the interest rate on a car loan means for a first-time buyer.": ("Here is a sentence from my essay: 'Their going to the park tomorrow.' Fix the mistake and explain it.", "language_writing"),
    "What does DNS do when someone enters a website name into a browser.": ("My bank texted: 'Your card was used for $1 at QuickMart. Reply YES if this was you.' What should I do?", "security_tech"),
    "Name two common seasonal allergy symptoms and one simple way to reduce pollen exposure.": ("A medicine label says 'Take one tablet every 8 hours, no more than 3 in 24 hours.' What is the daily maximum?", "health_medicine"),
    "What is the difference between a noun and a verb? Give examples.": ("My friend visits next week from abroad. She arrives Friday evening, leaves Sunday night, loves food markets and old buildings, and does not drive. Suggest a plan for Saturday in a typical European city.", "travel_transport"),
    "What does recycling mean, and name one material that can be recycled.": ("I get paid on the first of every month and keep forgetting which bills are due when. Rent is due on the 5th, phone on the 12th, and internet on the 20th. Make me a reminder list for the month.", "money_finance"),
    "List the main ingredients in hummus and name one common optional addition.": ("My nephew is in fourth grade and has a spelling test every Friday. He gets nervous and forgets words he knew the night before. Suggest three practice ideas his parents can try at home this week.", "language_writing"),
    "What is the difference between a virus and a bacterium, and give one example of each.": ("¿Cuál es la capital de Australia y en qué continente se encuentra?", "culture_history"),
    "A box has 5 red and 3 blue balls. How many balls are there altogether.": ("Quelle est la différence entre un fruit et un légume ? Donne un exemple de chaque.", "food_fitness"),
    "My son starts soccer practice next week. How many players does each team have on the field?": ("बारिश के मौसम में मच्छरों से बचने के तीन तरीके बताइए।", "health_medicine"),
    # ---------------- MID ----------------
    "A phone battery drops from 80% to 56% in three hours at a steady rate. Estimate the percentage after five hours total.": ("Should we repaint a damp basement wall now or fix the moisture first? Explain the order and why.", "home_family"),
    "A student scores 72, 81, and 87 on three equally weighted tests. What score is needed on the fourth to average 82?": ("Is an air fryer or a convection oven better for a family of five that bakes every week? Explain.", "food_fitness"),
    "A shop sells 3 notebooks for ₹90 each and gives 10% off the total. What is the final price?": ("Should I book a cheaper flight with a 50-minute connection or a pricier direct one for a wedding?", "travel_transport"),
    "A company has fixed costs of ₹12,000 and earns ₹80 per item after variable costs. How many items cover fixed costs?": ("Which is safer for a small office: one shared Wi-Fi password, or separate guest and staff networks?", "security_tech"),
    "A train travels 150 km at 60 km/h and returns at 75 km/h. Find its average speed.": ("Write a 100-word museum label for a 1920s jazz record player that a teenager would enjoy reading.", "culture_history"),
    "A fair six-sided die is rolled twice. Calculate the probability that the sum is at least 10.": ("Rewrite this lease rule in plain English: 'Lessee shall not sublet the premises absent prior written consent.'", "law_government"),
    "Given monthly sales of 10, 12, 11, 15, and 17 units, calculate the mean and describe the trend cautiously.": ("Our dog barks at every delivery driver. What training steps reduce it over a few weeks?", "home_family"),
    "A price rises by 20% and then falls by 20%. Is it back to its original value? Explain.": ("A ceiling fan wobbles at high speed. What are the likely causes, and how do I check each?", "home_family"),
    "Design a simple database relationship for customers who can place many orders, naming the keys.": ("What is the difference between a cold and the flu, and when should someone see a doctor?", "health_medicine"),
    "Design three REST endpoints for a library-book service, specifying each HTTP method and purpose.": ("Who were the main sides in the Cold War, and why did it never become a direct war between them?", "culture_history"),
    "Write a Python function that groups words by their first letter, ignoring case.": ("What happens legally if someone dies without a will? Give the general rule in most countries.", "law_government"),
    "When is a LEFT JOIN preferable to an INNER JOIN? Show it with a small library example.": ("Our train was cancelled and the next one arrives three hours late. What compensation rights usually apply in the EU?", "travel_transport"),
    "Return the first repeated integer in a Python list, or None if no value repeats; preserve input order.": ("My bread comes out dense every time. What are the three most likely causes?", "food_fitness"),
    "Create a two-day itinerary for a visitor who likes museums, walking, and inexpensive food, with no car.": ("What does two-factor authentication protect against, and what can it still not stop?", "security_tech"),
    "Design a small inventory alert rule using stock level, average daily sales, and supplier lead time.": ("A small shop's sales dip every January. What are likely reasons, and two ways to soften the dip?", "business_work"),
    "I want a budget-conscious vegetarian day of meals that provides roughly 90 grams of protein; please plan it.": ("This Python line fails on some days: total = sum(prices) / len(prices). It raises ZeroDivisionError. Why, and how do I fix it?", "software_coding"),
    "Plan an email campaign to win back customers who haven't ordered in six months, with subject lines and timing.": ("Here is my cover letter opening: 'I am writing to apply for the job I saw.' Rewrite it to sound confident and specific.", "language_writing"),
    "Why does binary search require sorted data, and what is its time complexity?": ("Survey results from 80 people: 'Satisfied 40%, Neutral 35%, Unsatisfied 25%'. What can and can't we conclude?", "data_statistics"),
    "How do precision and recall change when a fraud detector raises its decision threshold?": ("My blood test says 'LDL 165 mg/dL, HDL 38 mg/dL'. What do these numbers mean in general terms?", "health_medicine"),
    "Why can a random train-test split give misleading results for time-series forecasting?": ("I run a small bakery with two staff. On weekdays we sell out of bread by noon but throw away pastries at closing, and on weekends it is the opposite. Suggest how to adjust what we bake each day.", "business_work"),
    "Using a classifier that flags spam among 100 emails, show what precision, recall, and F1 each measure.": ("Our family of four shares one car. My partner commutes 30 km on weekdays, the kids have sports twice a week across town, and I work from home. Propose a weekly plan so everyone gets where they need to go.", "home_family"),
    "How can class imbalance make accuracy misleading? Use a confusion-matrix example.": ("I want a chatbot to help answer customer emails for my online shop. Most emails ask about delivery times, returns, and sizes. What should I check before letting it reply to customers?", "ai_ml"),
    "A project has 5 tasks with dependencies A before B and C, and D after both B and C. Draw a valid ordering if E follows D.": ("¿Por qué sube el precio de los vuelos cuanto más se acerca la fecha del viaje? Explica dos razones.", "travel_transport"),
    "Summarize an opinion piece while separating its central claim from the evidence offered to support it.": ("Was ist der Unterschied zwischen einer Kreditkarte und einer Debitkarte, und wann ist welche sinnvoller?", "money_finance"),
    "Write a regular expression that matches a simple email-like string, and mention why it is not full email validation.": ("Pourquoi la glace flotte-t-elle sur l'eau alors que la plupart des solides coulent dans leur propre liquide ?", "science_environment"),
    # ---------------- HIGH ----------------
    "Propose checks that show whether a benchmark measures real reasoning rather than memory or format tricks.": ("How many ways can you climb 20 stairs taking 1, 2, or 3 steps at a time? Give the exact count.", "math"),
    "Build a fair benchmark for agents that use tools, when the tools change and tasks have several right answers.": ("What is the probability that a random 5-card poker hand contains exactly two pairs? Give an exact fraction.", "math"),
    "Detect benchmark contamination in a chatbot without its training corpus, even when questions are paraphrased.": ("How many integers from 1 to 1,000,000 are neither perfect squares nor perfect cubes?", "math"),
    "Check whether a three-tier router really cuts cost without hurting answers. Name the baselines, metrics, and error checks.": ("A family has two children, and at least one is a boy born on a Tuesday. What is the chance both are boys?", "math"),
    "Judge a coding agent whose passing solutions may still hide security flaws, slowdowns, or messy code.": ("Why does searing meat not actually seal in juices, and what does it really change chemically?", "food_fitness"),
    "Overall accuracy rose but one small subgroup got worse. Write a release rule that makes this trade-off defensible.": ("Why did the 1929 stock market crash turn into a decade-long depression rather than a short recession?", "culture_history"),
    "Set a decision policy when false-negative costs vary by customer and human review capacity is limited.": ("Why can a vaccine with 95% efficacy still mean most infected people in a hospital were vaccinated?", "health_medicine"),
    "An optimizer's cheap schedule violates rare constraints in production. Design verification and fallback around it.": ("Why can an end-to-end encrypted messenger still reveal who talks to whom? Explain metadata attacks and defenses.", "security_tech"),
    "Build a stock portfolio when return guesses are noisy, rules are messy, and big losses matter more than averages.": ("Write a persuasive op-ed for parents arguing for later school start times, addressing the strongest counterargument fairly.", "language_writing"),
    "Answers need facts from many long documents. Plan how to split the work, track the evidence, and check the final answer.": ("Translate into French so the meaning, not the words, survives: 'He spilled the beans and let the cat out of the bag.'", "language_writing"),
    "Our electric delivery fleet wants less battery wear. Plan charging that balances heat, depth of discharge, and deadlines.": ("Write a function that returns the longest palindromic substring in linear time, and explain why it is linear.", "software_coding"),
    "Design account-takeover checks for a payments app that do not block honest travelers logging in from new devices.": ("Implement a lock-free single-producer single-consumer ring buffer in C++ and justify each memory ordering you use.", "software_coding"),
    "Compare optimizers for a bumpy objective with costly gradients and hard limits, and say how you would check the choice.": ("Our 1950s house has knob-and-tube wiring, an old fuse box, and a planned EV charger. What must change, in what order?", "home_family"),
    "Retrieved passages each look right but disagree. When should the system trust one, cite both, or refuse?": ("My dog has limped on and off for a month, worse after rest. Work through likely causes and tests that separate them.", "home_family"),
    "Compare pooling records with training where records cannot leave each site and uneven sites drop out.": ("Plan a week of meals for a teenager with type 1 diabetes who plays football, balancing carbs, insulin timing, and growth.", "food_fitness"),
    "Choose splits that reveal whether a graph neural network truly generalizes to organizations it never saw.": ("Our API logs show 'p50 40ms, p99 2.1s, error rate 0.1%, GC pauses up to 1.8s'. What is the likely culprit, and what evidence would confirm it?", "software_coding"),
    "Wind, sun, and demand are uncertain. Decide which power plants run, weighing reliability, cost, emissions, and backup.": ("A contract says: 'Supplier is not liable for any loss, howsoever arising.' Would a court likely enforce this against gross negligence?", "law_government"),
    "For support chats, should you pick a small custom system or a big general one? Weigh speed, handoffs, privacy, and mix.": ("Lab results: 'TSH 0.1, free T4 high, heart rate 115, weight loss 6 kg in two months.' Reason through the likely diagnosis and next tests.", "health_medicine"),
    "Tell population shift apart from feedback caused by the treatment itself when a clinical predictor degrades.": ("A study reports: 'Coffee drinkers lived 2 years longer (p = 0.03, n = 500, observational).' What else could explain this besides coffee?", "data_statistics"),
    "Sensor readings have gaps and a drifting baseline. Find rare anomalies without leaking future readings into training.": ("I run a 40-person software consultancy. Two clients bring 60% of revenue, one pays late, and a third wants a fixed price for a vague project. Our cash covers three months. What should I do this quarter, and why?", "business_work"),
    "Some sensors lie on purpose and links drop often. Track the true state of the system anyway.": ("My elderly mother lives alone two hours away. She has mild memory loss, refuses to move, and my siblings disagree about paying for care. Lay out the options, costs, and a way to decide together.", "home_family"),
    "Fuse satellite, ground-sensor, and simulation readings at different resolutions while carrying uncertainty through honestly.": ("Historians still debate why Bronze Age Mediterranean civilizations collapsed around 1200 BCE. Weigh drought, invasions, earthquakes, and trade breakdown, and say which best fits the archaeological record.", "culture_history"),
    "Two event feeds send duplicates and late records. Make each business action happen exactly once.": ("¿Por qué la inflación puede bajar el desempleo a corto plazo pero no a largo plazo? Explica la curva de Phillips.", "money_finance"),
    "Keep approximate betweenness scores for a huge, changing graph in little memory. What fails, and when?": ("Ein Unternehmen will seine Server in die Cloud verlegen. Welche versteckten Kosten und Risiken sollte es vorher prüfen?", "software_coding"),
    "A hospital triage predictor moves to new patient groups. List what to check before and after it goes live.": ("डॉक्टर एक ही लक्षण वाले दो मरीजों को अलग दवाइयाँ क्यों दे सकते हैं? दवा-अंतःक्रिया और आनुवंशिकी के आधार पर समझाइए।", "health_medicine"),
}
REFERENCE_V3 = {  # verified by computation
    "A class of 25 students has 15 girls. What fraction of the class are boys?": "2/5",
    "A car uses 6 liters of fuel per 100 km. How much fuel does a 250 km trip need?": "15 liters",
    "A recipe needs 200 g of flour for 10 cookies. How much flour makes 25 cookies?": "500 g",
    "A medicine label says 'Take one tablet every 8 hours, no more than 3 in 24 hours.' What is the daily maximum?": "3 tablets",
    "How many ways can you climb 20 stairs taking 1, 2, or 3 steps at a time? Give the exact count.": "121,415",
    "What is the probability that a random 5-card poker hand contains exactly two pairs? Give an exact fraction.": "123,552 / 2,598,960 = 198/4165 (about 0.0475)",
    "How many integers from 1 to 1,000,000 are neither perfect squares nor perfect cubes?": "998,910 (1,000 squares + 100 cubes - 10 sixth powers = 1,090 excluded)",
    "A family has two children, and at least one is a boy born on a Tuesday. What is the chance both are boys?": "13/27",
}

# Pass 3b: last domain balancing (math, data, security, travel, law were still uneven by tier).
REPLACE_V3B = {
    "Prove or refute: a connected graph with an even number of odd-degree vertices always has an Euler circuit.": ("Our small clinic keeps patient files on a shared laptop. Plan encryption, backups, and access rules that survive a stolen device.", "security_tech"),
    "In what precise sense does 1 + 2 + 3 + ... equal -1/12, and in what sense is it false?": ("How could attackers abuse a smart doorbell on a home network, and which defenses matter most for a non-technical family?", "security_tech"),
    "How many trailing zeros does 1000 factorial have when written in base 12? Justify each step.": ("Why can SMS-based two-factor codes be stolen even without the phone, and which stronger options work for ordinary users?", "security_tech"),
    "How many squares, including tilted ones, have all four corners on the points of a 9 by 9 grid?": ("Plan an overland route from Delhi to Istanbul for a solo traveler, handling visas, border closures, and seasonal mountain passes.", "travel_transport"),
    "In a trial, some people skip treatment, some drop out, and effects differ. Explain what you can estimate and how.": ("Why do airlines overbook flights on purpose, and how do they decide how many extra seats to sell?", "travel_transport"),
    "Treatment records have gaps and hidden causes. Estimate the true effect and check how fragile it is.": ("A city wants congestion charging. Predict how traffic, shops, and low-income commuters would be affected, using evidence from London and Stockholm.", "travel_transport"),
    "My skilled teams adopted the new assistant first and chose their timing. Is our measured productivity gain causal?": ("Why does bread go stale faster in the fridge than on the counter? Explain starch retrogradation and how to slow it.", "food_fitness"),
    "Compare SSD and HDD storage for a database server, giving one workload that suits each.": ("A taxi charges $3 to start plus $1.75 per km. How far can I ride for $24?", "math"),
    "When does caching beat database indexing, and when is it the other way round? Give one trade-off each.": ("Three friends split a $96 dinner bill so one pays twice what each of the others pays. How much does each pay?", "math"),
    "REST or GraphQL for a mobile app that needs different fields on different screens: which suits it, and why?": ("A tank fills in 6 hours with pipe A and in 4 hours with pipe B. How long does it take with both open?", "math"),
    "A movie marathon lasts 2.4 hours in total. Convert that duration into minutes and show the calculation you used.": ("In a class poll, 12 students chose pizza, 8 chose pasta, and 5 chose salad. Which was most popular, and by how much?", "data_statistics"),
    "A running app reports a route of 3.5 kilometers. How many meters is that? Show the multiplication you used.": ("My fitness tracker shows 6,000, 8,000, and 10,000 steps on three days. What was my average daily step count?", "data_statistics"),
    "A shirt costs ₹800 and has a 15% discount. Calculate its sale price and savings.": ("What is the difference between a survey and a census? Give one example of each.", "data_statistics"),
    "Which is safer for an online account, a long passphrase or a short password with symbols?": ("What is jury duty, and what usually happens if someone ignores a summons?", "law_government"),
    "Create a weekly reminder plan to back up phone photos to a cloud folder.": ("Explain what a small claims court is for, using an example of a broken phone repair.", "law_government"),
}
REFERENCE_V3.update({
    "A taxi charges $3 to start plus $1.75 per km. How far can I ride for $24?": "12 km",
    "Three friends split a $96 dinner bill so one pays twice what each of the others pays. How much does each pay?": "$48, $24, and $24",
    "A tank fills in 6 hours with pipe A and in 4 hours with pipe B. How long does it take with both open?": "2.4 hours (2 hours 24 minutes)",
    "In a class poll, 12 students chose pizza, 8 chose pasta, and 5 chose salad. Which was most popular, and by how much?": "Pizza, by 4 votes over pasta",
    "My fitness tracker shows 6,000, 8,000, and 10,000 steps on three days. What was my average daily step count?": "8,000 steps",
})

# English only: the non-English questions from pass 3, rewritten in English (same tier and domain).
ENGLISH_ONLY = {
    "Ein Unternehmen will seine Server in die Cloud verlegen. Welche versteckten Kosten und Risiken sollte es vorher prüfen?": "A company plans to move all its servers to the cloud. What hidden costs and risks should it check before committing?",
    "बारिश के मौसम में मच्छरों से बचने के तीन तरीके बताइए।": "Give three ways to avoid mosquito bites during the rainy season when we sit outside in the evenings.",
    "Was ist der Unterschied zwischen einer Kreditkarte und einer Debitkarte, und wann ist welche sinnvoller?": "My credit card charges 24% APR, compounded monthly. If I carry a $2,000 balance for a year without paying, how much interest builds up?",
    "¿Por qué la inflación puede bajar el desempleo a corto plazo pero no a largo plazo? Explica la curva de Phillips.": "Why can inflation lower unemployment in the short run but not in the long run? Explain the Phillips curve debate.",
    "Pourquoi la glace flotte-t-elle sur l'eau alors que la plupart des solides coulent dans leur propre liquide ?": "Why does ice float on water while most solids sink in their own melted liquid?",
    "¿Por qué sube el precio de los vuelos cuanto más se acerca la fecha del viaje? Explica dos razones.": "Why do flight prices usually rise as the travel date gets closer? Give two reasons airlines do this.",
    "¿Cuál es la capital de Australia y en qué continente se encuentra?": "What is the capital of Australia, and which continent is it on? My son thinks it is Sydney.",
    "Quelle est la différence entre un fruit et un légume ? Donne un exemple de chaque.": "Is a tomato a fruit or a vegetable? Explain the difference with one example of each.",
    "डॉक्टर एक ही लक्षण वाले दो मरीजों को अलग दवाइयाँ क्यों दे सकते हैं? दवा-अंतःक्रिया और आनुवंशिकी के आधार पर समझाइए।": "Why might a doctor prescribe different drugs to two patients with identical symptoms? Explain using drug interactions and genetics.",
}
REFERENCE_V3[ENGLISH_ONLY["Was ist der Unterschied zwischen einer Kreditkarte und einer Debitkarte, und wann ist welche sinnvoller?"]] = "About $536 (2,000 x (1.02^12 - 1) = $536.48)"

# Broad domains, assigned from the question's content (old topic tags were stale after rewrites).
DOMAIN_RULES = [
    ("math", r"factorial|rooks|\bcoins\b|squares, including|x\^2|euler circuit|1 \+ 2 \+ 3|mean score|\bmedian\b|percentage of parents|per kilogram|movie marathon|kilometers\. how many meters|tip\. calculate|shirt costs"),
    ("data_statistics", r"survey|imputation|missing ages|average salary|typical salary|correlation|causal|causation|difference-in-differences|curriculum take credit|treatment records|in a trial|study reports|productivity gain"),
    ("ai_ml", r"classifier|precision|recall|predictor|benchmark|neural|chatbot|calibration|coding agent|router|supervised|accuracy"),
    ("software_coding", r"python|sql|javascript|regex|regular expression|shell command|spreadsheet formula|function that|\brust\b|compiler|lock-free|database|\bjoin\b|\bapi\b|rest |graphql|http|cache|caching|flask|containers|endpoints|\bcode\b|transaction|async|latency|\bdns\b|\bcdn\b|hourly reports|backups|operating system|\.csv|design pattern|spreadsheets disagree|cloud"),
    ("health_medicine", r"doctor|patient|antibiotic|fever|dehydration|vaccin|\bflu\b|allerg|warfarin|ibuprofen|grapefruit|hydroxylase|depression|withdrawn|handwashing|clinical|triage|hospital|celiac|microscope"),
    ("food_fitness", r"recipe|cake|sourdough|eggs|stir-fry|hummus|meals|calories|protein|vitamin|stretch|running 5 km|interval training|cardio|tea and coffee"),
    ("business_work", r"café|bakery|small shop|clothing shop|marketing|ad spending|campaign|delivery stops|on-call|freight|supplier|startup offer|pay bands|interview|resume|probation|elasticity|lemonade|family business|consultancy|farmers' market|customer-support agents|presentation slide|meetings keep"),
    ("money_finance", r"\btax|mortgage|loan|interest rate|invest|portfolio|insurance|deductible|income|budget|savings|bank|debit|credit|inflation|pocket money|bills are due|lease or buy|keep renting|notebooks|buy two"),
    ("law_government", r"landlord|tenant|sublet|contract|clause|good faith|legal|court|liable|copyright|mayor|voting|districts|gerrymander|gas stoves|tax hits|doctrine|will\?|driver's license"),
    ("home_family", r"toddler|kids|children|parent|sibling|\bteen|chore|houseplant|garden|tomato|vegetables|balcony|\btap\b|mold|cracks|house|puppy|\bdog\b|\bcat\b|shelter|morning routine|packing list|basement|ceiling fan|elderly mother|wiring"),
    ("science_environment", r"moon|planet|astronaut|exoplanet|seasons|water cycle|states of matter|ecosystem|erosion|thunderstorm|rain|heatwave|climate|lithium|battery|\bfarm|crop|groundwater|\bnile\b|aswan|solar|wind power|renewable|electricity|heater|kwh|recycling|shopping bags|ice"),
    ("travel_transport", r"flight|airport|flying|\btrip\b|travel|itinerary|rail|commute|\bbus\b|train|check-engine|brakes|fleet|tracking number|port strikes|wheelchair|city walk"),
    ("language_writing", r"translate|spanish|french|japanese|\bser\b|estar|email|message|summarize|rewrite|resume|toast|eye contact|slogan|names for|catchy|apology|briefing|narrator|cinderella|gatsby|spelling|lesson objective|op-ed|essay|thank-you card|cover letter|thanking"),
    ("culture_history", r"\bwar\b|constantinople|printing press|reformation|ocean|cities built|quartet|melody|songs|soccer|offside|chess|bishop|clubs|cold war|1929|bronze age|movie night|museum label"),
    ("security_tech", r"phishing|password|wi-fi|scam|caller|cryptography|screenshot|android|phone battery|laptop|encrypted|two-factor|phone photos"),
]


DOMAIN_OVERRIDE = {  # where the keyword rules can't place a question, or place it wrongly
    "Outline test cases for a login form covering valid input, wrong passwords, and a locked account.": "software_coding",
    "Compare 15-year and 30-year mortgages on $300,000 at 6%: monthly payment and total interest for each.": "money_finance",
    "An investment grew from $10,000 to $16,000 over five years. What was its average yearly growth rate?": "money_finance",
    "Write a fair screen-time agreement for siblings aged 8 and 13 who have different school schedules.": "home_family",
    "Thunderstorms are forecast for this afternoon. What should I pack for a morning day hike in the hills?": "science_environment",
    "Is it acceptable for doctors to photograph patients on personal phones, given privacy law and security controls?": "law_government",
    "How should scarce vaccines be split across regions with different infection rates and trust in officials?": "health_medicine",
    "Why do astronauts on the space station feel weightless when gravity there is still about 90% of Earth's?": "science_environment",
    "Our home Wi-Fi is fast near the router but weak upstairs. Give three practical fixes and their rough costs.": "security_tech",
    "A dataset has several missing ages. Should you fill them with the mean or the median for a skewed distribution?": "data_statistics",
    "My sister is opening a small bakery that sells sourdough and coffee. Suggest three names for it.": "business_work",
    "My daughter's school is buying lab equipment and she asked me: what is the main purpose of a microscope in science?": "science_environment",
    "For a science worksheet for young children, please name the three states of matter that are commonly taught in school.": "science_environment",
    "In two sentences that a worried parent could follow easily, summarize what a vaccine is for and how it helps.": "health_medicine",
    "After a 30-minute evening dog walk, which three stretches help the legs recover?": "food_fitness",
    "My grandmother has a new Android phone. How does she take a screenshot of a message?": "security_tech",
    "An odd email asks the reader to confirm a password. List warning signs that it is phishing.": "security_tech",
    "Value an office building whose main tenant may leave in two years while remote work keeps rising.": "money_finance",
    "Translate an English liability clause into Japanese while keeping both its legal force and its politeness level.": "language_writing",
    "How do you write a public apology after a customer data leak that admits fault but limits legal exposure?": "language_writing",
    "Plan a five-year farm rotation that rebuilds soil, controls pests with no sprays, and keeps income steady.": "science_environment",
    "In a household budget spreadsheet, which formula adds up the values in cells A1 to A10?": "software_coding",
    "My brakes usually squeal only on mornings after rain and stop after a few minutes. What is happening?": "travel_transport",
    "My phone battery drains fast even when idle. List likely causes and how to check each in settings.": "security_tech",
    "Last month my laptop died and I lost some photos. Explain why regular backups of personal data are useful.": "security_tech",
    "A 'bank' caller asks me to read out a code they texted me. What is happening, and what should I do?": "security_tech",
    "My grandmother asked me this while setting up her new laptop: what is the purpose of an operating system on a computer?": "security_tech",
    "Why might our two sales spreadsheets disagree by 3% after currency conversion, refunds, and time-zone cutoffs?": "business_work",
    "Before a trip to Paris next week, how would you translate 'The train leaves at six o'clock' into French?": "language_writing",
    "An online order email included a tracking number. What is it for, and how is it used?": "business_work",
    "Tasks A takes 2 days; B and C follow A for 4 and 3 days. D takes 2 days after both. Find duration.": "math",
    "Make long-running jobs that call outside APIs survive rate limits, partial failures, and side effects that cannot repeat.": "software_coding",
    "Which differential-privacy parameters suit analytics on a small dataset, and how do you avoid misleading guarantees?": "data_statistics",
    "A CSV contains duplicate customer IDs with different timestamps. Keep only the latest record for each, safely.": "software_coding",
    "A safety problem is so rare that a normal A/B experiment would need millions of users. Evaluate a fix anyway.": "data_statistics",
    "A webpage is slow while server CPU remains low. Give two plausible causes and checks for each.": "software_coding",
    "Someone earned $8,000 freelancing and $50,000 in salary. Which records should they keep for filing, and why?": "money_finance",
    "My cousin is moving to a new city. What is the difference between renting and leasing an apartment?": "money_finance",
    "Why should bleach never be mixed with ammonia-based cleaners when cleaning the bathroom at home?": "home_family",
    "Why does baking soda absorb fridge smells, and when would vinegar work better for cleaning?": "home_family",
}


def domain_of(q):
    import re
    if q in DOMAIN_OVERRIDE:
        return DOMAIN_OVERRIDE[q]
    ql = q.lower()
    for d, pat in DOMAIN_RULES:
        if re.search(rf"(?<![a-z])(?:{pat})(?![a-z])", ql):  # whole words only: "train" is not "rain"
            return d
    raise ValueError(f"no domain for: {q}")


def main():
    old = make_150_balanced.build()
    new = []
    for topic, qs in NEW.items():
        assert len(qs) == 3, topic
        for k, (label, q) in enumerate(zip(("LOW", "MID", "HIGH"), qs)):
            q = FIX.get((topic, k), q)
            new.append({"query": q, "label": label, "topic": topic, "word_count": len(q.split())})
    seen = {r["query"].strip().lower() for r in old}
    dup = [r["query"] for r in new if r["query"].strip().lower() in seen]
    assert not dup, dup
    assert all(k[0] in NEW and 0 <= k[1] < 3 for k in FIX), "FIX names an unknown topic or tier"
    rows = old + new
    present = {r["query"] for r in rows}
    present |= {v[0] for v in REPLACE_V3.values()}
    missing = [q for q in list(REPLACE_V3) + list(REPLACE_V3B) if q not in present]
    assert not missing, f"pass 3 names questions that are not in the set: {missing}"
    out = []
    for r in rows:
        if r["query"] in REPLACE_V3:  # pass 3: a new question in the same tier and its own domain
            q, dom = REPLACE_V3[r["query"]]
            r = {"query": q, "label": r["label"], "topic": dom}
            if q in REFERENCE_V3:
                r["reference_answer"] = REFERENCE_V3[q]
        else:
            r = dict(r, topic=domain_of(r["query"]))
        if r["query"] in REPLACE_V3B:  # pass 3b, applied on top of pass 3
            q, dom = REPLACE_V3B[r["query"]]
            r = {"query": q, "label": r["label"], "topic": dom}
            if q in REFERENCE_V3:
                r["reference_answer"] = REFERENCE_V3[q]
        if r["query"] in ENGLISH_ONLY:  # English only, same tier and domain
            q = ENGLISH_ONLY[r["query"]]
            r = {"query": q, "label": r["label"], "topic": r["topic"]}
            if q in REFERENCE_V3:
                r["reference_answer"] = REFERENCE_V3[q]
        r["word_count"] = len(r["query"].split())
        out.append(r)
    assert len({r["query"].strip().lower() for r in out}) == len(out), "duplicate question"
    random.Random(42).shuffle(out)
    out = [{k: r[k] for k in FIELDS if k in r} for r in out]  # drop bookkeeping (edit, original_query)
    DST.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    print(f"wrote {DST.name}: {len(out)} rows ({len(REPLACE_V3)} replaced in pass 3, {len(REPLACE_V3B)} in pass 3b; topics are now broad domains)")


if __name__ == "__main__":
    main()
