# ── Deep eval ────────────────────────────────────────────────────────
# All questions use {company} placeholder — resolved at eval time.

# Consistency pairs — same question asked two ways.
# Format: {"name", "questions": [q_a, q_b], "expected_sources"}
CONSISTENCY_PAIRS: list[dict] = [
    {
        "name": "Revenue",
        "questions": [
            "What are {company}'s revenue and profit trends over the past two years?",
            "How much revenue and net income has {company} generated recently?",
        ],
        "expected_sources": ["fmp"],
    },
    {
        "name": "Risks",
        "questions": [
            "What are the biggest risks to {company}'s business according to their 10-K?",
            "What risk factors does {company} disclose in SEC filings?",
        ],
        "expected_sources": ["sec"],
    },
    {
        "name": "Cash position",
        "questions": [
            "What is {company}'s current cash position and is it generating or burning cash?",
            "How much cash does {company} hold and what is its free cash flow?",
        ],
        "expected_sources": ["fmp"],
    },
    {
        "name": "News",
        "questions": [
            "What is the latest news about {company}?",
            "Are there any recent headlines or developments about {company}?",
        ],
        "expected_sources": ["news"],
    },
]

# Grounding questions — 10 curated questions covering SEC, FMP, News.
# No ground truth — grounding judge evaluates answer vs retrieved context (correct for live data).
# Format: (capability, question_template, expected_sources)
GROUNDING_QUESTIONS: list[tuple[str, str, list[str]]] = [
    (
        "Revenue trends",
        "What are {company}'s revenue and profit trends over the past two years?",
        ["fmp"],
    ),
    (
        "Gross margin",
        "Has {company}'s gross margin improved or declined recently?",
        ["fmp"],
    ),
    (
        "Biggest risks (10-K)",
        "What are the biggest risks to {company}'s business according to their 10-K?",
        ["sec"],
    ),
    (
        "Earnings announcement",
        "Summarize {company}'s most recent earnings announcement.",
        ["sec"],
    ),
    (
        "YoY revenue comparison",
        "How does {company}'s most recent fiscal year revenue compare to the prior year?",
        ["fmp"],
    ),
    (
        "Latest news",
        "What is the latest news about {company}?",
        ["news"],
    ),
    (
        "Business segments",
        "What are {company}'s main business segments and revenue drivers?",
        ["sec"],
    ),
    (
        "Analyst EPS estimates",
        "What are analyst EPS estimates for {company}?",
        ["fmp"],
    ),
    (
        "R&D spending",
        "How much does {company} invest in research and development?",
        ["fmp"],
    ),
    (
        "Debt situation",
        "What is {company}'s total debt and how much is long-term?",
        ["fmp"],
    ),
]


# ── Smoke test ───────────────────────────────────────────────────────

# Normal questions — 3 per indexed ticker, rule-based only.
# Format: (capability, question_template, expected_sources)
SMOKE_QUESTIONS: list[tuple[str, str, list[str]]] = [
    (
        "Revenue",
        "What are {ticker}'s revenue trends over the past two years?",
        ["fmp"],
    ),
    (
        "Risks",
        "What are the biggest risks to {ticker}'s business?",
        ["sec"],
    ),
    (
        "News",
        "What is the latest news about {ticker}?",
        ["news"],
    ),
]

# Boundary refusals — must refuse without hallucinating a number.
# Format: (name, question)
BOUNDARY_REFUSALS: list[tuple[str, str]] = [
    ("Private company", "What is Stripe's revenue?"),
    ("Unknown company", "What is Cred's financial performance?"),
    ("Speculative", "What will Apple's stock price be next month?"),
]
