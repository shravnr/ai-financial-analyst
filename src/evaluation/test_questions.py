# Format: (capability_name, question, ground_truth_reference)
AAPL_TEST_QUESTIONS: list[tuple[str, str, str]] = [
    #  1. Revenue and profit trends 
    (
        "1. Revenue and profit trends",
        "What are Apple's revenue and profit trends over the past two years?",
        "Apple's revenue was approximately $391B in FY2024 and $394B in FY2025. Net income was approximately $94B in both years."
    ),
    (
        "1. Revenue and profit trends",
        "Has Apple's gross margin improved or declined recently?",
        "Apple's gross margin has been approximately 46-47%, with slight improvement."
    ),

    #  2. Management guidance 
    (
        "2. Management guidance",
        "What are analyst revenue estimates for Apple's next fiscal year?",
        "Analyst estimates project Apple's FY2026 revenue around $410-420 billion."
    ),

    #  3. Biggest risks 
    (
        "3. Biggest risks",
        "What are the biggest risks to Apple's business according to their 10-K?",
        "Key risks include competition, supply chain dependence, international exposure, regulatory proceedings, and macroeconomic conditions."
    ),

    #  4. Cash position and cash flow 
    (
        "4. Cash position and cash flow",
        "What is Apple's current cash position and is it generating or burning cash?",
        "Apple holds approximately $30-36B in cash, with over $100B in annual operating cash flow."
    ),
    (
        "4. Cash position and cash flow",
        "What is Apple's free cash flow?",
        "Apple's free cash flow is approximately $100-110 billion annually."
    ),

    #  5. Earnings call summary 
    (
        "5. Earnings call summary",
        "Summarize Apple's most recent earnings announcement.",
        "Apple's most recent earnings were announced via 8-K in January 2026 for Q1 FY2026."
    ),

    #  6. Year over year comparison 
    (
        "6. Year over year comparison",
        "How does Apple's FY2025 revenue compare to FY2024?",
        "FY2025 revenue ~$394B vs FY2024 ~$391B, roughly 1% growth."
    ),

    #  7. Debt situation ─
    (
        "7. Debt situation",
        "What is Apple's total debt and how much is long-term?",
        "Total debt approximately $96-100B, long-term debt approximately $85-97B."
    ),

    #  8. Latest news 
    (
        "8. Latest news",
        "What is the latest news about Apple?",
        "Recent news includes product announcements, quarterly earnings, and regulatory developments."
    ),

    #  9. Business segments 
    (
        "9. Business segments",
        "What are Apple's main business segments and revenue drivers?",
        "iPhone, Mac, iPad, Wearables/Home/Accessories, and Services. iPhone is the largest."
    ),

    #  10. Management strategy ─
    (
        "10. Management strategy",
        "How much does Apple invest in research and development?",
        "Apple's R&D spending is approximately $30-31 billion annually."
    ),
    (
        "10. Management strategy",
        "What is Apple's capital return program?",
        "Massive capital return including share repurchases and dividends, returning over $90B annually."
    ),

    #  11. Competitor comparison ─
    (
        "11. Competitor comparison",
        "What does Apple say about its competitive position in SEC filings?",
        "Apple's 10-K discloses intense competition across all product categories."
    ),

    #  12. Analyst ratings ─
    (
        "12. Analyst ratings",
        "What are analyst EPS estimates for Apple?",
        "Analyst EPS estimates are available from FMP data."
    ),

    #  13. Valuation ─
    (
        "13. Valuation",
        "What is Apple's current P/E ratio and EV/EBITDA?",
        "P/E and EV/EBITDA available from FMP key metrics."
    ),
    (
        "13. Valuation",
        "Is Apple overvalued or undervalued based on financial metrics?",
        "Requires comparing multiples to historical averages and peers."
    ),

    #  14. Industry-specific metrics ─
    (
        "14. Industry-specific metrics",
        "What are Apple's key metrics like R&D spending, margins, and ROE?",
        "R&D ~$30-31B, gross margin ~46-47%, operating margin ~30-34%, high ROE."
    ),
    (
        "14. Industry-specific metrics",
        "How does Apple's operating margin compare to its gross margin?",
        "Gross margin ~46-47%, operating margin ~30-34%. Gap is R&D and SGA expenses."
    ),

    #  Cross-cutting: refusal behavior ─
    (
        "Cross-cutting",
        "What are recent analyst ratings and grade changes for Apple stock?",
        "Multiple analysts have issued ratings. FMP grades data includes recent upgrades and downgrades."
    ),
]
