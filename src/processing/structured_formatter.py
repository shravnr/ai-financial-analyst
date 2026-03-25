import logging

logger = logging.getLogger(__name__)


def _fmt_currency(value, in_millions: bool = True) -> str:
    if value is None or value == "":
        return "N/A"
    try:
        v = float(value)
    except (ValueError, TypeError):
        return str(value)

    abs_v = abs(v)
    sign = "-" if v < 0 else ""

    if abs_v >= 1e12:
        return f"{sign}${abs_v / 1e12:,.2f} trillion"
    elif abs_v >= 1e9:
        return f"{sign}${abs_v / 1e9:,.2f} billion"
    elif abs_v >= 1e6:
        return f"{sign}${abs_v / 1e6:,.1f} million"
    elif abs_v >= 1e3:
        return f"{sign}${abs_v / 1e3:,.1f} thousand"
    else:
        return f"{sign}${v:,.2f}"


def _fmt_pct(value) -> str:
    if value is None or value == "":
        return "N/A"
    try:
        return f"{float(value) * 100:.2f}%" if abs(float(value)) < 1 else f"{float(value):.2f}%"
    except (ValueError, TypeError):
        return str(value)


def _fmt_number(value) -> str:
    if value is None or value == "":
        return "N/A"
    try:
        v = float(value)
        if v == int(v) and abs(v) < 1e15:
            return f"{int(v):,}"
        return f"{v:,.2f}"
    except (ValueError, TypeError):
        return str(value)



def _format_profile(data: dict, ticker: str) -> list[dict]:
    if not data:
        return []

    record = data[0] if isinstance(data, list) else data

    text = f"""{record.get('companyName', ticker)} ({ticker}) — Company Profile

Sector: {record.get('sector', 'N/A')}
Industry: {record.get('industry', 'N/A')}
Exchange: {record.get('exchangeShortName', 'N/A')}
Market Capitalization: {_fmt_currency(record.get('marketCap', record.get('mktCap')))}
Current Price: ${record.get('price', 'N/A')}
52-Week Range: {record.get('range', 'N/A')}
Beta: {_fmt_number(record.get('beta'))}
Average Volume: {_fmt_number(record.get('volAvg', record.get('averageVolume')))}
Dividend Yield: {_fmt_pct(record.get('lastDividend'))}
Country: {record.get('country', 'N/A')}
CEO: {record.get('ceo', 'N/A')}
Employees: {_fmt_number(record.get('fullTimeEmployees'))}
Website: {record.get('website', 'N/A')}
Description: {record.get('description', 'N/A')}"""

    return [{
        "text": text,
        "metadata": {
            "ticker": ticker,
            "company_name": record.get("companyName", ticker),
            "source_type": "fmp",
            "document_type": "company_profile",
            "date": "current",
            "section": "Company Profile",
            "chunk_index": 0,
        },
    }]


def _compute_ratio(numerator, denominator):
    """Compute ratio from raw values when pre-computed ratio is missing."""
    if numerator is None or denominator is None:
        return None
    try:
        num, den = float(numerator), float(denominator)
        if den == 0:
            return None
        return num / den
    except (ValueError, TypeError):
        return None


def _format_income_statement(records: list[dict], ticker: str, period: str) -> list[dict]:
    results = []
    for i, r in enumerate(records or []):
        date = r.get("date", r.get("fillingDate", "Unknown"))
        fiscal_year = r.get("fiscalYear") or r.get("calendarYear") or ""

        gross_margin = r.get('grossProfitRatio') or _compute_ratio(r.get('grossProfit'), r.get('revenue'))
        operating_margin = r.get('operatingIncomeRatio') or _compute_ratio(r.get('operatingIncome'), r.get('revenue'))
        net_margin = r.get('netIncomeRatio') or _compute_ratio(r.get('netIncome'), r.get('revenue'))

        text = f"""{ticker} — Income Statement (FY {fiscal_year}, ending {date})

Revenue: {_fmt_currency(r.get('revenue'))}
Cost of Revenue: {_fmt_currency(r.get('costOfRevenue'))}
Gross Profit: {_fmt_currency(r.get('grossProfit'))}
Gross Margin: {_fmt_pct(gross_margin)}

Operating Expenses: {_fmt_currency(r.get('operatingExpenses'))}
Research & Development: {_fmt_currency(r.get('researchAndDevelopmentExpenses'))}
Selling, General & Admin: {_fmt_currency(r.get('sellingGeneralAndAdministrativeExpenses'))}
Operating Income: {_fmt_currency(r.get('operatingIncome'))}
Operating Margin: {_fmt_pct(operating_margin)}

Interest Expense: {_fmt_currency(r.get('interestExpense'))}
Income Before Tax: {_fmt_currency(r.get('incomeBeforeTax'))}
Income Tax Expense: {_fmt_currency(r.get('incomeTaxExpense'))}
Net Income: {_fmt_currency(r.get('netIncome'))}
Net Margin: {_fmt_pct(net_margin)}

EPS (Basic): ${_fmt_number(r.get('eps'))}
EPS (Diluted): ${_fmt_number(r.get('epsdiluted'))}
Weighted Avg Shares (Diluted): {_fmt_number(r.get('weightedAverageShsOutDil'))}
EBITDA: {_fmt_currency(r.get('ebitda'))}"""

        results.append({
            "text": text,
            "metadata": {
                "ticker": ticker,
                "source_type": "fmp",
                "document_type": "income_statement",
                "date": date,
                "section": f"Income Statement - FY {fiscal_year}",
                "period": period,
                "chunk_index": i,
            },
        })
    return results


def _format_balance_sheet(records: list[dict], ticker: str, period: str) -> list[dict]:
    results = []
    for i, r in enumerate(records or []):
        date = r.get("date", "Unknown")
        fiscal_year = r.get("fiscalYear") or r.get("calendarYear") or ""
        text = f"""{ticker} — Balance Sheet (FY {fiscal_year}, ending {date})

ASSETS
Total Assets: {_fmt_currency(r.get('totalAssets'))}
Cash & Equivalents: {_fmt_currency(r.get('cashAndCashEquivalents'))}
Short-Term Investments: {_fmt_currency(r.get('shortTermInvestments'))}
Cash & Short-Term Investments: {_fmt_currency(r.get('cashAndShortTermInvestments'))}
Net Receivables: {_fmt_currency(r.get('netReceivables'))}
Inventory: {_fmt_currency(r.get('inventory'))}
Total Current Assets: {_fmt_currency(r.get('totalCurrentAssets'))}
Property, Plant & Equipment (Net): {_fmt_currency(r.get('propertyPlantEquipmentNet'))}
Goodwill: {_fmt_currency(r.get('goodwill'))}
Intangible Assets: {_fmt_currency(r.get('intangibleAssets'))}
Long-Term Investments: {_fmt_currency(r.get('longTermInvestments'))}
Total Non-Current Assets: {_fmt_currency(r.get('totalNonCurrentAssets'))}

LIABILITIES
Total Liabilities: {_fmt_currency(r.get('totalLiabilities'))}
Accounts Payable: {_fmt_currency(r.get('accountPayables'))}
Short-Term Debt: {_fmt_currency(r.get('shortTermDebt'))}
Total Current Liabilities: {_fmt_currency(r.get('totalCurrentLiabilities'))}
Long-Term Debt: {_fmt_currency(r.get('longTermDebt'))}
Total Non-Current Liabilities: {_fmt_currency(r.get('totalNonCurrentLiabilities'))}
Total Debt: {_fmt_currency(r.get('totalDebt'))}
Net Debt: {_fmt_currency(r.get('netDebt'))}

EQUITY
Total Stockholders' Equity: {_fmt_currency(r.get('totalStockholdersEquity'))}
Retained Earnings: {_fmt_currency(r.get('retainedEarnings'))}
Total Equity: {_fmt_currency(r.get('totalEquity'))}"""

        results.append({
            "text": text,
            "metadata": {
                "ticker": ticker,
                "source_type": "fmp",
                "document_type": "balance_sheet",
                "date": date,
                "section": f"Balance Sheet - FY {fiscal_year}",
                "period": period,
                "chunk_index": i,
            },
        })
    return results


def _format_cash_flow(records: list[dict], ticker: str, period: str) -> list[dict]:
    results = []
    for i, r in enumerate(records or []):
        date = r.get("date", "Unknown")
        fiscal_year = r.get("fiscalYear") or r.get("calendarYear") or ""
        text = f"""{ticker} — Cash Flow Statement (FY {fiscal_year}, ending {date})

OPERATING ACTIVITIES
Net Income: {_fmt_currency(r.get('netIncome'))}
Depreciation & Amortization: {_fmt_currency(r.get('depreciationAndAmortization'))}
Stock-Based Compensation: {_fmt_currency(r.get('stockBasedCompensation'))}
Change in Working Capital: {_fmt_currency(r.get('changeInWorkingCapital'))}
Operating Cash Flow: {_fmt_currency(r.get('operatingCashFlow'))}

INVESTING ACTIVITIES
Capital Expenditures: {_fmt_currency(r.get('capitalExpenditure'))}
Acquisitions: {_fmt_currency(r.get('acquisitionsNet'))}
Purchases of Investments: {_fmt_currency(r.get('purchasesOfInvestments'))}
Sales/Maturities of Investments: {_fmt_currency(r.get('salesMaturitiesOfInvestments'))}
Investing Cash Flow: {_fmt_currency(r.get('netCashUsedForInvestingActivites'))}

FINANCING ACTIVITIES
Debt Repayment: {_fmt_currency(r.get('debtRepayment'))}
Common Stock Repurchased: {_fmt_currency(r.get('commonStockRepurchased'))}
Dividends Paid: {_fmt_currency(r.get('dividendsPaid'))}
Financing Cash Flow: {_fmt_currency(r.get('netCashUsedProvidedByFinancingActivities'))}

Free Cash Flow: {_fmt_currency(r.get('freeCashFlow'))}
Net Change in Cash: {_fmt_currency(r.get('netChangeInCash'))}"""

        results.append({
            "text": text,
            "metadata": {
                "ticker": ticker,
                "source_type": "fmp",
                "document_type": "cash_flow_statement",
                "date": date,
                "section": f"Cash Flow Statement - FY {fiscal_year}",
                "period": period,
                "chunk_index": i,
            },
        })
    return results


def _format_key_metrics(records: list[dict], ticker: str) -> list[dict]:
    results = []
    for i, r in enumerate(records or []):
        date = r.get("date", "Unknown")
        period_label = r.get("period", "annual")

        text = f"""{ticker} — Key Financial Metrics ({period_label}, {date})

Valuation
Market Cap: {_fmt_currency(r.get('marketCap'))}
Enterprise Value: {_fmt_currency(r.get('enterpriseValue'))}
P/E Ratio: {_fmt_number(r.get('peRatio'))}
Price-to-Sales: {_fmt_number(r.get('priceToSalesRatio'))}
Price-to-Book: {_fmt_number(r.get('pbRatio'))}
EV/EBITDA: {_fmt_number(r.get('enterpriseValueOverEBITDA'))}
EV/Revenue: {_fmt_number(r.get('evToSales'))}

Per Share
Revenue Per Share: ${_fmt_number(r.get('revenuePerShare'))}
Net Income Per Share: ${_fmt_number(r.get('netIncomePerShare'))}
Book Value Per Share: ${_fmt_number(r.get('bookValuePerShare'))}
Free Cash Flow Per Share: ${_fmt_number(r.get('freeCashFlowPerShare'))}
Dividends Per Share: ${_fmt_number(r.get('dividendPerShare'))}

Returns & Margins
ROE: {_fmt_pct(r.get('roe'))}
ROA: {_fmt_pct(r.get('returnOnTangibleAssets'))}
ROIC: {_fmt_pct(r.get('roic'))}
Earnings Yield: {_fmt_pct(r.get('earningsYield'))}
Dividend Yield: {_fmt_pct(r.get('dividendYield'))}

Debt
Debt-to-Equity: {_fmt_number(r.get('debtToEquity'))}
Debt-to-Assets: {_fmt_number(r.get('debtToAssets'))}
Net Debt-to-EBITDA: {_fmt_number(r.get('netDebtToEBITDA'))}
Current Ratio: {_fmt_number(r.get('currentRatio'))}
Interest Coverage: {_fmt_number(r.get('interestCoverage'))}"""

        results.append({
            "text": text,
            "metadata": {
                "ticker": ticker,
                "source_type": "fmp",
                "document_type": "key_metrics",
                "date": date,
                "section": f"Key Metrics - {period_label} {date}",
                "chunk_index": i,
            },
        })
    return results


def _format_grades(records: list[dict], ticker: str) -> list[dict]:
    if not records:
        return []

    # ~15 grades per chunk (~6 months of activity)
    chunk_size = 15
    results = []

    for chunk_idx in range(0, min(len(records), 60), chunk_size):  # cap at 60 grades
        batch = records[chunk_idx:chunk_idx + chunk_size]

        lines = [f"{ticker} — Analyst Grades and Rating Changes\n"]
        for g in batch:
            date = g.get("date", "Unknown")[:10]
            company = g.get("gradingCompany", "Unknown")
            prev = g.get("previousGrade", "N/A")
            new = g.get("newGrade", "N/A")
            action = g.get("action", "")
            lines.append(f"{date}: {company} — {action} from {prev} to {new}")

        date_range_start = batch[-1].get("date", "")[:10]
        date_range_end = batch[0].get("date", "")[:10]

        results.append({
            "text": "\n".join(lines),
            "metadata": {
                "ticker": ticker,
                "source_type": "fmp",
                "document_type": "analyst_grades",
                "date": date_range_end,
                "section": f"Analyst Grades ({date_range_start} to {date_range_end})",
                "chunk_index": chunk_idx // chunk_size,
            },
        })

    logger.info(f"Formatted {len(results)} analyst grade chunks for {ticker}")
    return results


def _format_analyst_estimates(records: list[dict], ticker: str) -> list[dict]:
    results = []
    for i, r in enumerate(records or []):
        date = r.get("date", "Unknown")

        text = f"""{ticker} — Analyst Estimates (for period ending {date})

Revenue Estimates
  Low: {_fmt_currency(r.get('revenueLow'))}
  Average: {_fmt_currency(r.get('revenueAvg'))}
  High: {_fmt_currency(r.get('revenueHigh'))}

EBITDA Estimates
  Low: {_fmt_currency(r.get('ebitdaLow'))}
  Average: {_fmt_currency(r.get('ebitdaAvg'))}
  High: {_fmt_currency(r.get('ebitdaHigh'))}

EPS Estimates
  Low: ${_fmt_number(r.get('epsLow'))}
  Average: ${_fmt_number(r.get('epsAvg'))}
  High: ${_fmt_number(r.get('epsHigh'))}

Net Income Estimates
  Low: {_fmt_currency(r.get('netIncomeLow'))}
  Average: {_fmt_currency(r.get('netIncomeAvg'))}
  High: {_fmt_currency(r.get('netIncomeHigh'))}

Number of Analysts: {_fmt_number(r.get('numberAnalystsEstimatedRevenue'))}"""

        results.append({
            "text": text,
            "metadata": {
                "ticker": ticker,
                "source_type": "fmp",
                "document_type": "analyst_estimates",
                "date": date,
                "section": f"Analyst Estimates - {date}",
                "chunk_index": i,
            },
        })
    return results


