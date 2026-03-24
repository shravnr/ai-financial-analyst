import json
import logging
import re

from openai import OpenAI

from src.config import OPENAI_API_KEY, LLM_MODEL
from src.processing.vector_store import query as vector_query
from src.rag.query_router import route_query
from src.guardrails.validator import validate_response

logger = logging.getLogger(__name__)

_client = OpenAI(api_key=OPENAI_API_KEY)

MAX_TOOL_ITERATIONS = 3

# Tool definitions for function calling

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_sec_filings",
            "description": "Search SEC filings (10-K annual reports, 10-Q quarterly reports, 8-K earnings press releases) for a company. Best for: risks, strategy, business segments, legal disclosures, management discussion, qualitative analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query describing what information to find in SEC filings",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of relevant chunks to retrieve (default 8)",
                        "default": 8,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_financial_data",
            "description": "Search structured financial data (income statements, balance sheets, cash flow statements, key metrics, ratios, analyst grades, analyst estimates). Best for: specific numbers, financial trends, ratios, valuation metrics, analyst ratings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query describing what financial data to find",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of relevant chunks to retrieve (default 8)",
                        "default": 8,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": "Search recent news articles about a company. Best for: latest developments, current events, market sentiment, recent announcements.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query describing what news to find",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of relevant chunks to retrieve (default 6)",
                        "default": 6,
                    },
                },
                "required": ["query"],
            },
        },
    },
]

_TOOL_SOURCE_MAP = {
    "search_sec_filings": ["sec"],
    "search_financial_data": ["fmp"],
    "search_news": ["news"],
}

SYSTEM_PROMPT = """You are a professional, straightforward financial analyst. Direct, specific, and concise.

## RULES

1. Call tools to retrieve relevant data. Use multiple tools when the question spans different data types (financials, SEC filings, news).
2. Answer using ONLY the data your tools returned. Every claim — numbers, facts, events, qualitative statements — must have an inline [N] citation matching the exact source it came from.
3. Derived metrics (margins, growth rates, ratios, differences) computed from returned data are encouraged — that is analysis, not fabrication.
4. If tools return data relevant to the question, USE it to build the best answer you can — even if the match isn't exact. For example, an earnings summary can be built from 8-K filings and income statement data.
5. Never fabricate data or cite a source for data it does not contain. If the tools returned nothing useful, say so in one sentence.

## CITATIONS

Place [N] inline immediately after each claim. N = the Source number shown in tool results (e.g., [Source 3] → cite as [3]).

Critical rules:
- Cite the SPECIFIC source containing the data point. If a number appears in Source 3, cite [3] — not [1].
- Every sentence with a factual claim needs at least one citation. This includes news headlines, qualitative facts, and events — not just numbers.
- Do NOT renumber sources. Do NOT write a footnotes or references section — the system generates that automatically.

Example:
  Revenue grew 12% to $716.9B[1], up from $638.0B[2]. Operating margin expanded to 11.2%[1].

## DATA COVERAGE

Your tools search data that has been pre-loaded for the company:
- **SEC filings**: 10-K (annual), 10-Q (quarterly), 8-K filings from the last ~2 years
- **Financial data**: Annual income statements, balance sheets, cash flow, key metrics, and ratios (last ~2 years). Quarterly breakdowns are limited — use SEC 10-Q filings for quarterly detail.
- **News**: Articles from the last ~30 days only.

If a question falls outside this range, say so directly instead of searching.

**CRITICAL**: You have ZERO general knowledge. Every single number, fact, date, percentage, event, risk factor, product name, and claim in your answer MUST come from the tool results returned to you. If a piece of information is not explicitly present in the tool results, do NOT include it — even if you "know" it from training. Treat your training data as nonexistent. If the tool results are thin, give a shorter answer — never pad with outside knowledge.

## STYLE

- Lead with the answer. Be direct and specific ("$108.8B in FY 2024", not "approximately $109B").
- When comparing periods, always use the most recent data available and work backwards.
- Analyze trends, comparisons, and implications across cited data points.
- Be concise. No caveats, no padding, no hedging."""


def _format_tool_results(
    chunks: list[dict],
    source_counter: int,
    seen_sources: dict,
    source_metadata: dict,
) -> tuple[str, int]:
    parts = []
    for chunk in chunks:
        meta = chunk["metadata"]
        ticker_str = meta.get("ticker", "")
        doc_type = meta.get("document_type", "unknown").replace("_", " ").title()
        date = meta.get("date", "")
        section = meta.get("section", "")

        section_key = section.split(">")[0].strip() if section else ""
        key = f"{ticker_str}_{doc_type}_{date}_{section_key}"
        if key in seen_sources:
            num = seen_sources[key]
        else:
            source_counter += 1
            num = source_counter
            seen_sources[key] = num
            source_metadata[num] = {
                "source_type": meta.get("source_type", ""),
                "document_type": meta.get("document_type", ""),
                "ticker": ticker_str,
                "company_name": meta.get("company_name", ""),
                "date": date,
                "section": section,
                "file_path": meta.get("file_path", ""),
                "edgar_url": meta.get("edgar_url", ""),
                "url": meta.get("url", ""),
            }

        source_type = meta.get("source_type", "").upper()
        header = f"{source_type} — {ticker_str} {doc_type}" if source_type else f"{ticker_str} {doc_type}"
        if date:
            header += f" ({date})"
        if section:
            header += f", {section}"

        parts.append(f"[Source {num}] {header}\n{chunk['text']}")

    return "\n\n---\n\n".join(parts), source_counter


def _execute_tool(
    tool_name: str,
    args: dict,
    ticker: str,
    source_counter: int,
    seen_sources: dict,
    source_metadata: dict,
) -> tuple[str, list[dict], int]:
    query = args.get("query", "")
    n_results = max(args.get("n_results", 8), 5)  # Minimum 5 to avoid thin context
    source_types = _TOOL_SOURCE_MAP.get(tool_name)

    chunks = vector_query(
        query_text=query,
        ticker=ticker,
        source_types=source_types,
        n_results=n_results,
    )

    if not chunks:
        logger.warning(
            f"Tool {tool_name}(query='{query[:50]}', n={n_results}) → 0 chunks"
        )
        return (
            "No results found for this query. Try broadening your search "
            "terms or using a different tool.",
            [],
            source_counter,
        )

    formatted, source_counter = _format_tool_results(
        chunks, source_counter, seen_sources, source_metadata
    )

    logger.info(
        f"Tool {tool_name}(query='{query[:50]}', n={n_results}) → "
        f"{len(chunks)} chunks"
    )

    return formatted, chunks, source_counter


def _fallback_retrieve(
    question: str, ticker: str
) -> tuple[str, list[dict], dict]:
    routing = route_query(question)

    chunks = vector_query(
        query_text=question,
        ticker=ticker,
        source_types=routing["source_types"],
        n_results=routing["n_results"],
    )

    if len(chunks) < 3 and routing["source_types"] is not None:
        chunks = vector_query(
            query_text=question,
            ticker=ticker,
            source_types=None,
            n_results=routing["n_results"],
        )

    source_metadata = {}
    formatted, _ = _format_tool_results(
        chunks, 0, seen_sources={}, source_metadata=source_metadata
    )
    return formatted, chunks, source_metadata


# Strip LLM-generated footnotes, rebuild from actual chunk metadata
def _strip_footnotes(answer: str) -> str:
    lines = answer.rstrip().split("\n")

    cut = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        s = lines[i].strip()
        if re.match(r"^\[\d+\]", s):
            cut = i
        elif s == "" or s == "---":
            continue
        else:
            break

    body = "\n".join(lines[:cut]).rstrip()
    if body.endswith("---"):
        body = body[:-3].rstrip()

    return body


def _format_footnote(num: int, meta: dict) -> str:
    source_type = meta.get("source_type", "").upper()
    ticker = meta.get("ticker", "")
    doc_type = meta.get("document_type", "").replace("_", " ").title()
    date = meta.get("date", "")
    section = meta.get("section", "")
    edgar_url = meta.get("edgar_url", "")
    url = meta.get("url", "")

    desc = f"{source_type} — {ticker} {doc_type}"
    if date:
        desc += f" ({date})"
    if section:
        desc += f", {section}"

    ref = edgar_url or url
    if ref:
        desc += f" → {ref}"

    return f"- [{num}] {desc}"


def _postprocess_citations(answer: str, source_metadata: dict) -> str:
    if not source_metadata:
        return answer

    body = _strip_footnotes(answer)

    refs_in_order = []
    seen = set()
    for m in re.finditer(r"\[(\d+)\]", body):
        num = int(m.group(1))
        if num not in seen:
            refs_in_order.append(num)
            seen.add(num)

    if not refs_in_order:
        return answer  # No citations to process

    valid_refs = [n for n in refs_in_order if n in source_metadata]

    if not valid_refs:
        return answer

    # Renumber sequentially by first appearance
    remap = {old: new for new, old in enumerate(valid_refs, 1)}

    def _replace_ref(m):
        old = int(m.group(1))
        if old in remap:
            return f"[{remap[old]}]"
        return m.group(0)

    new_body = re.sub(r"\[(\d+)\]", _replace_ref, body)

    # Rebuild footnotes from actual chunk metadata
    footnotes = []
    for old_num in valid_refs:
        new_num = remap[old_num]
        meta = source_metadata[old_num]
        footnotes.append(_format_footnote(new_num, meta))

    citations_block = "\n---\n### Citations\n" + "\n".join(footnotes)
    return new_body.rstrip() + "\n\n" + citations_block



def ask(
    question: str,
    ticker: str,
    show_context: bool = False,
) -> dict:
    ticker = ticker.upper()
    all_chunks = []
    all_context_parts = []
    source_counter = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    tools_called = []
    seen_sources = {}
    source_metadata = {}

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Company: {ticker}\n\nQuestion: {question}"},
    ]

    try:
        # Agentic loop — LLM picks tools, max 3 iterations
        for iteration in range(MAX_TOOL_ITERATIONS):
            response = _client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.1,
                max_completion_tokens=2000,
            )

            msg = response.choices[0].message
            usage = response.usage
            total_prompt_tokens += usage.prompt_tokens
            total_completion_tokens += usage.completion_tokens

            if msg.tool_calls:
                messages.append(msg)

                for tool_call in msg.tool_calls:
                    fn_name = tool_call.function.name
                    fn_args = json.loads(tool_call.function.arguments)
                    tools_called.append(fn_name)

                    logger.info(f"Tool call [{iteration+1}]: {fn_name}({fn_args})")

                    result_str, chunks, source_counter = _execute_tool(
                        fn_name, fn_args, ticker,
                        source_counter, seen_sources, source_metadata,
                    )

                    all_chunks.extend(chunks)
                    all_context_parts.append(result_str)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_str,
                    })
            else:
                answer = msg.content
                break
        else:
            if msg.content:
                answer = msg.content
            else:
                tried = set(tools_called)
                source_names = []
                if "search_sec_filings" in tried:
                    source_names.append("SEC filings")
                if "search_financial_data" in tried:
                    source_names.append("financial data")
                if "search_news" in tried:
                    source_names.append("news")
                sources_str = ", ".join(source_names) if source_names else "available sources"
                answer = (
                    f"I searched {sources_str} for {ticker} but couldn't find "
                    f"data that directly answers this question. Try rephrasing "
                    f"— for example, asking about a specific metric, time "
                    f"period, or filing."
                )

    except Exception as e:
        logger.error(f"Tool calling failed ({e}), falling back to static retrieval")
        # Fallback to static retrieval if tool-calling fails
        context_str, all_chunks, source_metadata = _fallback_retrieve(question, ticker)
        all_context_parts = [context_str]
        tools_called = ["fallback_static"]

        fallback_response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Company: {ticker}\n\nQuestion: {question}\n\n"
                    f"## RETRIEVED CONTEXT\n{context_str}\n\n"
                    f"Answer the question using ONLY the context above."
                )},
            ],
            temperature=0.1,
            max_completion_tokens=2000,
        )
        answer = fallback_response.choices[0].message.content
        fu = fallback_response.usage
        total_prompt_tokens += fu.prompt_tokens
        total_completion_tokens += fu.completion_tokens

    answer = _postprocess_citations(answer, source_metadata)

    full_context = "\n\n---\n\n".join(all_context_parts)
    sources = []
    seen = set()
    for chunk in all_chunks:
        meta = chunk["metadata"]
        key = f"{meta.get('source_type')}_{meta.get('document_type')}_{meta.get('date')}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "source_type": meta.get("source_type"),
                "document_type": meta.get("document_type"),
                "company": meta.get("company_name", ticker),
                "date": meta.get("date"),
                "section": meta.get("section"),
                "file_path": meta.get("file_path", ""),
                "edgar_url": meta.get("edgar_url", ""),
                "url": meta.get("url", ""),
            })

    # Guardrails: citation check, number verification, LLM grounding
    validation = validate_response(answer, full_context, len(all_chunks))

    if validation["warnings"]:
        logger.warning(f"Guardrail warnings: {validation['warnings']}")

    total_tokens = total_prompt_tokens + total_completion_tokens

    result = {
        "answer": answer,
        "sources": sources,
        "ticker": ticker,
        "question": question,
        "routing": {
            "method": "tool_calling",
            "tools_called": tools_called,
            "iterations": min(len(set(tools_called)), MAX_TOOL_ITERATIONS),
        },
        "validation": validation,
        "token_usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
        },
    }

    if show_context:
        result["context"] = full_context

    return result
