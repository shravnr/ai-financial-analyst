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

#  Tool definitions for OpenAI function calling ─

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

# Source type mapping for each tool
_TOOL_SOURCE_MAP = {
    "search_sec_filings": ["sec"],
    "search_financial_data": ["fmp"],
    "search_news": ["news"],
}

SYSTEM_PROMPT = """You are a sharp, confident financial analyst. Answer questions using data retrieved through your tools. Be direct and analytical — like a real analyst would be.

## WORKFLOW

1. Figure out what data you need.
2. Call the right tool(s). Use multiple tools if the question spans different data types.
3. Analyze the data and give a clear, direct answer.

## RULES

1. **Base your answer on tool results.** You may compute derived metrics from the data (margins, growth rates, ratios) — that's what analysts do. Do not recall specific numbers from your training data.

2. **Cite inline.** Place [N] right after the claim, where N is the exact Source number from the retrieved context. If data comes from [Source 3], cite as [3].

   Example: Revenue was $394.3B[1], driven by strong Services growth[4].

   Do NOT renumber. Use the exact source reference numbers. Do NOT write a footnotes block at the end — footnotes are generated automatically by the system.

3. **Never invent numbers.** If a number isn't in the data and can't be computed from it, say so briefly and move on. Don't dwell on what's missing — focus on what you CAN answer.

4. **Be specific with numbers.** Always include units, timeframes, and currency: "$394.3B in revenue for FY 2025."

## RESPONSE STYLE

- Lead with the direct answer. No preamble.
- Support with data and inline [N] citations.
- If you can partially answer, do so confidently — then note what's missing in one sentence, not a paragraph.
- Synthesize and analyze. Don't just list numbers — explain what they mean (trends, comparisons, implications).
- Keep it concise. A good analyst doesn't pad their answers."""


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

        # Normalize section to top-level only (e.g., "Item 1A - Risk Factors > ..." → "Item 1A")
        section_key = section.split(">")[0].strip() if section else ""
        key = f"{ticker_str}_{doc_type}_{date}_{section_key}"
        if key in seen_sources:
            num = seen_sources[key]
        else:
            source_counter += 1
            num = source_counter
            seen_sources[key] = num
            # Store full metadata for mechanical footnote generation
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

    # Broaden if too few results
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


# ── Citation post-processing ─────────────────────────────────────────

def _strip_footnotes(answer: str) -> str:
    """Remove any LLM-generated footnotes block from end of answer."""
    lines = answer.rstrip().split("\n")

    # Walk backwards to find where footnotes start
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
    # Remove trailing --- separator if present
    if body.endswith("---"):
        body = body[:-3].rstrip()

    return body


def _format_footnote(num: int, meta: dict) -> str:
    """Build a single citation footnote with verifiable source reference."""
    source_type = meta.get("source_type", "").upper()
    ticker = meta.get("ticker", "")
    doc_type = meta.get("document_type", "").replace("_", " ").title()
    date = meta.get("date", "")
    section = meta.get("section", "")
    file_path = meta.get("file_path", "")
    edgar_url = meta.get("edgar_url", "")
    url = meta.get("url", "")

    # Build description
    desc = f"{source_type} — {ticker} {doc_type}"
    if date:
        desc += f" ({date})"
    if section:
        desc += f", {section}"

    # Add verifiable reference (only public URLs — skip local file paths)
    ref = edgar_url or url
    if ref:
        desc += f" → {ref}"

    return f"- \\[{num}\\] {desc}"


def _postprocess_citations(answer: str, source_metadata: dict) -> str:
    """Strip LLM footnotes, renumber citations, rebuild with accurate metadata."""
    if not source_metadata:
        return answer

    # 1. Strip any model-generated footnotes
    body = _strip_footnotes(answer)

    # 2. Find all [N] in body, ordered by first appearance
    refs_in_order = []
    seen = set()
    for m in re.finditer(r"\[(\d+)\]", body):
        num = int(m.group(1))
        if num not in seen:
            refs_in_order.append(num)
            seen.add(num)

    if not refs_in_order:
        return answer  # No citations to process

    # 3. Filter to citations that exist in our metadata
    valid_refs = [n for n in refs_in_order if n in source_metadata]

    if not valid_refs:
        # Model likely renumbered and nothing matches — return original
        return answer

    # 4. Build renumbering map: old source num → new sequential num
    remap = {old: new for new, old in enumerate(valid_refs, 1)}

    # 5. Renumber in body text
    def _replace_ref(m):
        old = int(m.group(1))
        if old in remap:
            return f"[{remap[old]}]"
        return m.group(0)

    new_body = re.sub(r"\[(\d+)\]", _replace_ref, body)

    # 6. Build accurate footnotes from chunk metadata
    footnotes = []
    for old_num in valid_refs:
        new_num = remap[old_num]
        meta = source_metadata[old_num]
        footnotes.append(_format_footnote(new_num, meta))

    citations_block = "\n---\n### Citations\n" + "\n".join(footnotes)
    return new_body.rstrip() + "\n\n" + citations_block


# ── Main entry point ─────────────────────────────────────────────────

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
        #  Agentic tool-calling loop
        for iteration in range(MAX_TOOL_ITERATIONS):
            response = _client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.1,
                max_tokens=2000,
            )

            msg = response.choices[0].message
            usage = response.usage
            total_prompt_tokens += usage.prompt_tokens
            total_completion_tokens += usage.completion_tokens

            # If the model wants to call tools
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
                # Model is done — has the final answer
                answer = msg.content
                break
        else:
            # Exhausted iterations — take the last response
            answer = msg.content or "I was unable to gather sufficient data to answer this question."

    except Exception as e:
        logger.error(f"Tool calling failed ({e}), falling back to static retrieval")
        # Graceful fallback to static pipeline
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
            max_tokens=2000,
        )
        answer = fallback_response.choices[0].message.content
        fu = fallback_response.usage
        total_prompt_tokens += fu.prompt_tokens
        total_completion_tokens += fu.completion_tokens

    #  Post-process citations: replace LLM footnotes with accurate ones
    answer = _postprocess_citations(answer, source_metadata)

    #  Build source list
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

    #  Run guardrails (rule-based only; LLM grounding runs in eval) ─
    validation = validate_response(answer, full_context, len(all_chunks), use_llm_grounding=False)
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
