import json
import logging

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

2. **Cite with footnotes.** Place [N] inline right after the claim. End your response with a compact footnotes block. Build each footnote from the [Source N] header metadata you received.

   Format: [N] TICKER FILING (DATE), SECTION
   Example:
   Revenue was $394.3B[1], up from $383.3B[2].

   [1] AAPL 10-K (2024-11-01), Item 8
   [2] AAPL Income Statement (2023-09-30)

3. **Never invent numbers.** If a number isn't in the data and can't be computed from it, say so briefly and move on. Don't dwell on what's missing — focus on what you CAN answer.

4. **Be specific with numbers.** Always include units, timeframes, and currency: "$394.3B in revenue for FY 2025."

## RESPONSE STYLE

- Lead with the direct answer. No preamble.
- Support with data and footnote citations.
- If you can partially answer, do so confidently — then note what's missing in one sentence, not a paragraph.
- Synthesize and analyze. Don't just list numbers — explain what they mean (trends, comparisons, implications).
- Keep it concise. A good analyst doesn't pad their answers.
- Always end with the footnotes block. No exceptions."""


def _format_tool_results(chunks: list[dict], source_counter: int) -> tuple[str, int]:
    if not chunks:
        return "No results found.", source_counter

    parts = []
    for chunk in chunks:
        source_counter += 1
        meta = chunk["metadata"]

        ticker_str = meta.get("ticker", "")
        doc_type = meta.get("document_type", "unknown").replace("_", " ").title()
        date = meta.get("date", "")
        section = meta.get("section", "")

        header = f"{ticker_str} {doc_type}"
        if date:
            header += f" ({date})"
        if section:
            header += f", {section}"

        parts.append(f"[Source {source_counter}] {header}\n{chunk['text']}")

    return "\n\n---\n\n".join(parts), source_counter


def _execute_tool(
    tool_name: str,
    args: dict,
    ticker: str,
    source_counter: int,
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

    formatted, source_counter = _format_tool_results(chunks, source_counter)

    logger.info(
        f"Tool {tool_name}(query='{query[:50]}', n={n_results}) → "
        f"{len(chunks)} chunks"
    )

    return formatted, chunks, source_counter


def _fallback_retrieve(question: str, ticker: str) -> tuple[str, list[dict]]:
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

    formatted, _ = _format_tool_results(chunks, 0)
    return formatted, chunks


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
                tool_choice="auto" if iteration == 0 else "auto",
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
                        fn_name, fn_args, ticker, source_counter,
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
        context_str, all_chunks = _fallback_retrieve(question, ticker)
        all_context_parts = [context_str]
        tools_called = ["fallback_static"]

        fallback_response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Company: {ticker}\n\nQuestion: {question}\n\n"
                    f"## RETRIEVED CONTEXT\n{context_str}\n\n"
                    f"Answer the question using ONLY the context above. "
                    f"Cite every factual claim with [Source N]."
                )},
            ],
            temperature=0.1,
            max_tokens=2000,
        )
        answer = fallback_response.choices[0].message.content
        fu = fallback_response.usage
        total_prompt_tokens += fu.prompt_tokens
        total_completion_tokens += fu.completion_tokens

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
