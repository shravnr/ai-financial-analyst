import json
import logging
import random
import time
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown

from src.config import OPENAI_API_KEY, LLM_MODEL_MINI, RAW_DATA_DIR
from src.ingestion.orchestrator import ingest_company
from src.processing.pipeline import process_company
from src.processing.vector_store import get_stats, is_ticker_indexed
from src.rag.chain import ask

console = Console()


logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)



_THINKING_MESSAGES = [
    "Let me look into that...",
    "Pulling that up for you...",
    "On it — give me a sec...",
    "Good question, let me check the data...",
    "Looking into it...",
]

_ROUTER_PROMPT = """\
You are a professional, straightforward financial analyst. No fluff — direct and concise. Classify the user's message and respond as JSON.

If they're asking about a specific company's financials, stock, or business:
  {{"type": "financial", "ticker": "AAPL", "company": "Apple"}}

If they mention a company you can't resolve to a US ticker:
  {{"type": "financial", "ticker": null, "company": "Acme Corp"}}

If they're asking a financial question but don't mention which company:
  {{"type": "financial", "ticker": null, "company": null}}

If they're saying goodbye or ending the conversation:
  {{"type": "farewell", "response": "a warm goodbye, 1 sentence"}}

If they're making casual conversation (greeting, thanks, chitchat, anything else):
  {{"type": "chat", "response": "a brief, professional reply (1 sentence) that steers toward asking about a company"}}

Only return US-listed tickers."""

_llm_client = OpenAI(api_key=OPENAI_API_KEY)


def _route_input(user_message: str) -> dict:
    try:
        resp = _llm_client.chat.completions.create(
            model=LLM_MODEL_MINI,
            messages=[
                {"role": "system", "content": _ROUTER_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_completion_tokens=150,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return {
            "type": "chat",
            "response": "Hey, I'm here to help with financial questions! "
                        "Try asking about a company like Apple or Tesla.",
        }



def _show_greeting():
    console.print()
    console.print(
        "  [bold cyan]Hey! I'm your financial analyst.[/bold cyan]\n"
        "  Ask me anything about a public company — earnings, risks, news, "
        "whatever you need.\n"
        "  [dim]Just mention a company by name and I'll take it from there. "
        "Type /quit when you're done.[/dim]"
    )
    console.print()


def _show_answer(result: dict):
    answer = result["answer"]

    separator = "\n---\n### Citations\n"
    if separator in answer:
        body, citations = answer.split(separator, 1)
        console.print(Markdown(body))
        console.print()
        console.print("[dim]Citations:[/dim]")
        for line in citations.strip().splitlines():
            console.print(f"  [dim]{line}[/dim]")
    else:
        console.print(Markdown(answer))

    console.print()


def _load_company(ticker: str) -> bool:
    ticker = ticker.upper()
    raw_dir = RAW_DATA_DIR / ticker

    if raw_dir.exists():
        console.print(f"  [dim]{ticker} data already cached, skipping download.[/dim]")
    else:
        console.print(f"  [cyan]Fetching data for {ticker}...[/cyan]")
        start = time.time()
        result = ingest_company(ticker)
        elapsed = time.time() - start

        summary = result.get("summary", {})
        errors = summary.get("errors", [])

        if not summary.get("sec_filings") and not summary.get("fmp_datasets"):
            console.print(f"  [red]Could not find data for {ticker}.[/red]")
            return False

        console.print(
            f"  [green]Got it:[/green] "
            f"{summary.get('sec_filings', 0)} SEC filings, "
            f"{summary.get('fmp_datasets', 0)} financial datasets, "
            f"{summary.get('news_articles', 0)} news articles "
            f"[dim]({elapsed:.1f}s)[/dim]"
        )

    console.print(f"  [cyan]Indexing {ticker}...[/cyan]")
    start = time.time()
    stats = process_company(ticker)
    elapsed = time.time() - start

    if stats.get("error"):
        console.print(f"  [red]{stats['error']}[/red]")
        return False

    chunks = stats["chunks"]
    console.print(
        f"  [green]Ready:[/green] {chunks['total']} chunks indexed "
        f"[dim]({elapsed:.1f}s)[/dim]"
    )

    return True



def main():
    _show_greeting()
    current_ticker = None

    while True:
        try:
            user_input = console.input("[bold cyan]you >[/bold cyan] ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("/quit", "/exit", "/q"):
                console.print(f"\n[bold green]analyst >[/bold green] See you later! Happy investing.")
                break

            route = _route_input(user_input)
            intent = route.get("type", "chat")

            if intent == "farewell":
                console.print(f"\n[bold green]analyst >[/bold green] {route.get('response', 'See you!')}")
                break

            if intent == "chat":
                console.print(f"\n[bold green]analyst >[/bold green] {route.get('response', '')}")
                console.print() 
                continue

            # intent == "financial"
            ticker = (route.get("ticker") or "").upper().strip()

            if not ticker:
                company = route.get("company")
                if company:
                    console.print(
                        f"\n[bold green]analyst >[/bold green] "
                        f"I don't have data for {company} — I can only look up "
                        f"US-listed public companies. Try a company like Apple or Tesla."
                    )
                    console.print()
                    continue
                elif current_ticker:
                    ticker = current_ticker
                else:
                    console.print(
                        f"\n[bold green]analyst >[/bold green] "
                        "I'd love to dig into that, but which company? "
                        "Mention a name like Apple or Tesla."
                    )
                    console.print()
                    continue

            if ticker != current_ticker:
                if not is_ticker_indexed(ticker):
                    console.print(
                        f"\n  [cyan]Let me pull data for [bold]{ticker}[/bold] first — "
                        f"this takes about 30 seconds for a new company...[/cyan]"
                    )
                    if _load_company(ticker):
                        current_ticker = ticker
                    else:
                        console.print(f"\n  [red]Hmm, I couldn't load {ticker}. Try a different ticker?[/red]")
                        console.print()
                        continue
                else:
                    current_ticker = ticker

            # Bare company name / ticker without a real question
            stripped = user_input.lower().strip().rstrip("?.! ")
            company_name = (route.get("company") or "").lower()
            if stripped in (ticker.lower(), company_name) or (
                len(user_input.split()) <= 2 and not any(
                    w in user_input.lower() for w in ("what", "how", "why", "when", "tell", "show", "compare", "is", "are", "does", "did", "can", "will")
                )
            ):
                console.print(
                    f"\n[bold green]analyst >[/bold green] "
                    f"Got it — what would you like to know about {ticker}? "
                    f"Try something like earnings, risks, or recent news."
                )
                console.print()
                continue

            console.print(f"\n[bold green]analyst >[/bold green] [dim]{random.choice(_THINKING_MESSAGES)}[/dim]")
            result = ask(user_input, current_ticker)
            _show_answer(result)

        except KeyboardInterrupt:
            console.print(f"\n\n[bold green]analyst >[/bold green] See you later! Happy investing.")
            break
        except Exception as e:
            console.print(f"  [red]Oops, something went wrong: {e}[/red]")
            logging.getLogger(__name__).exception("Unexpected error")


if __name__ == "__main__":
    main()
