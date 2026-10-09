"""
Runs the sample query through the LangGraph pipeline and prints each node as it runs,
so the step order / stop behaviour can be compared against the old run_pipeline loop.

Usage: python test_graph.py ["your query"]
The report and summary are also saved to runs/run-<timestamp>.txt (UTF-8).
"""
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler

from graph import build_graph, make_config, is_qualified
from tools import is_product_page_url
from tracing import trace_metadata

SAMPLE_QUERY = "I want to buy a referigerator, budget 60000, family use"
RUNS_DIR = Path(__file__).resolve().parent / "runs"


# structured-output tool name -> call letter; plain-text calls are named after their node
CALL_TYPES = {"Call-A": "A", "Call-B": "B", "Candidate-Extraction": "C", "Spec-Merge": "D",
              "Price-Extraction": "E", "Fit-Evaluation": "F"}
TEXT_CALL_TYPES = {"search": "query", "report": "report"}
CALL_TYPE_ORDER = ["A", "B", "query", "C", "D", "E", "F", "report"]


def call_type(invocation_params, metadata):
    tools = (invocation_params or {}).get("tools") or []
    if tools:
        name = tools[0].get("function", {}).get("name")
        return CALL_TYPES.get(name, name)
    node = (metadata or {}).get("langgraph_node", "?")
    return TEXT_CALL_TYPES.get(node, node)


class LLMCallCounter(BaseCallbackHandler):
    """Counts every chat-model request (including retries), failures and tokens, per call type (A-F, query, report)."""

    def __init__(self):
        self.calls = Counter()
        self.errors = Counter()
        self.tokens = Counter()
        self.prompt_tokens = Counter()      # per call type
        self.completion_tokens = Counter()  # per call type
        self._run_types = {}

    def on_chat_model_start(self, serialized, messages, *, run_id=None, metadata=None, invocation_params=None, **kwargs):
        kind = call_type(invocation_params, metadata)
        self._run_types[run_id] = kind
        self.calls[kind] += 1

    def on_llm_end(self, response, *, run_id=None, **kwargs):
        kind = self._run_types.pop(run_id, "?")
        usage = (response.llm_output or {}).get("token_usage") or {}
        self.tokens["prompt"] += usage.get("prompt_tokens", 0)
        self.tokens["completion"] += usage.get("completion_tokens", 0)
        self.prompt_tokens[kind] += usage.get("prompt_tokens", 0)
        self.completion_tokens[kind] += usage.get("completion_tokens", 0)

    def on_llm_error(self, error, *, run_id=None, **kwargs):
        self.errors[f"{self._run_types.pop(run_id, '?')}: {type(error).__name__}"] += 1

    def per_type_lines(self):
        kinds = [k for k in CALL_TYPE_ORDER if self.calls[k]] + sorted(set(self.calls) - set(CALL_TYPE_ORDER))
        return [f"  {k:7} requests={self.calls[k]:3}  prompt={self.prompt_tokens[k]:6}  completion={self.completion_tokens[k]:6}"
                for k in kinds]


def summarize(node, update):
    if node == "intake":
        if "clarify_question" in update:
            return f"clarify -> {update['clarify_question']!r}"
        req = update["requirements"]
        return f"category={req['category']!r} budget={req['budget']} non_negotiable={list(req['non_negotiable_specs'])}"
    if node == "start_iteration":
        return f"iteration={update['iteration']}"
    if node == "search":
        urls = list(update.get("search_snippets", {}))
        product_pages = sum(is_product_page_url(u) for u in urls)
        return (f"attempt={update['search_attempt']} query={update.get('search_query')!r} "
                f"results={len(urls)} product_pages={product_pages}/{len(urls)}")
    if node in ("extract_candidates", "verify_specs", "check_prices", "score_fit"):
        return ", ".join(
            f"{c['product_name']} (specs_found={c.get('specs_found')}, price={c.get('price')}, "
            f"within_budget={c.get('within_budget')}, fit={c.get('fit_score')})"
            for c in update["new_candidates"]) or "no candidates"
    if node == "merge":
        all_c = update["all_candidates"]
        return f"all_candidates={len(all_c)} qualified={sum(1 for c in all_c if is_qualified(c))}"
    if node == "report":
        return f"is_degraded={update['is_degraded']} report_candidates={len(update['report_candidates'])}"
    return ""


def build_summary(counter, search_totals, all_candidates):
    lines = [
        f"LLM requests: {sum(counter.calls.values())} {dict(counter.calls)}",
        f"LLM errors:   {sum(counter.errors.values())} {dict(counter.errors)}",
        f"LLM tokens:   {counter.tokens['prompt'] + counter.tokens['completion']} "
        f"(prompt {counter.tokens['prompt']}, completion {counter.tokens['completion']})",
        "LLM tokens per call type (requests include retries; errored requests report no tokens):",
        *counter.per_type_lines(),
        f"Search product-page hit rate: {search_totals['product_pages']}/{search_totals['results']}",
    ]
    priced = [c for c in all_candidates if c.get("price") is not None]
    lines.append(f"Candidates priced: {len(priced)}/{len(all_candidates)} "
                 f"{dict(Counter(c.get('price_source') for c in all_candidates))}")
    for c in all_candidates:
        lines.append(f"  - {c['product_name'][:70]} | price={c.get('price')} via {c.get('price_source')} {c.get('price_source_url') or ''}")
        lines.append(f"      specs_found={c.get('specs_found')} known_specs={c.get('known_specs')}")
    return lines


def rejection_lines(all_candidates):
    """Every rejected ₹ amount; only written to the run file (the console gets one summary line per source)."""
    lines = ["Price rejections (full list):"]
    for c in all_candidates:
        for r in c.get("price_rejections", []):
            lines.append(f"  - {c['product_name'][:50]} | {r['source']} | {r['amount']} | {r['reason']} | {r['url']}")
    return lines if len(lines) > 1 else []


def save_run(query, final_report, summary_lines):
    """Writes the report and summary to runs/run-<timestamp>.txt, so a console problem can't lose them."""
    RUNS_DIR.mkdir(exist_ok=True)
    path = RUNS_DIR / f"run-{datetime.now():%Y%m%d-%H%M%S}.txt"
    path.write_text(f"Query: {query}\n\n{final_report}\n\n" + "\n".join(summary_lines) + "\n", encoding="utf-8")
    return path


def main():
    # Windows uses cp1252 for redirected output, which cannot encode ₹ and crashed the report print
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    query = sys.argv[1] if len(sys.argv) > 1 else SAMPLE_QUERY
    graph = build_graph()
    config = make_config(progress_callback=lambda msg: print(f"    [progress] {msg}"),
                         run_name="product-research-test", tags=["test"], metadata=trace_metadata(query))
    counter = LLMCallCounter()
    config["callbacks"] = [counter]

    final_report = "(no report: the run did not finish)"
    all_candidates = []
    search_totals = Counter()
    try:
        for chunk in graph.stream({"user_query": query}, config=config, stream_mode="updates"):
            for node, update in chunk.items():
                print(f"==> {node}: {summarize(node, update or {})}")
                if node in ("extract_candidates", "verify_specs"):
                    for c in update["new_candidates"]:
                        print(f"      known_specs[{c['product_name'][:40]}]: {c.get('known_specs')}")
                if node == "search":
                    urls = list(update.get("search_snippets", {}))
                    search_totals["results"] += len(urls)
                    search_totals["product_pages"] += sum(is_product_page_url(u) for u in urls)
                elif node == "merge":
                    all_candidates = update["all_candidates"]
                if node == "report":
                    final_report = update["report"]
                elif node == "intake" and "clarify_question" in update:
                    final_report = update["clarify_question"]
    finally:
        # also runs when the pipeline raises (e.g. daily quota), so the partial summary is kept
        summary_lines = build_summary(counter, search_totals, all_candidates)
        print("\n" + "=" * 80 + "\n")
        print(final_report)
        print("\n" + "=" * 80)
        print("\n".join(summary_lines))
        print(f"Saved to {save_run(query, final_report, summary_lines + rejection_lines(all_candidates))}")


if __name__ == "__main__":
    main()
