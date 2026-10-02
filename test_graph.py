"""
Runs the sample query through the LangGraph pipeline and prints each node as it runs,
so the step order / stop behaviour can be compared against the old run_pipeline loop.

Usage: python test_graph.py ["your query"]
"""
import sys
from collections import Counter

from langchain_core.callbacks import BaseCallbackHandler

from graph import build_graph, make_config, is_qualified
from tools import is_product_page_url

SAMPLE_QUERY = "I want to buy a referigerator, budget 60000, family use"


class LLMCallCounter(BaseCallbackHandler):
    """Counts every chat-model request (including retries), failures and tokens, per graph node."""

    def __init__(self):
        self.calls = Counter()
        self.errors = Counter()
        self.tokens = Counter()

    def on_chat_model_start(self, serialized, messages, *, metadata=None, **kwargs):
        self.calls[(metadata or {}).get("langgraph_node", "?")] += 1

    def on_llm_end(self, response, **kwargs):
        usage = (response.llm_output or {}).get("token_usage") or {}
        self.tokens["prompt"] += usage.get("prompt_tokens", 0)
        self.tokens["completion"] += usage.get("completion_tokens", 0)

    def on_llm_error(self, error, *, metadata=None, **kwargs):
        self.errors[f"{(metadata or {}).get('langgraph_node', '?')}: {type(error).__name__}"] += 1


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
        return f"attempt={update['search_attempt']} results={len(urls)} product_pages={product_pages}/{len(urls)}"
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


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else SAMPLE_QUERY
    graph = build_graph()
    config = make_config(progress_callback=lambda msg: print(f"    [progress] {msg}"))
    counter = LLMCallCounter()
    config["callbacks"] = [counter]

    final_report = None
    all_candidates = []
    search_totals = Counter()
    for chunk in graph.stream({"user_query": query}, config=config, stream_mode="updates"):
        for node, update in chunk.items():
            print(f"==> {node}: {summarize(node, update or {})}")
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

    print("\n" + "=" * 80 + "\n")
    print(final_report)

    print("\n" + "=" * 80)
    print(f"LLM requests: {sum(counter.calls.values())} {dict(counter.calls)}")
    print(f"LLM errors:   {sum(counter.errors.values())} {dict(counter.errors)}")
    print(f"LLM tokens:   {counter.tokens['prompt'] + counter.tokens['completion']} "
          f"(prompt {counter.tokens['prompt']}, completion {counter.tokens['completion']})")
    print(f"Search product-page hit rate: {search_totals['product_pages']}/{search_totals['results']}")
    priced = [c for c in all_candidates if c.get("price") is not None]
    print(f"Candidates priced: {len(priced)}/{len(all_candidates)} "
          f"{dict(Counter(c.get('price_source') for c in all_candidates))}")
    for c in all_candidates:
        print(f"  - {c['product_name'][:70]} | price={c.get('price')} via {c.get('price_source')} {c.get('price_source_url') or ''}")


if __name__ == "__main__":
    main()
