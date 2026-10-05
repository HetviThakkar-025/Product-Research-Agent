"""
Token-load changes: per-call-type token counts, Call C skipped when no result is a product page,
Call D skipped when the spec search is empty, and the title trimmed from Call C's input.
The graph runs end to end on FakeGroq/FakeTavily.
"""
import contextlib
import io
import unittest

from support import FakeGroq, FakeTavily

import graph
import test_graph
import tools

PRODUCT_URL = "https://www.amazon.in/HP-15-Laptop/dp/B0DCG26YC5"
PRODUCT_URL_2 = "https://www.flipkart.com/hp-15s-laptop/p/itm1234567890"
LISTING_URL = "https://www.amazon.in/i5-12th-gen-laptops/s"


def answers(product_pages=True):
    handlers = {
        "Call-A": lambda p: {"status": "clear", "usecase": "coding", "budget": 60000, "category": "laptop", "question": None},
        "Call-B": lambda p: {"usecase": "coding", "budget": 60000, "category": "laptop",
                             "non_negotiable_specs": {"processor": "i5 12th Gen", "ram": "8GB"},
                             "negotiable_specs": {"battery": "6 hours"}},
        "text": lambda p: "i5 12th gen 8GB RAM" if "Shorten them" in p else "## Report",
        "Candidate-Extraction": lambda p: {"candidates": [
            {"product_name": "HP 15 fd0070TU", "known_specs": {"Processor": "i5-1235U", "RAM": "8GB"},
             "specs_found": True, "source_url": PRODUCT_URL},
            # Call C wrongly says False; the Python rule must still find both keys after Call D is skipped
            {"product_name": "HP 15s fy5007TU", "known_specs": {"processor": "i5-1235U", "Ram": "8GB"},
             "specs_found": False, "source_url": PRODUCT_URL_2}]},
        "Spec-Merge": lambda p: {"new_specs": {}},
        "Price-Extraction": lambda p: {"price": 52990, "availability": "in_stock"},
        "Fit-Evaluation": lambda p: {"fit_score": 8, "reasoning": "ok", "missing_or_weak_specs": []},
    }
    retail = [{"url": LISTING_URL, "title": "Amazon.in : i5 laptops", "content": "many laptops", "score": 0.9}]
    if product_pages:
        retail += [{"url": PRODUCT_URL, "title": "HP 15 fd0070TU : Amazon.in: Electronics",
                    "content": "HP 15 fd0070TU i5-1235U 8GB RAM Buy for ₹52,990", "score": 0.8},
                   {"url": PRODUCT_URL_2, "title": "HP 15s fy5007TU", "content": "HP 15s fy5007TU i5-1235U ₹52,990", "score": 0.7}]

    def search(query, domains):
        if domains == tools.SPEC_DOMAINS:
            return {"results": []}  # spec search finds nothing -> Call D must be skipped
        return {"results": retail}
    return handlers, search


def run_graph(product_pages=True):
    handlers, search = answers(product_pages)
    fake = FakeGroq(handlers, prompt_tokens=100, completion_tokens=10).install()
    tavily = FakeTavily(search=search).install()
    counter = test_graph.LLMCallCounter()
    config = graph.make_config()
    config["callbacks"] = [counter]
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            final = graph.build_graph().invoke({"user_query": "laptop for coding under 60000"}, config=config)
    finally:
        fake.uninstall()
        tavily.uninstall()
    return final, counter, fake, out.getvalue()


class FullMockedRunTest(unittest.TestCase):
    def test_tokens_counted_per_call_type(self):
        final, counter, _, _ = run_graph()
        self.assertEqual(sum(1 for c in final["all_candidates"] if graph.is_qualified(c)), 2)
        self.assertEqual(dict(counter.calls), {"A": 1, "B": 1, "query": 1, "C": 1, "E": 2, "F": 2, "report": 1})
        self.assertEqual(counter.prompt_tokens["E"], 200)
        self.assertEqual(counter.completion_tokens["F"], 20)
        self.assertEqual(counter.tokens["prompt"], 900)
        lines = counter.per_type_lines()
        self.assertTrue(lines[0].strip().startswith("A "))
        self.assertTrue(lines[-1].strip().startswith("report"))

    def test_call_d_skipped_when_spec_search_empty(self):
        final, counter, fake, log = run_graph()
        self.assertNotIn("Spec-Merge", fake.names())
        self.assertIn("Skipped Call D for HP 15s fy5007TU: spec search returned nothing", log)
        hp15s = next(c for c in final["all_candidates"] if c["product_name"] == "HP 15s fy5007TU")
        self.assertTrue(hp15s["specs_found"])

    def test_call_c_skipped_when_no_product_pages(self):
        final, counter, fake, log = run_graph(product_pages=False)
        self.assertNotIn("Candidate-Extraction", fake.names())
        self.assertIn("Skipped Call C: none of the 1 search results is a product page", log)
        self.assertTrue(final["is_degraded"])
        self.assertEqual(final["all_candidates"], [])


class CounterTest(unittest.TestCase):
    def test_errors_attributed_to_call_type(self):
        counter = test_graph.LLMCallCounter()
        counter.on_chat_model_start({}, [], run_id="r1", metadata={"langgraph_node": "extract_candidates"},
                                    invocation_params={"tools": [{"function": {"name": "Candidate-Extraction"}}]})
        counter.on_llm_error(RuntimeError("429"), run_id="r1")
        self.assertEqual(dict(counter.errors), {"C: RuntimeError": 1})

    def test_plain_text_calls_named_by_node(self):
        self.assertEqual(test_graph.call_type({}, {"langgraph_node": "search"}), "query")
        self.assertEqual(test_graph.call_type({}, {"langgraph_node": "report"}), "report")


class DropRepeatedTitleTest(unittest.TestCase):
    def trim(self, title, content):
        return tools.drop_repeated_title({"results": [{"url": "u", "title": title, "content": content}]})["results"][0]["content"]

    def test_title_prefix_removed_with_retailer_suffix(self):
        self.assertEqual(self.trim("HP 15 fd0070TU Laptop : Amazon.in: Electronics", "## HP 15 fd0070TU Laptop, 8GB RAM ₹52,990"),
                         "8gb ram ₹52,990")

    def test_content_that_is_only_the_title_emptied(self):
        self.assertEqual(self.trim("HP 15s, 12th Gen Intel Core i5-1235U Laptop (8GB DDR4, 512GB SSD)", "HP 15s, 12th Gen Intel"), "")

    def test_unrelated_content_unchanged(self):
        content = "3.   Image 28: HP 15s,12th Gen Intel Core i5-1235U"
        self.assertEqual(self.trim("HP 15, 12 Gen Intel Core i5-1235U : Amazon.in", content), content)

    def test_missing_title_unchanged(self):
        self.assertEqual(self.trim("", "anything"), "anything")


if __name__ == "__main__":
    unittest.main()
