"""Query rewrite: one LLM rewrite per iteration; search attempt 2 reuses it with one core spec's words removed."""
import contextlib
import io
import unittest

from support import FakeGroq, FakeTavily

import graph
import test_token_load
import tools
from test_token_load import run_graph

CALL_B = {"category": "laptop",
          "non_negotiable_specs": {"processor": "Intel Core i5 12th Gen", "ram": "8 GB", "storage": "512GB SSD"}}


class DropSpecFromQueryTest(unittest.TestCase):
    def test_drops_last_core_spec_present_in_query(self):
        query, change = tools.drop_spec_from_query("laptop i5 12th gen 8GB RAM 512GB SSD", CALL_B)
        self.assertEqual((query, change), ("laptop i5 12th gen 8GB RAM", "dropped 'storage'"))

    def test_skips_specs_whose_words_are_not_in_query(self):
        query, change = tools.drop_spec_from_query("laptop i5 12th gen 8GB RAM", CALL_B)
        self.assertEqual((query, change), ("laptop i5 12th gen", "dropped 'ram'"))

    def test_falls_back_to_dropping_last_word(self):
        query, change = tools.drop_spec_from_query("laptop backlit keyboard FHD", CALL_B)
        self.assertEqual((query, change), ("laptop backlit keyboard", "dropped the last word"))

    def test_category_words_never_dropped(self):
        call_b = {"category": "washing machine", "non_negotiable_specs": {"type": "front load machine"}}
        query, change = tools.drop_spec_from_query("washing machine front load 7kg", call_b)
        self.assertEqual((query, change), ("washing machine 7kg", "dropped 'type'"))

    def test_never_below_three_words(self):
        call_b = {"category": "laptop", "non_negotiable_specs": {"ram": "8GB", "everything": "i5 8GB"}}
        query, change = tools.drop_spec_from_query("laptop i5 8GB", call_b)
        self.assertEqual((query, change), ("laptop i5 8GB buy online price", "added 'buy online price'"))

    def test_bare_category_query_is_varied_not_emptied(self):
        # the reported run: every query was "refrigerator " and attempt 2 dropped "the last word" of nothing
        call_b = {"category": "refrigerator", "non_negotiable_specs": {"capacity": "at least 300 L"}}
        query, change = tools.drop_spec_from_query("refrigerator ", call_b)
        self.assertEqual((query, change), ("refrigerator buy online price", "added 'buy online price'"))
        query, change = tools.drop_spec_from_query("refrigerator 3 star buy online price", call_b,
                                                   keep_words="3 star buy online price")
        self.assertEqual(change, "reversed the word order")
        self.assertEqual(query, "refrigerator price online buy star 3")

    def test_user_words_are_never_dropped(self):
        call_b = {"category": "refrigerator",
                  "non_negotiable_specs": {"capacity": "at least 250 L", "energy_rating": "at least 3 star"}}
        query, change = tools.drop_spec_from_query("refrigerator double door 250 litre 3 star frost free", call_b,
                                                   keep_words="double door frost free 250 litre 3 star")
        self.assertEqual(query, "refrigerator double door 250 litre 3 star frost free buy online price")
        self.assertEqual(change, "added 'buy online price'")


class RequestKeywordsTest(unittest.TestCase):
    CONVERSATION = "refrigerator under 50000. double door frost free refrigerator 250 litre 3 star under 50000"

    def test_reported_conversation(self):
        self.assertEqual(tools.request_keywords(self.CONVERSATION, "refrigerator"),
                         "double door frost free 250 litre 3 star")

    def test_budget_amounts_and_filler_removed_units_kept(self):
        self.assertEqual(tools.request_keywords("I want a 1.5 ton LG split AC for my bedroom, budget ₹45,000", "AC"),
                         "1.5 ton lg split bedroom")
        self.assertEqual(tools.request_keywords("laptop for coding under 60k with 16GB RAM", "laptop"), "coding 16gb ram")
        self.assertEqual(tools.request_keywords("Samsung 1000 litre fridge around rs 90000", "fridge"),
                         "samsung 1000 litre")

    def test_display_resolution_shorthand_kept(self):
        self.assertEqual(tools.request_keywords("55 inch 4k TV under 50k", "TV"), "55 inch 4k")

    def test_nothing_but_category_and_budget(self):
        self.assertEqual(tools.request_keywords("refrigerator under 50000", "refrigerator"), "")


class BuildQueryTest(unittest.TestCase):
    LIVE_CALL_B = {  # the live Call B output for the reported refrigerator conversation (current prompt)
        "category": "refrigerator", "budget": 50000, "usecase": "general everyday use",
        "non_negotiable_specs": {"capacity": "at least 300 L", "cooling_technology": "frost free",
                                 "energy_rating": "at least 3 star", "voltage": "220-240 V"},
        "negotiable_specs": {"color": "any", "features": "water dispenser, ice maker optional",
                             "type": "double door or single door"}}

    def build(self, rewrite, user_query, call_b=None, include_negotiable=True):
        seen = []

        def text(prompt):
            seen.append(prompt)
            return rewrite
        fake = FakeGroq({"text": text}).install()
        try:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                query = tools.build_query(call_b or self.LIVE_CALL_B, include_negotiable, user_query=user_query)
        finally:
            fake.uninstall()
        return query, seen[0], out.getvalue()

    def test_empty_rewrite_falls_back_to_users_words(self):
        query, _, log = self.build("", RequestKeywordsTest.CONVERSATION)
        self.assertEqual(query, "refrigerator double door frost free 250 litre 3 star")
        self.assertIn("Query rewrite returned nothing", log)

    def test_empty_rewrite_and_no_user_words_uses_spec_values(self):
        query, _, _ = self.build("  ", "refrigerator under 50000", include_negotiable=False)
        self.assertEqual(query, "refrigerator 300 L frost free 3 star 220-240 V")

    def test_user_words_come_before_rewrite_without_repeats(self):
        query, prompt, _ = self.build("300L frost free 3 star inverter", RequestKeywordsTest.CONVERSATION)
        self.assertEqual(query, "refrigerator double door frost free 250 litre 3 star 300L inverter")
        self.assertIn("'capacity': 'at least 300 L'", prompt)
        self.assertIn("'color': 'any'", prompt)  # include_negotiable adds the first 2 negotiable specs

    def test_rewrite_prompt_is_category_neutral_and_never_empty(self):
        _, prompt, _ = self.build("x", "refrigerator")
        self.assertIn("or capacity, type, energy rating and brand for an appliance", prompt)
        self.assertIn("Always output at least one keyword phrase.", prompt)
        self.assertNotIn("Prioritize only the specs", prompt)


class RewritePerIterationTest(unittest.TestCase):
    def test_one_rewrite_per_iteration_and_attempt_two_differs(self):
        # no product pages, so every iteration runs both search attempts
        final, counter, fake, log = run_graph(product_pages=False)
        self.assertEqual(counter.calls["query"], 4)
        self.assertIn("Search attempt 2: dropped 'ram' in 'laptop coding i5 12th gen 8GB RAM' -> 'laptop coding i5 12th gen'", log)

    def test_attempt_two_query_sent_to_tavily(self):
        queries = []
        handlers, search = test_token_load.answers(product_pages=False)

        def recording_search(query, domains):
            queries.append(query)
            return search(query, domains)
        fake = FakeGroq(handlers).install()
        tavily = FakeTavily(search=recording_search).install()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                graph.build_graph().invoke({"user_query": "laptop"}, config=graph.make_config())
        finally:
            fake.uninstall()
            tavily.uninstall()
        self.assertEqual(queries, ["laptop i5 12th gen 8GB RAM", "laptop i5 12th gen"] * 4)


if __name__ == "__main__":
    unittest.main()
