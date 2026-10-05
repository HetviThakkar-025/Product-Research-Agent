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
        query, dropped = tools.drop_spec_from_query("laptop i5 12th gen 8GB RAM 512GB SSD", CALL_B)
        self.assertEqual((query, dropped), ("laptop i5 12th gen 8GB RAM", "storage"))

    def test_skips_specs_whose_words_are_not_in_query(self):
        query, dropped = tools.drop_spec_from_query("laptop i5 12th gen 8GB RAM", CALL_B)
        self.assertEqual((query, dropped), ("laptop i5 12th gen", "ram"))

    def test_falls_back_to_dropping_last_word(self):
        query, dropped = tools.drop_spec_from_query("laptop backlit keyboard FHD", CALL_B)
        self.assertEqual((query, dropped), ("laptop backlit keyboard", None))

    def test_category_words_never_dropped(self):
        call_b = {"category": "washing machine", "non_negotiable_specs": {"type": "front load machine"}}
        query, dropped = tools.drop_spec_from_query("washing machine front load 7kg", call_b)
        self.assertEqual((query, dropped), ("washing machine 7kg", "type"))

    def test_spec_covering_whole_query_is_skipped(self):
        call_b = {"category": "laptop", "non_negotiable_specs": {"ram": "8GB", "everything": "i5 8GB"}}
        query, dropped = tools.drop_spec_from_query("laptop i5 8GB", call_b)
        self.assertEqual((query, dropped), ("laptop i5", "ram"))


class RewritePerIterationTest(unittest.TestCase):
    def test_one_rewrite_per_iteration_and_attempt_two_differs(self):
        # no product pages, so every iteration runs both search attempts
        final, counter, fake, log = run_graph(product_pages=False)
        self.assertEqual(counter.calls["query"], 4)
        self.assertIn("Search attempt 2: dropped 'ram' from 'laptop i5 12th gen 8GB RAM'", log)

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
