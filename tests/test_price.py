"""
Price attribution: product match for fallback sources, model numbers kept from dropped duplicates,
and the 0.25x-3x budget plausibility bound. Uses the real Tavily fallback response for the HP 15
candidate from the 2026-10-05 laptop run (tests/fixtures/tavily_fallback_hp15.json).
"""
import contextlib
import io
import unittest

from support import FakeGroq, FakeTavily, load_fixture

import graph
import tools

HP15_URL = "https://www.amazon.in/HP-i5-1235U-Anti-Glare-Micro-Edge-15-6-inch/dp/B0DCG26YC5"
HP15S_URL = "https://www.amazon.in/HP-i5-1235U-15-6-inch-Graphics-fy5007TU/dp/B0CJBP38HR"
HP15_NAME = "HP 15, 12 Gen Intel Core i5-1235U, 8GB DDR4, 512GB SSD, 15.6-inch"
REQUIREMENTS = {'budget': 60000, 'non_negotiable_specs': {'processor': 'i5'}}


def fallback_fixture(query, domains):
    return load_fixture("tavily_fallback_hp15.json")


def candidate(name=HP15_NAME, url=HP15_URL, **extra):
    return {'product_name': name, 'source_url': url, 'search_snippet': '', **extra}


def check_prices(candidates, price_answer, requirements=REQUIREMENTS):
    """Runs check_prices with Call E answering price_answer(prompt); page extract returns no price."""
    fake = FakeGroq({"Price-Extraction": lambda prompt: {"price": price_answer(prompt), "availability": "unknown"}}).install()
    tavily = FakeTavily(search=fallback_fixture, extract=lambda urls: {"results": [{"url": urls[0], "raw_content": "no price"}]}).install()
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            result = graph.check_prices({"requirements": requirements, "new_candidates": candidates}, {})
    finally:
        fake.uninstall()
        tavily.uninstall()
    return result["new_candidates"], out.getvalue(), fake


class FallbackSearchTest(unittest.TestCase):
    def test_keeps_title_and_drops_listing_pages(self):
        tavily = FakeTavily(search=fallback_fixture).install()
        try:
            results = tools.search_price_fallback(HP15_NAME)
        finally:
            tavily.uninstall()
        self.assertEqual([r['url'] for r in results], [HP15_URL, HP15S_URL])
        self.assertTrue(results[0]['title'].startswith("HP 15, 12 Gen Intel Core i5-1235U"))
        self.assertIn("fd0070TU", results[0]['title'])


class ProductMatchTest(unittest.TestCase):
    def test_model_number_from_name(self):
        self.assertEqual(tools.product_match(candidate("HP 15 fd0070TU", url="x"), "https://other", "HP 15 fd0070TU laptop", ""),
                         'model number')

    def test_model_number_from_dropped_duplicate(self):
        c = candidate(url="https://www.flipkart.com/hp/p/itm123", model_numbers=['fd0070tu'])
        self.assertEqual(tools.product_match(c, HP15_URL, "... Backlit KB fd0070TU : Amazon.in", ""), 'model number')

    def test_model_number_in_content_counts(self):
        c = candidate("HP 15s fy5007TU", url="x")
        self.assertEqual(tools.product_match(c, "https://y", "HP laptop", "Intel UHD Graphics,fy5007TU"), 'model number')

    def test_same_normalized_url(self):
        c = candidate("Lenovo V15", url="https://www.amazon.in/Lenovo-V15/dp/B0ABCDEFGH?ref=x")
        self.assertEqual(tools.product_match(c, "https://www.amazon.in/dp/B0ABCDEFGH", "something else", ""), 'same url')

    def test_brand_and_distinctive_tokens(self):
        c = candidate(url="https://www.flipkart.com/hp/p/itm123")
        title = "HP 15s, 12th Gen Intel Core i5 1235U, 8 GB DDR4 Ram, 512GB SSD"
        self.assertEqual(tools.product_match(c, HP15S_URL, title, ""), 'brand and specs')

    def test_brand_alone_is_not_enough(self):
        c = candidate(url="https://www.flipkart.com/hp/p/itm123")
        self.assertIsNone(tools.product_match(c, HP15S_URL, "HP 15s, Intel Core i3-1215U, 8GB, 512GB SSD", ""))

    def test_other_brand_rejected(self):
        c = candidate("Lenovo V15 Intel Core i5-1235U 8GB 512GB", url="https://www.flipkart.com/lenovo/p/itm9")
        self.assertIsNone(tools.product_match(c, HP15_URL, "HP 15, Intel Core i5-1235U, 8GB DDR4, 512GB SSD", ""))

    def test_size_token_is_whole_word(self):
        c = candidate("HP 15 Intel Core i5-1235U 8GB 512GB", url="https://www.flipkart.com/hp/p/itm123")
        self.assertIsNone(tools.product_match(c, HP15S_URL, "HP 15 Intel Core i5-1235U 128GB 512GB", ""))


class DuplicateModelNumbersTest(unittest.TestCase):
    def run_extract(self, all_candidates, c_candidates):
        fake = FakeGroq({"Candidate-Extraction": lambda prompt: {"candidates": c_candidates}}).install()
        state = {"requirements": {'non_negotiable_specs': {'processor': 'i5'}, 'negotiable_specs': None},
                 "all_candidates": all_candidates, "search_snippets": {},
                 "search_results": {"results": [{"url": c['source_url'], "title": "", "content": ""} for c in c_candidates]}}
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                return graph.extract_candidates(state, {})
        finally:
            fake.uninstall()

    def test_dropped_duplicate_models_stored_on_earlier_candidate(self):
        earlier = candidate(known_specs={}, specs_found=False)
        update = self.run_extract([earlier], [
            {"product_name": "HP 15 fd0070TU", "source_url": HP15_URL, "known_specs": {"cpu": "i5"}, "specs_found": False}])
        self.assertEqual(update["new_candidates"], [])
        self.assertEqual(update["all_candidates"][0]['model_numbers'], ['fd0070tu'])
        self.assertNotIn('model_numbers', earlier, "state must not be mutated in place")

    def test_dropped_duplicate_models_stored_within_same_batch(self):
        update = self.run_extract([], [
            {"product_name": HP15_NAME, "source_url": HP15_URL, "known_specs": {"cpu": "i5"}, "specs_found": False},
            {"product_name": "HP 15 fd0070TU", "source_url": HP15_URL, "known_specs": {"cpu": "i5"}, "specs_found": False}])
        self.assertEqual(len(update["new_candidates"]), 1)
        self.assertEqual(update["new_candidates"][0]['model_numbers'], ['fd0070tu'])


class CheckPricesTest(unittest.TestCase):
    def test_live_run_price_246490_rejected_as_implausible(self):
        # the run's case: the fallback hit the candidate's own page, whose content is a carousel of other laptops
        [c], log, _ = check_prices([candidate()], lambda prompt: 246490)
        self.assertIn("Rejected implausible price 246490 for HP 15", log)
        self.assertIsNone(c['price'])
        self.assertEqual(c['within_budget'], "unknown")

    def test_known_gap_carousel_price_within_bound_is_accepted(self):
        # KNOWN GAP: ₹47,880 is another laptop in the page's carousel, but the page matches by URL and the
        # amount is plausible, so nothing rejects it. Pins current behaviour until a snippet-level check exists.
        [c], log, _ = check_prices([candidate()], lambda prompt: 47880)
        self.assertEqual((c['price'], c['price_source'], c['price_source_url']), (47880, 'fallback_search', HP15_URL))

    def test_mismatched_fallback_source_rejected_before_call_e(self):
        lenovo = candidate("Lenovo V15 Intel Core i5-1235U 8GB 512GB", url="https://www.flipkart.com/lenovo/p/itm9")
        [c], log, fake = check_prices([lenovo], lambda prompt: 47880)
        self.assertIn(f"Rejected price: product mismatch for {lenovo['product_name']}: {HP15_URL}", log)
        self.assertNotIn("Price-Extraction", fake.names())
        self.assertIsNone(c['price'])

    def test_bounds_are_inclusive_and_skipped_without_budget(self):
        self.assertTrue(graph.is_plausible_price(15000, 60000))
        self.assertTrue(graph.is_plausible_price(180000, 60000))
        self.assertFalse(graph.is_plausible_price(14999, 60000))
        self.assertFalse(graph.is_plausible_price(180001, 60000))
        self.assertTrue(graph.is_plausible_price(246490, None))

    def test_snippet_price_still_wins_first(self):
        snippet = "HP 15 fd0070TU Buy for ₹52,990 today"
        [c], _, _ = check_prices([candidate(search_snippet=snippet)], lambda prompt: 52990)
        self.assertEqual((c['price'], c['price_source']), (52990, 'search_snippet'))


if __name__ == "__main__":
    unittest.main()
