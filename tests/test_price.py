"""
Price attribution: product match for fallback sources, model numbers kept from dropped duplicates and
the own result title, ₹ amounts kept only near the candidate's own model/name, and the 0.25x-3x budget bound. Uses the real Tavily fallback response for the HP 15
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

    def test_own_result_title_models_stored(self):
        state_title = "HP 15, 12 Gen Intel Core i5-1235U ... Backlit KB fd0070TU : Amazon.in: Electronics"
        fake = FakeGroq({"Candidate-Extraction": lambda prompt: {"candidates": [
            {"product_name": HP15_NAME, "source_url": HP15_URL, "known_specs": {"cpu": "i5"}, "specs_found": False}]}}).install()
        state = {"requirements": {'non_negotiable_specs': {'processor': 'i5'}, 'negotiable_specs': None},
                 "all_candidates": [], "search_snippets": {},
                 "search_results": {"results": [{"url": HP15_URL, "title": state_title, "content": ""}]}}
        try:
            update = graph.extract_candidates(state, {})
        finally:
            fake.uninstall()
        self.assertEqual(update["new_candidates"][0]['model_numbers'], ['fd0070tu'])

    def test_dropped_duplicate_models_stored_within_same_batch(self):
        update = self.run_extract([], [
            {"product_name": HP15_NAME, "source_url": HP15_URL, "known_specs": {"cpu": "i5"}, "specs_found": False},
            {"product_name": "HP 15 fd0070TU", "source_url": HP15_URL, "known_specs": {"cpu": "i5"}, "specs_found": False}])
        self.assertEqual(len(update["new_candidates"]), 1)
        self.assertEqual(update["new_candidates"][0]['model_numbers'], ['fd0070tu'])


class AttributePricesTest(unittest.TestCase):
    def amounts(self, text, **cand):
        kept, rejected = tools.attribute_prices(candidate(**cand), text)
        return [tools.rupee_amounts(m.group()).pop() for m in kept], sorted(set(rejected))

    def test_own_model_nearby_kept(self):
        self.assertEqual(self.amounts("HP 15 fd0070TU Buy for ₹52,990", model_numbers=['fd0070tu']), ([52990], []))

    def test_other_model_nearby_rejected(self):
        self.assertEqual(self.amounts("HP 15s fy5007TU ₹2,46,490", model_numbers=['fd0070tu']),
                         ([], [(246490, 'other model nearby (fy5007tu)')]))

    def test_nearest_preceding_model_decides(self):
        text = "HP 15 fd0070TU ₹52,990 | Similar: HP 15s fy5007TU ₹2,46,490"
        self.assertEqual(self.amounts(text, model_numbers=['fd0070tu']),
                         ([52990], [(246490, 'other model nearby (fy5007tu)')]))

    def test_name_prefix_nearby_kept_when_no_model_in_window(self):
        self.assertEqual(self.amounts(f"{HP15_NAME}(39.6cm) Laptop, Silver Deal price ₹49,990"), ([49990], []))

    def test_nothing_nearby_rejected(self):
        self.assertEqual(self.amounts("Thin & Light Business/15.6\" -53%₹47,880.00"),
                         ([], [(47880, 'product not named nearby')]))

    def test_own_page_id_counts_as_own_model(self):
        self.assertEqual(self.amounts("/dp/B0DCG26YC5 ₹52,990"), ([52990], []))

    def test_without_known_model_any_model_nearby_is_other(self):
        self.assertEqual(self.amounts("HP 15 fd0070TU ₹52,990", url="https://www.flipkart.com/hp/p/itm1"),
                         ([], [(52990, 'other model nearby (fd0070tu)')]))

    def test_real_carousel_page_yields_no_amount(self):
        content = load_fixture("tavily_fallback_hp15.json")["results"][0]["content"]
        kept, rejected = tools.attribute_prices(candidate(model_numbers=['fd0070tu']), content)
        self.assertEqual(kept, [])
        self.assertIn((246490, 'other model nearby (fy5007tu)'), rejected)
        self.assertIn((47880, 'product not named nearby'), rejected)


class CheckPricesTest(unittest.TestCase):
    def test_carousel_prices_on_own_page_rejected(self):
        # the live run's case: the fallback hit the candidate's own page, whose content is a carousel of other laptops
        for extra in ({}, {'model_numbers': ['fd0070tu']}):
            [c], log, fake = check_prices([candidate(**extra)], lambda prompt: 47880)
            self.assertIn(f"Rejected price: other model nearby (fy5007tu) for {HP15_NAME}: 246490 (fallback_search: {HP15_URL})", log)
            self.assertIn(f"Rejected price: product not named nearby for {HP15_NAME}: 47880 (fallback_search: {HP15_URL})", log)
            self.assertNotIn("Price-Extraction", fake.names())
            self.assertIsNone(c['price'])
            self.assertEqual(c['within_budget'], "unknown")

    def test_implausible_price_rejected(self):
        snippet = "HP 15 fd0070TU Buy for ₹2,46,490"
        [c], log, _ = check_prices([candidate(search_snippet=snippet, model_numbers=['fd0070tu'])], lambda prompt: 246490)
        self.assertIn("Rejected implausible price 246490 for HP 15", log)
        self.assertIsNone(c['price'])

    def test_call_e_cannot_pick_a_rejected_amount(self):
        snippet = "HP 15 fd0070TU ₹52,990 | Similar: HP 15s fy5007TU ₹54,990"
        [c], log, _ = check_prices([candidate(search_snippet=snippet, model_numbers=['fd0070tu'])], lambda prompt: 54990)
        self.assertIn("Rejected price 54990 for HP 15", log)
        self.assertNotEqual(c['price_source'], 'search_snippet')

    def test_page_extract_uses_full_page_text(self):
        page = f"{HP15_NAME}(39.6cm) Laptop\nVisit the HP Store\nDeal price ₹49,990 Inclusive of all taxes"
        fake = FakeGroq({"Price-Extraction": lambda prompt: {"price": 49990, "availability": "in_stock"}}).install()
        tavily = FakeTavily(extract=lambda urls: {"results": [{"url": urls[0], "raw_content": page}]}).install()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                [c] = graph.check_prices({"requirements": REQUIREMENTS, "new_candidates": [candidate()]}, {})["new_candidates"]
        finally:
            fake.uninstall()
            tavily.uninstall()
        self.assertEqual((c['price'], c['price_source']), (49990, 'page_extract'))

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
        [c], _, _ = check_prices([candidate(search_snippet=snippet, model_numbers=['fd0070tu'])], lambda prompt: 52990)
        self.assertEqual((c['price'], c['price_source']), (52990, 'search_snippet'))


if __name__ == "__main__":
    unittest.main()
