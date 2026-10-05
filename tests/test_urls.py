"""URL checks: a bare domain or path-less URL is never a product page or a match for a search result."""
import contextlib
import io
import unittest

import support  # noqa: F401  (network guard)

import tools

RESULTS = {"results": [{"url": "https://www.vijaysales.com/p/257856/hp-15s-standard-laptop-15-fd0651tu"},
                       {"url": "https://www.amazon.in/HP-15-Laptop/dp/B0DCG26YC5"}]}


def filter_sources(*sources, results=RESULTS):
    candidates = [{"product_name": f"c{i}", "source_url": s} for i, s in enumerate(sources)]
    with contextlib.redirect_stdout(io.StringIO()) as out:
        kept = tools.filter_hallucinated_candidates(candidates, results)
    return [c["source_url"] for c in kept], out.getvalue()


class ProductPageUrlTest(unittest.TestCase):
    def test_bare_domains_are_not_product_pages(self):
        for url in ("https://www.vijaysales.com", "https://www.vijaysales.com/", "www.vijaysales.com",
                    "https://www.amazon.in/?ref=nav_logo", "https://www.flipkart.com#top"):
            self.assertFalse(tools.is_product_page_url(url), url)

    def test_product_pages_still_accepted(self):
        for url in ("https://www.vijaysales.com/p/257856/hp-15s-standard-laptop-15-fd0651tu",
                    "https://www.amazon.in/HP-15-Laptop/dp/B0DCG26YC5",
                    "https://www.flipkart.com/hp-15s/p/itm7efcb2faf35a3"):
            self.assertTrue(tools.is_product_page_url(url), url)

    def test_listing_pages_still_rejected(self):
        self.assertFalse(tools.is_product_page_url("https://www.amazon.in/i5-12th-gen-laptops/s"))


class HallucinationFilterTest(unittest.TestCase):
    def test_bare_domain_source_does_not_match_result_on_that_site(self):
        # the live run's case: Call C gave https://www.vijaysales.com as the source
        kept, log = filter_sources("https://www.vijaysales.com")
        self.assertEqual(kept, [])
        self.assertIn("Dropped hallucinated candidate: c0 (fake source: https://www.vijaysales.com)", log)

    def test_bare_domain_result_does_not_match_any_source(self):
        results = {"results": [{"url": "https://www.croma.com"}]}
        kept, _ = filter_sources("https://www.croma.com/hp-15-laptop/p/123456", results=results)
        self.assertEqual(kept, [])

    def test_real_product_url_still_matches(self):
        kept, _ = filter_sources("https://www.amazon.in/HP-15-Laptop/dp/B0DCG26YC5",
                                 "https://www.vijaysales.com/p/257856/hp-15s-standard-laptop-15-fd0651tu")
        self.assertEqual(len(kept), 2)

    def test_has_path(self):
        self.assertFalse(tools.has_path("https://www.vijaysales.com/"))
        self.assertTrue(tools.has_path("https://www.vijaysales.com/p/1"))


if __name__ == "__main__":
    unittest.main()
