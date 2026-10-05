"""Report ranking, the Python-built Final Recommendation headline, and the report prompt's price instructions."""
import contextlib
import io
import unittest

from support import FakeGroq

import graph
import prompts
import tools

INSTRUCTION = ('A candidate whose price is null must be labelled "price unverified" and must never be called the best '
               'or the clear top choice; if no candidate has a verified price within budget, say so first, before the '
               'Requirements Summary.')
HEADLINE_INSTRUCTION = ('Begin the Final Recommendation section with this sentence, copied verbatim: "{recommendation_headline}" '
                        'Then add only supporting detail for it, and never call a candidate without a verified price the '
                        'best, strongest, top or recommended option.')


def cand(name, price, within_budget, fit):
    return {'product_name': name, 'price': price, 'within_budget': within_budget, 'fit_score': fit,
            'known_specs': {}, 'source_url': f'https://www.flipkart.com/{name}/p/itm1'}


# live run 2's top candidates: Lenovo V15 (no price, fit 10) was ranked first and called "the clear top choice"
LIVE_RUN_2 = [cand('HP 15s FY5008TU', None, 'unknown', 9), cand('HP Laptop 15 fd0022TU', 73990, False, 9),
              cand('Lenovo V15', None, 'unknown', 10), cand('HP 15s 15-FD0651TU', 82022, False, 5)]
# live run 3: three verified prices, all over budget; the unpriced Acer (fit 10) was called "the strongest candidate"
LIVE_RUN_3 = [cand('HP Laptop 15 fd0022TU', 73990, False, 7), cand('HP 15 fd0577TU', 60499, False, 8),
              cand('Samsung Galaxy Book 4', 71800, False, 8), cand('HP Latest 15.6" FHD Laptop', None, 'unknown', 9),
              cand('HP 250R G9 A37RKET', None, 'unknown', 9), cand('Acer Aspire 5 15 A515-58P-58FK', None, 'unknown', 10)]


class RankingTest(unittest.TestCase):
    def names(self, candidates, top_n=3):
        return [c['product_name'] for c in tools.select_report_candidates(candidates, top_n=top_n)]

    def test_unknown_price_no_longer_ranks_above_over_budget(self):
        candidates = [cand('over-9', 70000, False, 9), cand('unknown-10', None, 'unknown', 10),
                      cand('in-6', 50000, True, 6), cand('unknown-7', None, 'unknown', 7), cand('in-8', 55000, True, 8)]
        self.assertEqual(self.names(candidates, top_n=5), ['in-8', 'in-6', 'over-9', 'unknown-10', 'unknown-7'])

    def test_live_run_2_order(self):
        self.assertEqual(self.names(LIVE_RUN_2, top_n=4),
                         ['HP Laptop 15 fd0022TU', 'HP 15s 15-FD0651TU', 'Lenovo V15', 'HP 15s FY5008TU'])

    def test_in_budget_beats_higher_fit_unknown(self):
        self.assertEqual(self.names([cand('unknown-10', None, 'unknown', 10), cand('in-5', 40000, True, 5)], top_n=1),
                         ['in-5'])

    def test_missing_fit_score_sorts_last_in_group(self):
        no_fit = {k: v for k, v in cand('no-fit', None, 'unknown', 0).items() if k != 'fit_score'}
        self.assertEqual(self.names([no_fit, cand('unknown-3', None, 'unknown', 3)]), ['unknown-3', 'no-fit'])


class HeadlineTest(unittest.TestCase):
    def test_best_fit_in_budget_named(self):
        candidates = [cand('in-6', 50000, True, 6), cand('in-8', 55000, True, 8), cand('unknown-10', None, 'unknown', 10)]
        self.assertEqual(tools.recommendation_headline(candidates, 60000),
                         "in-8 is the recommended choice: it has the best fit score (8/10) of the candidates with a "
                         "verified price within the ₹60,000 budget, at ₹55,000.")

    def test_live_run_3_closest_over_budget_with_gap(self):
        self.assertEqual(tools.recommendation_headline(LIVE_RUN_3, 60000),
                         "No candidate has a verified price within the ₹60,000 budget; the closest is HP 15 fd0577TU "
                         "at ₹60,499, ₹499 (0.8%) over budget.")

    def test_no_verified_price(self):
        self.assertEqual(tools.recommendation_headline([cand('unknown-10', None, 'unknown', 10)], 60000),
                         "No candidate has a verified price, so none can be recommended as a purchase within the ₹60,000 budget.")

    def test_no_budget(self):
        self.assertEqual(tools.recommendation_headline([cand('a', 132489, True, 7)], None),
                         "a is the recommended choice: it has the best fit score (7/10) of the candidates with a "
                         "verified price, at ₹1,32,489.")
        self.assertEqual(tools.recommendation_headline([], None), "No candidate has a verified price, so none can be recommended as a purchase.")

    def test_format_inr_indian_grouping(self):
        self.assertEqual([tools.format_inr(n) for n in (999, 60000, 132489, 1324890, 60499.0)],
                         ['₹999', '₹60,000', '₹1,32,489', '₹13,24,890', '₹60,499'])


class ReportPromptTest(unittest.TestCase):
    def test_instruction_added_once_after_data_rule(self):
        template = prompts.prompt7.template
        self.assertEqual(template.count(INSTRUCTION), 1)
        self.assertIn("not listed in the candidates\n\n" + INSTRUCTION + "\n", template)

    def test_headline_instruction_added_once(self):
        self.assertEqual(prompts.prompt7.template.count(HEADLINE_INSTRUCTION), 1)
        self.assertIn(INSTRUCTION + "\n" + HEADLINE_INSTRUCTION + "\n", prompts.prompt7.template)

    def run_report(self, candidates, answer):
        fake = FakeGroq({"text": answer}).install()
        state = {"requirements": {'category': 'laptop', 'usecase': 'coding', 'budget': 60000,
                                  'non_negotiable_specs': {'ram': '8GB'}, 'negotiable_specs': None},
                 "all_candidates": candidates}
        try:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                update = graph.report(state, {})
        finally:
            fake.uninstall()
        return update, fake.calls[0][1], out.getvalue()

    def test_report_prompt_carries_instructions_and_headline(self):
        headline = tools.recommendation_headline(LIVE_RUN_3, 60000)
        update, prompt, log = self.run_report(LIVE_RUN_3, lambda p: f"## 5. Final Recommendation\n{headline} More.")
        self.assertIn(INSTRUCTION, prompt)
        self.assertIn(HEADLINE_INSTRUCTION.replace("{recommendation_headline}", headline), prompt)
        self.assertTrue(update["is_degraded"])
        self.assertNotIn("Report check", log)

    def test_missing_headline_is_logged(self):
        _, _, log = self.run_report(LIVE_RUN_3, lambda p: "## 5. Final Recommendation\nAcer is the strongest candidate.")
        self.assertIn("Report check: the Final Recommendation headline was not used verbatim", log)


if __name__ == "__main__":
    unittest.main()
