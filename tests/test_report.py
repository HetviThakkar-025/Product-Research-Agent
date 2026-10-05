"""Report ranking, the Python-built Final Recommendation headline, and the report prompt's price instructions."""
import contextlib
import io
import unittest

from support import FakeGroq

import graph
import prompts
import tools

INSTRUCTION = ('A candidate whose price is null must be labelled "price unverified" and must never be called the best '
               'or the clear top choice.')
START_INSTRUCTION = "Start the report directly with section 1, the Requirements Summary; write nothing before it."
HEADLINE_INSTRUCTION = ('Begin the Final Recommendation section with this sentence, copied verbatim: "{recommendation_headline}" '
                        'Then add only supporting detail for it, and never call a candidate without a verified price the '
                        'best, strongest, top or recommended option.')


def cand(name, price, within_budget, fit, specs_found=True, missing=None):
    return {'product_name': name, 'price': price, 'within_budget': within_budget, 'fit_score': fit,
            'specs_found': specs_found, 'missing_or_weak_specs': missing or [],
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

    def test_live_run_3_verified_prices_first_by_gap_then_unpriced_max_4(self):
        self.assertEqual([c['product_name'] for c in tools.select_report_candidates(LIVE_RUN_3)],
                         ['HP 15 fd0577TU', 'Samsung Galaxy Book 4', 'HP Laptop 15 fd0022TU', 'Acer Aspire 5 15 A515-58P-58FK'])

    def test_over_budget_ordered_by_gap_not_fit(self):
        candidates = [cand('over-far-fit-10', 90000, False, 10), cand('over-near-fit-3', 61000, False, 3)]
        self.assertEqual(self.names(candidates), ['over-near-fit-3', 'over-far-fit-10'])

    def test_in_budget_by_fit_then_price(self):
        candidates = [cand('in-7-55k', 55000, True, 7), cand('in-9', 59000, True, 9), cand('in-7-45k', 45000, True, 7)]
        self.assertEqual(self.names(candidates), ['in-9', 'in-7-45k', 'in-7-55k'])

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

    def test_fit_9_with_verified_in_budget_price_is_recommended(self):
        self.assertEqual(tools.recommendation_headline([cand('Laptop A', 54990, True, 9)], 60000),
                         "Laptop A is the recommended choice: it has the best fit score (9/10) of the candidates with a "
                         "verified price within the ₹60,000 budget, at ₹54,990.")

    def test_fit_2_with_verified_in_budget_price_is_not_recommended(self):
        lg = cand('LG 446 L 1 Star', 45990, True, 2,
                  missing=['Energy rating: 1 Star (required ≥ 3 Star)', 'Voltage: not listed'])
        self.assertEqual(tools.recommendation_headline([lg], 60000),
                         "No candidate meets your requirements within the budget. Closest verified price: LG 446 L 1 Star "
                         "at ₹45,990, but it fails energy rating.")

    def test_mixed_qualified_beats_unqualified_with_lower_price(self):
        candidates = [cand('cheap-fit-3', 30000, True, 3, missing=['RAM: 4GB (required 8GB)']),
                      cand('good-fit-8', 58000, True, 8), cand('unpriced-fit-10', None, 'unknown', 10)]
        self.assertTrue(tools.recommendation_headline(candidates, 60000).startswith("good-fit-8 is the recommended choice"))

    def test_mixed_specs_not_found_blocks_recommendation(self):
        candidates = [cand('fit-9-specs-missing', 50000, True, 9, specs_found=False),
                      cand('fit-6', 52000, True, 6, missing=['Display: HD only (required FHD)'])]
        self.assertEqual(tools.recommendation_headline(candidates, 60000),
                         "No candidate meets your requirements within the budget. Closest verified price: "
                         "fit-9-specs-missing at ₹50,000, but it fails required specs that could not all be confirmed.")

    def test_mixed_unqualified_in_budget_and_qualified_over_budget(self):
        candidates = [cand('in-fit-4', 45000, True, 4, missing=['Storage: 1TB HDD (required 512GB SSD)']),
                      cand('over-fit-9', 64000, False, 9)]
        self.assertEqual(tools.recommendation_headline(candidates, 60000),
                         "No candidate meets your requirements within the budget. Closest verified price: in-fit-4 at "
                         "₹45,000, but it fails storage.")

    def test_fit_below_threshold_without_listed_gaps(self):
        self.assertTrue(tools.recommendation_headline([cand('a', 50000, True, 6)], 60000).endswith(
            "but it fails the fit threshold (fit score 6/10)."))

    def test_live_run_3_closest_over_budget_with_gap(self):
        self.assertEqual(tools.recommendation_headline(LIVE_RUN_3, 60000),
                         "No candidate has a verified price within the ₹60,000 budget; the closest is HP 15 fd0577TU "
                         "at ₹60,499, just over budget by ₹499 (0.8%).")

    def test_closest_far_over_budget_gives_plain_gap(self):
        self.assertEqual(tools.recommendation_headline([cand('far', 73990, False, 9)], 60000),
                         "No candidate has a verified price within the ₹60,000 budget; the closest is far at ₹73,990, "
                         "₹13,990 (23.3%) over budget.")

    def test_budget_note_only_under_5_percent(self):
        self.assertEqual(tools.budget_note(cand('a', 62999, False, 5), 60000), "just over budget by ₹2,999 (5.0%)")
        self.assertIsNone(tools.budget_note(cand('b', 63000, False, 5), 60000))
        self.assertIsNone(tools.budget_note(cand('c', 59000, True, 5), 60000))
        self.assertIsNone(tools.budget_note(cand('d', None, 'unknown', 5), 60000))
        self.assertIsNone(tools.budget_note(cand('e', 61000, True, 5), None))

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
        self.assertIn(INSTRUCTION + "\n" + START_INSTRUCTION + "\n" + HEADLINE_INSTRUCTION + "\n", prompts.prompt7.template)

    def test_no_warning_before_requirements_summary_instruction(self):
        self.assertNotIn("say so first", prompts.prompt7.template)
        self.assertEqual(prompts.prompt7.template.count(START_INSTRUCTION), 1)

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
        update, prompt, log = self.run_report(
            LIVE_RUN_3, lambda p: f"## 1. Requirements Summary\nLaptop.\n## 5. Final Recommendation\n{headline} More.")
        self.assertIn(INSTRUCTION, prompt)
        self.assertIn(HEADLINE_INSTRUCTION.replace("{recommendation_headline}", headline), prompt)
        self.assertTrue(update["is_degraded"])
        self.assertNotIn("Report check", log)
        self.assertEqual(len(update["report_candidates"]), 4)
        self.assertIn("'budget_note': 'just over budget by ₹499 (0.8%)'", prompt)
        self.assertIn("'product_name': 'HP 15 fd0577TU'", prompt)
        self.assertNotIn("HP 250R G9", prompt)

    def test_headline_elsewhere_than_section_5_start_is_logged(self):
        headline = tools.recommendation_headline(LIVE_RUN_3, 60000)
        _, _, log = self.run_report(LIVE_RUN_3, lambda p: (
            f"## 1. Requirements Summary\n{headline}\n## 5. Final Recommendation\nAcer is the strongest candidate."))
        self.assertIn("Report check: section 5 does not start with the headline sentence verbatim", log)

    def test_live_check_1_shape_is_logged_and_trimmed(self):
        # live app check 1: headline repeated above section 1, section 5 paraphrased it
        headline = tools.recommendation_headline(LIVE_RUN_3, 60000)
        report = (f"**{headline}**\n\n## 1. Requirements Summary\nFridge.\n\n## 5. Final Recommendation\n"
                  "Given that the only candidate lacks a verified price, no product can be recommended.")
        update, _, log = self.run_report(LIVE_RUN_3, lambda p: report)
        self.assertIn("Report check: text before section 1 was dropped", log)
        self.assertIn("Report check: section 5 does not start with the headline sentence verbatim", log)
        self.assertTrue(update["report"].startswith("## 1. Requirements Summary"))
        self.assertEqual(update["report"].count(headline), 0)


class ReportIssuesTest(unittest.TestCase):
    HEADLINE = "No candidate has a verified price, so none can be recommended as a purchase within the ₹60,000 budget."

    def issues(self, text):
        return tools.report_issues(text, self.HEADLINE)

    def test_well_formed_report_has_no_issues(self):
        for section_5 in (f"## 5. Final Recommendation\n\n{self.HEADLINE} Detail.",
                          f"## 5. Final Recommendation\n> **{self.HEADLINE}**\n\nDetail.",
                          f"**5. Final Recommendation** — {self.HEADLINE}",
                          f"### 5. **Final Recommendation**\n{self.HEADLINE}"):
            self.assertEqual(self.issues(f"## 1. Requirements Summary\nx\n\n{section_5}"), [], section_5)

    def test_missing_sections(self):
        self.assertEqual(self.issues("Just text."), ["no section 1 (Requirements Summary) heading found",
                                                     "no section 5 (Final Recommendation) heading found"])

    def test_trim_to_section_1(self):
        self.assertEqual(tools.trim_to_section_1("**Warning**\n\n## 1. Requirements Summary\nx"), "## 1. Requirements Summary\nx")
        self.assertEqual(tools.trim_to_section_1("no sections here"), "no sections here")


if __name__ == "__main__":
    unittest.main()
