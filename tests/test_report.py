"""Report ranking (verified in budget > price unknown > over budget, by fit) and the "price unverified" instruction."""
import unittest

from support import FakeGroq

import graph
import prompts
import tools

INSTRUCTION = ('A candidate whose price is null must be labelled "price unverified" and must never be called the best '
               'or the clear top choice; if no candidate has a verified price within budget, say so first, before the '
               'Requirements Summary.')


def cand(name, price, within_budget, fit):
    return {'product_name': name, 'price': price, 'within_budget': within_budget, 'fit_score': fit,
            'known_specs': {}, 'source_url': f'https://www.flipkart.com/{name}/p/itm1'}


# live run 2's top candidates: Lenovo V15 (no price, fit 10) was ranked first and called "the clear top choice"
LIVE_RUN_2 = [cand('HP 15s FY5008TU', None, 'unknown', 9), cand('HP Laptop 15 fd0022TU', 73990, False, 9),
              cand('Lenovo V15', None, 'unknown', 10), cand('HP 15s 15-FD0651TU', 82022, False, 5)]


class RankingTest(unittest.TestCase):
    def names(self, candidates, top_n=3):
        return [c['product_name'] for c in tools.select_report_candidates(candidates, top_n=top_n)]

    def test_groups_in_budget_then_unknown_then_over_budget(self):
        candidates = [cand('over-9', 70000, False, 9), cand('unknown-10', None, 'unknown', 10),
                      cand('in-6', 50000, True, 6), cand('unknown-7', None, 'unknown', 7), cand('in-8', 55000, True, 8)]
        self.assertEqual(self.names(candidates, top_n=5), ['in-8', 'in-6', 'unknown-10', 'unknown-7', 'over-9'])

    def test_live_run_2_order(self):
        self.assertEqual(self.names(LIVE_RUN_2, top_n=4),
                         ['Lenovo V15', 'HP 15s FY5008TU', 'HP Laptop 15 fd0022TU', 'HP 15s 15-FD0651TU'])

    def test_in_budget_beats_higher_fit_unknown(self):
        self.assertEqual(self.names([cand('unknown-10', None, 'unknown', 10), cand('in-5', 40000, True, 5)], top_n=1),
                         ['in-5'])

    def test_missing_fit_score_sorts_last_in_group(self):
        no_fit = {k: v for k, v in cand('no-fit', None, 'unknown', 0).items() if k != 'fit_score'}
        self.assertEqual(self.names([no_fit, cand('unknown-3', None, 'unknown', 3)]), ['unknown-3', 'no-fit'])


class ReportPromptTest(unittest.TestCase):
    def test_instruction_added_once_after_data_rule(self):
        template = prompts.prompt7.template
        self.assertEqual(template.count(INSTRUCTION), 1)
        self.assertIn("not listed in the candidates\n\n" + INSTRUCTION + "\n", template)

    def test_report_prompt_carries_instruction_and_ranked_candidates(self):
        fake = FakeGroq({"text": lambda prompt: "## Report"}).install()
        state = {"requirements": {'category': 'laptop', 'usecase': 'coding', 'budget': 60000,
                                  'non_negotiable_specs': {'ram': '8GB'}, 'negotiable_specs': None},
                 "all_candidates": LIVE_RUN_2}
        try:
            update = graph.report(state, {})
        finally:
            fake.uninstall()
        prompt = fake.calls[0][1]
        self.assertIn(INSTRUCTION, prompt)
        self.assertLess(prompt.index("'Lenovo V15'"), prompt.index("'HP Laptop 15 fd0022TU'"))
        self.assertTrue(update["is_degraded"])
        self.assertEqual([c['product_name'] for c in update["report_candidates"]],
                         ['Lenovo V15', 'HP 15s FY5008TU', 'HP Laptop 15 fd0022TU'])


if __name__ == "__main__":
    unittest.main()
