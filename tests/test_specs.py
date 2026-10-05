"""Spec keys: Calls C and D are told Call B's key names, and specs_found compares keys case-insensitively."""
import unittest

from support import FakeGroq, FakeTavily

import graph
import prompts

REQUIREMENTS = {
    'category': 'laptop', 'usecase': 'coding', 'budget': 60000,
    'non_negotiable_specs': {'processor': 'Intel Core i5 12th Gen', 'ram': '8GB', 'storage': '512GB SSD',
                             'display': '15.6" FHD', 'keyboard': 'backlit'},
    'negotiable_specs': {'battery': '6+ hours', 'weight': 'under 1.8 kg'},
}
URL = "https://www.amazon.in/HP-i5-1235U-Anti-Glare-Micro-Edge-15-6-inch/dp/B0DCG26YC5"
SPEC_SITE_RESULT = {"results": [{"url": "https://www.smartprix.com/laptops/hp-15-fd0070tu", "title": "HP 15 fd0070TU specs",
                                 "content": "Backlit keyboard, 15.6 inch FHD display", "score": 0.9}]}


def verify(known_specs, new_specs):
    fake = FakeGroq({"Spec-Merge": lambda prompt: {"new_specs": new_specs}}).install()
    tavily = FakeTavily(search=lambda query, domains: SPEC_SITE_RESULT).install()
    try:
        state = {"requirements": REQUIREMENTS, "new_candidates": [
            {"product_name": "HP 15 fd0070TU", "known_specs": dict(known_specs), "specs_found": False, "source_url": URL}]}
        return graph.verify_specs(state, {})["new_candidates"][0], fake
    finally:
        fake.uninstall()
        tavily.uninstall()


class SpecsFoundTest(unittest.TestCase):
    def test_keys_match_case_insensitively(self):
        candidate, _ = verify({'Processor': 'i5-1235U', 'RAM': '8GB', 'Storage': '512GB SSD', 'Display': '15.6" FHD'},
                              {'Keyboard': 'Backlit'})
        self.assertTrue(candidate['specs_found'])

    def test_missing_required_spec_is_still_not_found(self):
        candidate, _ = verify({'processor': 'i5-1235U', 'ram': '8GB', 'storage': '512GB SSD', 'display': '15.6" FHD'}, {})
        self.assertFalse(candidate['specs_found'])

    def test_differently_named_keys_still_do_not_count(self):
        # only case is forgiven; synonyms are left to the prompt sentence
        candidate, _ = verify({'CPU': 'i5-1235U', 'RAM': '8GB', 'SSD': '512GB', 'Screen': '15.6"'}, {'keyboard': 'Backlit'})
        self.assertFalse(candidate['specs_found'])

    def test_merge_compares_keys_case_insensitively_keeping_first_casing(self):
        candidate, _ = verify({'processor': 'i5-1235U'}, {'Processor': 'i3-1215U', 'Keyboard': 'Backlit', 'keyboard': 'Full size'})
        self.assertEqual(candidate['known_specs'], {'processor': 'i5-1235U', 'Keyboard': 'Backlit'})

    def test_merge_never_overwrites_known_spec(self):
        candidate, _ = verify({'processor': 'i5-1235U'}, {'processor': 'i3-1215U', 'keyboard': 'Backlit'})
        self.assertEqual(candidate['known_specs']['processor'], 'i5-1235U')
        self.assertEqual(candidate['known_specs']['keyboard'], 'Backlit')


class SpecKeyPromptTest(unittest.TestCase):
    KEYS = "['processor', 'ram', 'storage', 'display', 'keyboard', 'battery', 'weight']"

    def test_call_d_prompt_lists_call_b_keys(self):
        _, fake = verify({'processor': 'i5-1235U'}, {})
        name, prompt = fake.calls[0]
        self.assertEqual(name, "Spec-Merge")
        self.assertIn(f"Use exactly these key names in new_specs for any of these specs you find: {self.KEYS}.", prompt)

    def test_call_c_prompt_lists_call_b_keys(self):
        fake = FakeGroq({"Candidate-Extraction": lambda prompt: {"candidates": []}}).install()
        try:
            state = {"requirements": REQUIREMENTS, "all_candidates": [], "search_snippets": {},
                     "search_results": {"results": [{"url": URL, "title": "HP 15", "content": "i5-1235U 8GB"}]}}
            graph.extract_candidates(state, {})
        finally:
            fake.uninstall()
        name, prompt = fake.calls[0]
        self.assertEqual(name, "Candidate-Extraction")
        self.assertIn(f"Use exactly these key names in known_specs for any of these specs you find: {self.KEYS}.", prompt)

    def test_only_one_sentence_added_to_each_prompt(self):
        sentence_c = " Use exactly these key names in known_specs for any of these specs you find: {spec_keys}."
        sentence_d = " Use exactly these key names in new_specs for any of these specs you find: {spec_keys}."
        self.assertEqual(prompts.prompt3.template.count("{spec_keys}"), 1)
        self.assertEqual(prompts.prompt4.template.count("{spec_keys}"), 1)
        self.assertIn(sentence_c, prompts.prompt3.template)
        self.assertIn(sentence_d, prompts.prompt4.template)

    def test_call_b_prompt_has_checkable_specs_sentence_once(self):
        sentence = ("Only mark a spec non-negotiable if product listings state it as a concrete, checkable value "
                    "(e.g. processor, RAM, storage, display size/resolution); put subjective or rarely listed qualities "
                    "(keyboard feel, build quality, speakers) under negotiable specs, and keep non-negotiable specs to at most 4.")
        self.assertEqual(prompts.prompt2.template.count(sentence), 1)
        self.assertIn("which can be ignored if there are budget constraints.\n" + sentence + "\n", prompts.prompt2.template)

    def test_call_b_prompt_has_minimum_threshold_sentence_once(self):
        sentence = ("State non-negotiable specs as minimum thresholds or classes (for example 'RAM at least 8 GB', "
                    "'Intel Core i5 or Ryzen 5 or better'), never as specific model numbers or the highest configuration.")
        self.assertEqual(prompts.prompt2.template.count(sentence), 1)
        self.assertIn("keep non-negotiable specs to at most 4.\n" + sentence + "\n", prompts.prompt2.template)

    def test_call_b_prompt_has_user_stated_specs_sentence_once(self):
        sentence = ("Do not make a spec non-negotiable unless the user's request states it or the use case clearly "
                    "requires it; for a general request, use broad thresholds and put everything else under negotiable.")
        self.assertEqual(prompts.prompt2.template.count(sentence), 1)
        self.assertIn("or the highest configuration.\n" + sentence + "\n", prompts.prompt2.template)

    def test_call_b_runs_at_temperature_0_and_others_unchanged(self):
        # langchain-groq sends temperature=0 as 1e-8 (Groq's own handling of 0); it is in every Call B request body
        call_b_model = prompts.str_model_call_b.first.bound
        self.assertEqual(call_b_model._default_params["temperature"], 1e-8)
        self.assertEqual((call_b_model.max_tokens, call_b_model.max_retries), (1200, 0))
        for name in ("str_model_call_a", "str_model_call_c", "str_model_call_d", "str_model_call_e", "str_model_call_f"):
            self.assertEqual(getattr(prompts, name).first.bound._default_params["temperature"], 0.7, name)
        self.assertEqual(prompts.llm_report._default_params["temperature"], 0.7)

    def test_no_negotiable_specs(self):
        self.assertEqual(graph.spec_key_names({'non_negotiable_specs': {'a': '1'}, 'negotiable_specs': None}), ['a'])


if __name__ == "__main__":
    unittest.main()
