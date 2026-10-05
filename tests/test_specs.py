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

    def test_no_negotiable_specs(self):
        self.assertEqual(graph.spec_key_names({'non_negotiable_specs': {'a': '1'}, 'negotiable_specs': None}), ['a'])


if __name__ == "__main__":
    unittest.main()
