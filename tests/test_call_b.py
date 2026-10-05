"""Call B sees the user's own words, keeps stated values, and never keeps the budget as a spec."""
import contextlib
import io
import unittest

from support import FakeGroq

import graph
import prompts
import tools

CONVERSATION = "refrigerator under 50000. double door frost free refrigerator 250 litre 3 star under 50000"
KEEP_SENTENCE = ("Keep every value the user's request states exactly as given and make it non-negotiable (for example "
                 "'250 litre' stays 'at least 250 L' and 'double door' stays type 'double door'; 'frost free' is a cooling "
                 "feature, not a type); the threshold wording above applies only to specs the user did not state, and the "
                 "budget is never a spec.")
CALL_A = {"status": "clear", "question": None, "usecase": "family", "budget": 50000, "category": "refrigerator"}


def run_intake(call_b):
    fake = FakeGroq({"Call-A": lambda p: CALL_A, "Call-B": lambda p: call_b}).install()
    try:
        with contextlib.redirect_stdout(io.StringIO()) as out:
            update = graph.intake({"user_query": CONVERSATION}, {})
    finally:
        fake.uninstall()
    return update, fake, out.getvalue()


class CallBRequestTest(unittest.TestCase):
    def test_prompt2_has_request_line_and_keep_sentence_once(self):
        template = prompts.prompt2.template
        self.assertIn("usecase is -> {usecase}\nUser's request, in their own words: {request}\n", template)
        self.assertEqual(template.count(KEEP_SENTENCE), 1)
        self.assertIn("put everything else under negotiable.\n" + KEEP_SENTENCE + "\n", template)
        self.assertIn("request", prompts.prompt2.input_variables)

    def test_intake_sends_users_words_to_call_b(self):
        call_b = {"usecase": "family", "budget": 50000, "category": "refrigerator",
                  "non_negotiable_specs": {"capacity": "at least 250 L", "type": "double door"}, "negotiable_specs": None}
        update, fake, _ = run_intake(call_b)
        self.assertEqual(fake.names(), ["Call-A", "Call-B"])
        self.assertIn(f"User's request, in their own words: {CONVERSATION}", fake.calls[1][1])
        self.assertEqual(update["requirements"]["non_negotiable_specs"], {"capacity": "at least 250 L", "type": "double door"})

    def test_budget_never_kept_as_a_spec(self):
        call_b = {"usecase": "family", "budget": 50000, "category": "refrigerator",
                  "non_negotiable_specs": {"capacity": "at least 250 L", "Budget": "under 50000"},
                  "negotiable_specs": {"price_range": "40000-50000", "color": "any"}}
        update, _, log = run_intake(call_b)
        self.assertEqual(update["requirements"]["non_negotiable_specs"], {"capacity": "at least 250 L"})
        self.assertEqual(update["requirements"]["negotiable_specs"], {"color": "any"})
        self.assertIn("Dropped budget-like spec(s) from non_negotiable_specs: ['Budget']", log)

    def test_drop_budget_specs_keeps_none_and_other_fields(self):
        cleaned = tools.drop_budget_specs({"category": "tv", "budget": 30000, "non_negotiable_specs": {"size": "43 inch"},
                                           "negotiable_specs": None})
        self.assertEqual(cleaned, {"category": "tv", "budget": 30000, "non_negotiable_specs": {"size": "43 inch"},
                                   "negotiable_specs": None})

    def test_final_chain_still_works_and_passes_the_query(self):
        call_b = {"usecase": "family", "budget": 50000, "category": "refrigerator",
                  "non_negotiable_specs": {"capacity": "at least 250 L"}, "negotiable_specs": None}
        fake = FakeGroq({"Call-A": lambda p: CALL_A, "Call-B": lambda p: call_b}).install()
        try:
            result = prompts.final_chain.invoke({"query": CONVERSATION})
        finally:
            fake.uninstall()
        self.assertEqual(result["non_negotiable_specs"], {"capacity": "at least 250 L"})
        self.assertIn(f"User's request, in their own words: {CONVERSATION}", fake.calls[1][1])


if __name__ == "__main__":
    unittest.main()
