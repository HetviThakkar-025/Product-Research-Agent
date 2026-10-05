"""Call B sees the user's own words, keeps stated values, and never keeps the budget as a spec."""
import contextlib
import io
import unittest

from support import FakeGroq, FakeTavily

import graph
import prompts
import test_token_load
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

    def test_schema_allows_null_spec_values(self):
        props = prompts.call_b["properties"]
        self.assertEqual(props["non_negotiable_specs"]["additionalProperties"], {"type": ["string", "null"]})
        self.assertEqual(props["negotiable_specs"]["additionalProperties"], {"type": ["string", "null"]})

    def test_null_specs_dropped_at_intake(self):
        # the live error: /negotiable_specs/brand, color, additional_features were null
        call_b = {"usecase": "family use", "budget": 50000, "category": "refrigerator",
                  "non_negotiable_specs": {"capacity": "at least 250 L", "type": "double door", "warranty": None},
                  "negotiable_specs": {"brand": None, "color": None, "additional_features": None, "inverter": " ",
                                       "energy_rating": "5 star preferred"}}
        update, _, log = run_intake(call_b)
        self.assertEqual(update["requirements"]["non_negotiable_specs"], {"capacity": "at least 250 L", "type": "double door"})
        self.assertEqual(update["requirements"]["negotiable_specs"], {"energy_rating": "5 star preferred"})
        self.assertIn("Dropped spec(s) without a value from negotiable_specs: "
                      "['additional_features', 'brand', 'color', 'inverter']", log)

    def test_graph_continues_with_null_specs(self):
        handlers, search = test_token_load.answers()
        handlers["Call-B"] = lambda p: {
            "usecase": "coding", "budget": 60000, "category": "laptop",
            "non_negotiable_specs": {"processor": "i5 12th Gen", "ram": "8GB", "gpu": None},
            "negotiable_specs": {"brand": None, "color": None, "additional_features": None, "battery": "6 hours"}}
        fake = FakeGroq(handlers).install()
        tavily = FakeTavily(search=search).install()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                final = graph.build_graph().invoke({"user_query": "laptop for coding under 60000"},
                                                   config=graph.make_config())
        finally:
            fake.uninstall()
            tavily.uninstall()
        self.assertEqual(final["requirements"]["non_negotiable_specs"], {"processor": "i5 12th Gen", "ram": "8GB"})
        self.assertEqual(final["requirements"]["negotiable_specs"], {"battery": "6 hours"})
        self.assertIn("report", final)
        rewrite_prompts = [prompt for name, prompt in fake.calls if name == "text" and "Shorten them" in prompt]
        self.assertTrue(rewrite_prompts)
        for prompt in rewrite_prompts:
            specs_line = prompt.split("\n", 1)[0]  # "Given these required specs: {...}"
            self.assertNotIn("None", specs_line)
            self.assertNotIn("brand", specs_line)

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
