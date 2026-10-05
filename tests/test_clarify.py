"""
Clarify loop: the repeated question was intake asking again, not the app rendering twice. The app passes the
previous question; intake then assumes a general use case when an answer gives the budget but no use case.
"""
import contextlib
import io
import unittest
from unittest import mock

from support import ROOT, FakeGroq

from streamlit.testing.v1 import AppTest

import agent
import graph

QUESTION = "What will be the primary use for the refrigerator, and what is your budget?"


class AppClarifyTest(unittest.TestCase):
    def run_turns(self, replies, *inputs):
        calls = []
        replies = iter(replies)

        def stream(user_query, previous_question=None):
            calls.append((user_query, previous_question))
            yield next(replies)
        with mock.patch.object(agent, "run_pipeline_stream", stream):
            at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
            at.run()
            for text in inputs:
                at.chat_input[0].set_value(text).run()
        return at, calls

    def test_question_rendered_once_per_turn(self):
        at, _ = self.run_turns([{"type": "final", "status": "clarify", "question": QUESTION},
                                {"type": "final", "status": "clarify", "question": "What is your budget?"}],
                               "I want a refrigerator", "family of four")
        texts = [m.value for m in at.markdown]
        self.assertEqual(texts.count(QUESTION), 1)
        self.assertEqual(texts.count("What is your budget?"), 1)

    def test_previous_question_passed_only_when_answering_one(self):
        done = {"type": "final", "status": "done", "report": "## 1. Requirements Summary", "candidates": [],
                "is_degraded": True}
        _, calls = self.run_turns([{"type": "final", "status": "clarify", "question": QUESTION}, done, done],
                                  "I want a refrigerator", "50000", "I want a laptop for coding under 60000")
        self.assertEqual(calls, [("I want a refrigerator", None),
                                 ("I want a refrigerator. 50000", QUESTION),
                                 ("I want a laptop for coding under 60000", None)])


class IntakeClarifyLoopTest(unittest.TestCase):
    def run_intake(self, call_a, previous_question=None):
        call_b = {"usecase": "general everyday use", "budget": 50000, "category": "refrigerator",
                  "non_negotiable_specs": {"capacity": "at least 250 L"}, "negotiable_specs": None}
        fake = FakeGroq({"Call-A": lambda p: call_a, "Call-B": lambda p: call_b}).install()
        state = {"user_query": "I want a refrigerator. 50000"}
        if previous_question:
            state["previous_question"] = previous_question
        try:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                update = graph.intake(state, {})
        finally:
            fake.uninstall()
        return update, fake, out.getvalue()

    BUDGET_ONLY = {"status": "unclear", "question": QUESTION, "usecase": None, "budget": 50000, "category": "refrigerator"}

    def test_budget_only_answer_proceeds_with_general_use(self):
        update, fake, log = self.run_intake(self.BUDGET_ONLY, previous_question=QUESTION)
        self.assertIsNone(update["clarify_question"])  # cleared, so a checkpointed thread moves on
        self.assertEqual(update["requirements"]["category"], "refrigerator")
        self.assertEqual(fake.names(), ["Call-A", "Call-B"])  # no extra Call A
        self.assertIn("usecase is -> general everyday use", fake.calls[1][1])
        self.assertIn("Clarify loop avoided", log)

    def test_first_turn_still_asks(self):
        update, fake, _ = self.run_intake(self.BUDGET_ONLY)
        self.assertEqual(update, {"clarify_question": QUESTION})
        self.assertEqual(fake.names(), ["Call-A"])

    def test_missing_budget_is_still_asked_again(self):
        no_budget = {**self.BUDGET_ONLY, "budget": None, "usecase": "family", "question": "What is your budget?"}
        update, _, _ = self.run_intake(no_budget, previous_question=QUESTION)
        self.assertEqual(update, {"clarify_question": "What is your budget?"})

    def test_unknown_category_is_still_asked_again(self):
        update, _, _ = self.run_intake({**self.BUDGET_ONLY, "category": None}, previous_question=QUESTION)
        self.assertEqual(update, {"clarify_question": QUESTION})

    def test_clear_query_unchanged(self):
        clear = {"status": "clear", "question": None, "usecase": "family", "budget": 50000, "category": "refrigerator"}
        update, fake, log = self.run_intake(clear, previous_question=QUESTION)
        self.assertIn("requirements", update)
        self.assertIn("usecase is -> family", fake.calls[1][1])
        self.assertNotIn("Clarify loop avoided", log)

    def test_stream_passes_previous_question_into_the_graph(self):
        seen = {}

        class Graph:
            def stream(self, inputs, config=None, stream_mode=None):
                seen.update(inputs)
                yield ("updates", {"intake": {"clarify_question": "What is your budget?"}})
        with mock.patch.object(agent, "_graph", Graph()):
            list(agent.run_pipeline_stream("I want a refrigerator. 50000", previous_question=QUESTION))
        self.assertEqual(seen, {"user_query": "I want a refrigerator. 50000", "previous_question": QUESTION})


if __name__ == "__main__":
    unittest.main()
