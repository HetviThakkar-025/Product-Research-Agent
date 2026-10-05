"""app.py with a fake run_pipeline_stream (Streamlit AppTest, no API calls): report, clarify and error paths."""
import unittest
from unittest import mock

from support import ROOT

from streamlit.testing.v1 import AppTest

import agent
from tools import DailyQuotaExceeded

APP = str(ROOT / "app.py")


def fake_stream(*events, error=None):
    def run_pipeline_stream(user_query):
        yield from events
        if error:
            raise error
    return run_pipeline_stream


def run_app(stream, query="I want a laptop for coding under 60000"):
    with mock.patch.object(agent, "run_pipeline_stream", stream):
        at = AppTest.from_file(APP, default_timeout=30)
        at.run()
        at.chat_input[0].set_value(query).run()
    return at


def markdown(at):
    return [m.value for m in at.markdown]


class AppStreamTest(unittest.TestCase):
    def test_report_path_shows_progress_headline_then_report(self):
        at = run_app(fake_stream(
            {"type": "progress", "text": "Iteration 1/4 — 0 qualified so far"},
            {"type": "progress", "text": "Generating final report..."},
            {"type": "headline", "text": "No candidate has a verified price."},
            {"type": "token", "text": "## Report "},
            {"type": "token", "text": "body"},
            {"type": "final", "status": "done", "report": "## Report body", "candidates": [], "is_degraded": True},
        ))
        self.assertFalse(at.exception)
        texts = markdown(at)
        self.assertIn("Iteration 1/4 — 0 qualified so far", texts)
        headline = texts.index("**No candidate has a verified price.**")
        self.assertLess(texts.index("Generating final report..."), headline)
        self.assertLess(headline, texts.index("## Report body"))
        self.assertEqual(at.status[0].label, "Research complete")
        self.assertEqual(at.status[0].state, "complete")
        stored = at.session_state.messages[-1]["content"]
        self.assertTrue(stored.startswith("**No candidate has a verified price.**\n\n## Report body"))
        self.assertFalse(at.session_state.awaiting_clarification)

    def test_clarify_path_returns_question_without_report(self):
        at = run_app(fake_stream({"type": "final", "status": "clarify", "question": "What is your budget?"}),
                     query="I want a fridge")
        self.assertFalse(at.exception)
        self.assertIn("What is your budget?", markdown(at))
        self.assertEqual(at.status[0].label, "Need one more detail")
        self.assertTrue(at.session_state.awaiting_clarification)
        self.assertEqual(at.session_state.messages[-1]["content"], "What is your budget?")

    def test_error_path_friendly_message_and_error_status(self):
        with mock.patch("traceback.print_exc") as print_exc:
            at = run_app(fake_stream({"type": "progress", "text": "Searching Indian retail sites..."},
                                     error=RuntimeError("boom")))
        self.assertFalse(at.exception)
        print_exc.assert_called_once()
        self.assertEqual(at.status[0].state, "error")
        self.assertIn("Sorry, something went wrong while researching this — please try again in a moment.", markdown(at))

    def test_daily_quota_path(self):
        at = run_app(fake_stream(error=DailyQuotaExceeded("daily limit")))
        self.assertEqual(at.status[0].state, "error")
        self.assertTrue(any("free usage limit" in m for m in markdown(at)))


if __name__ == "__main__":
    unittest.main()
