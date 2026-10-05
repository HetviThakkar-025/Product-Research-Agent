"""
run_pipeline_stream: event order, only report-node text streamed as tokens, clarify path, error path.
Fake graphs check the event mapping; the real graph runs on FakeGroq/FakeTavily (no API calls).
"""
import contextlib
import io
import unittest
from unittest import mock

import httpx
from groq import RateLimitError
from langchain_core.messages import AIMessageChunk

from support import FakeGroq, FakeTavily

import agent
import test_token_load
import tools
from tools import DailyQuotaExceeded


class FakeGraph:
    """Yields (mode, chunk) pairs like graph.stream(stream_mode=[...]); raises `error` at the end if given."""

    def __init__(self, chunks, error=None):
        self.chunks = chunks
        self.error = error
        self.stream_modes = None

    def stream(self, inputs, config=None, stream_mode=None):
        self.stream_modes = stream_mode
        yield from self.chunks
        if self.error:
            raise self.error


def token(text, node, **kwargs):
    return ("messages", (AIMessageChunk(content=text, **kwargs), {"langgraph_node": node}))


def run_fake(graph):
    with mock.patch.object(agent, "_graph", graph):
        return list(agent.run_pipeline_stream("query"))


def run_real(handlers=None, product_pages=True):
    default_handlers, search = test_token_load.answers(product_pages)
    fake = FakeGroq({**default_handlers, **(handlers or {})}).install()
    tavily = FakeTavily(search=search).install()
    events = []
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            for event in agent.run_pipeline_stream("laptop for coding under 60000"):
                events.append(event)
    finally:
        fake.uninstall()
        tavily.uninstall()
    return events


DONE_STATE = {"report": "## Report", "report_candidates": [{"product_name": "a"}], "is_degraded": True}


class FakeGraphStreamTest(unittest.TestCase):
    def test_events_mapped_in_order(self):
        graph = FakeGraph([
            ("custom", {"type": "progress", "text": "Iteration 1/4 — 0 qualified so far"}),
            ("updates", {"start_iteration": {"iteration": 1}}),
            ("custom", {"type": "headline", "text": "No candidate has a verified price."}),
            token("## ", "report"),
            token("Report", "report"),
            ("updates", {"report": DONE_STATE}),
        ])
        events = run_fake(graph)
        self.assertEqual(graph.stream_modes, ["updates", "messages", "custom"])
        self.assertEqual(events, [
            {"type": "progress", "text": "Iteration 1/4 — 0 qualified so far"},
            {"type": "headline", "text": "No candidate has a verified price."},
            {"type": "token", "text": "## Report"},  # no section 1 heading: held back, sent whole at the end
            {"type": "final", "status": "done", "report": "## Report", "candidates": [{"product_name": "a"}],
             "is_degraded": True},
        ])

    def test_only_report_text_tokens_streamed(self):
        events = run_fake(FakeGraph([
            token("laptop i5 8GB", "search"),                                    # query rewrite
            token("", "report", additional_kwargs={"reasoning_content": "hmm"}),  # reasoning only
            token("", "extract_candidates", tool_call_chunks=[{"name": "x", "args": "{}", "id": "1", "index": 0}]),
            token("Hello", "report"),
            ("updates", {"report": DONE_STATE}),
        ]))
        self.assertEqual([e for e in events if e["type"] == "token"], [{"type": "token", "text": "Hello"}])

    def test_text_before_section_1_is_not_streamed(self):
        events = run_fake(FakeGraph([
            token("**No candidate has a verified price.**\n\n", "report"),
            token("## 1. Requirements", "report"),
            token(" Summary\n", "report"),
            token("- Budget: 60000", "report"),
            ("updates", {"report": DONE_STATE}),
        ]))
        self.assertEqual([e["text"] for e in events if e["type"] == "token"],
                         ["## 1. Requirements Summary\n", "- Budget: 60000"])

    def test_clarify_path(self):
        events = run_fake(FakeGraph([("updates", {"intake": {"clarify_question": "What is your budget?"}})]))
        self.assertEqual(events, [{"type": "final", "status": "clarify", "question": "What is your budget?"}])

    def test_error_path_propagates_after_earlier_events(self):
        graph = FakeGraph([("custom", {"type": "progress", "text": "Searching Indian retail sites..."})],
                          error=DailyQuotaExceeded("daily limit"))
        received = []
        with mock.patch.object(agent, "_graph", graph):
            with self.assertRaises(DailyQuotaExceeded):
                for event in agent.run_pipeline_stream("query"):
                    received.append(event)
        self.assertEqual(received, [{"type": "progress", "text": "Searching Indian retail sites..."}])


class RealGraphStreamTest(unittest.TestCase):
    def test_order_progress_headline_tokens_final(self):
        events = run_real()
        types = [e["type"] for e in events]
        self.assertEqual(types[-1], "final")
        first_headline, first_token = types.index("headline"), types.index("token")
        self.assertLess(types.index("progress"), first_headline)
        self.assertLess(first_headline, first_token)
        self.assertEqual(set(types[first_token:-1]), {"token"})
        self.assertIn("Generating final report...", [e["text"] for e in events if e["type"] == "progress"])

    def test_tokens_are_exactly_the_report(self):
        events = run_real({"text": lambda p: "i5 12th gen 8GB RAM" if "Shorten them" in p else "## Report for you"})
        tokens = "".join(e["text"] for e in events if e["type"] == "token")
        self.assertEqual(tokens, "## Report for you")
        self.assertEqual(events[-1]["report"], "## Report for you")
        self.assertNotIn("thinking about it", tokens)  # FakeGroq's reasoning chunk

    def test_streamed_text_equals_stored_report_when_model_adds_a_preamble(self):
        report = "**Warning first**\n\n## 1. Requirements Summary\nLaptop for coding.\n## 5. Final Recommendation\nPick."
        events = run_real({"text": lambda p: "i5 12th gen 8GB RAM" if "Shorten them" in p else report})
        tokens = "".join(e["text"] for e in events if e["type"] == "token")
        self.assertEqual(tokens, "## 1. Requirements Summary\nLaptop for coding.\n## 5. Final Recommendation\nPick.")
        self.assertEqual(events[-1]["report"], tokens)

    def test_final_matches_run_pipeline(self):
        events = run_real()
        default_handlers, search = test_token_load.answers()
        fake = FakeGroq(default_handlers).install()
        tavily = FakeTavily(search=search).install()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                result = agent.run_pipeline("laptop for coding under 60000")
        finally:
            fake.uninstall()
            tavily.uninstall()
        final = {k: v for k, v in events[-1].items() if k != "type"}
        self.assertEqual(final, result)

    def test_clarify_path(self):
        events = run_real({"Call-A": lambda p: {"status": "unclear", "question": "What is your budget?"}})
        self.assertEqual(events, [{"type": "final", "status": "clarify", "question": "What is your budget?"}])

    def test_rate_limit_wait_streamed_as_progress(self):
        calls = []

        def fit_with_one_429(prompt):
            calls.append(prompt)
            if len(calls) == 1:
                response = httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"))
                raise RateLimitError("Rate limit reached ... Please try again in 6.105s.", response=response, body=None)
            return {"fit_score": 8, "reasoning": "ok", "missing_or_weak_specs": []}

        with mock.patch.object(tools.time, "sleep") as sleep:
            events = run_real({"Fit-Evaluation": fit_with_one_429})
        sleep.assert_called_once()
        progress = [e["text"] for e in events if e["type"] == "progress"]
        self.assertIn("Waiting 7s for Groq rate limit, retry 1/3", progress)
        self.assertLess(progress.index("Scoring fit for 2 candidate(s)..."),
                        progress.index("Waiting 7s for Groq rate limit, retry 1/3"))
        self.assertEqual(events[-1]["type"], "final")

    def test_rate_limit_wait_outside_a_graph_is_silent(self):
        response = httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com"))
        attempts = []

        class Chain:
            def invoke(self, inputs):
                attempts.append(inputs)
                if len(attempts) == 1:
                    raise RateLimitError("Please try again in 1s.", response=response, body=None)
                return "ok"
        with mock.patch.object(tools.time, "sleep"), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(tools.invoke_with_retry(Chain(), {}), "ok")

    def test_error_path(self):
        def quota(prompt):
            raise DailyQuotaExceeded("daily limit")
        with self.assertRaises(DailyQuotaExceeded):
            run_real({"Fit-Evaluation": quota})


if __name__ == "__main__":
    unittest.main()
