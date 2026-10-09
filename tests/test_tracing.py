"""
LangSmith tracing: off without a key (no tracer, no network), on with LANGSMITH_TRACING=true and a key.
No test reaches LangSmith: the "on" side only checks the switch, the run config uses a recording tracer.
"""
import contextlib
import io
import logging
import os
import unittest
import warnings
from unittest import mock

import langsmith
import requests
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableLambda
from langchain_core.tracers.context import _tracing_v2_is_enabled
from langsmith import utils as ls_utils

import support  # noqa: F401  (network guard, tracing forced off)

import agent
import graph
import test_app
import test_stream
import test_retry
import test_token_load
import tools
from support import FakeGroq, FakeTavily
from tracing import DEFAULT_PROJECT, configure_tracing, trace_metadata


def _no_langsmith(*args, **kwargs):
    raise AssertionError("tracing is off, yet something tried to reach LangSmith")


class TracingSwitchTest(unittest.TestCase):
    def test_off_without_any_settings(self):
        env = {}
        self.assertFalse(configure_tracing(env))
        self.assertEqual(env["LANGSMITH_TRACING"], "false")
        self.assertEqual(env["LANGCHAIN_TRACING_V2"], "false")

    def test_off_when_tracing_requested_without_key(self):
        env = {"LANGSMITH_TRACING": "true", "LANGSMITH_API_KEY": " "}
        self.assertFalse(configure_tracing(env))
        self.assertEqual(env["LANGSMITH_TRACING"], "false")

    def test_off_with_key_but_tracing_not_requested(self):
        env = {"LANGSMITH_API_KEY": "lsv2-fake"}
        self.assertFalse(configure_tracing(env))

    def test_on_with_key_and_default_project(self):
        env = {"LANGSMITH_TRACING": "true", "LANGSMITH_API_KEY": "lsv2-fake"}
        self.assertTrue(configure_tracing(env))
        self.assertEqual(env["LANGSMITH_PROJECT"], DEFAULT_PROJECT)
        self.assertEqual(env["LANGSMITH_TRACING"], "true")

    def test_on_keeps_a_chosen_project(self):
        env = {"LANGSMITH_TRACING": "True", "LANGSMITH_API_KEY": "lsv2-fake", "LANGSMITH_PROJECT": "mine"}
        self.assertTrue(configure_tracing(env))
        self.assertEqual(env["LANGSMITH_PROJECT"], "mine")


class TracingOffRunTest(unittest.TestCase):
    def test_full_run_without_key_makes_no_langsmith_call_or_warning(self):
        with mock.patch.dict(os.environ, {"LANGSMITH_TRACING": "true"}):
            os.environ.pop("LANGSMITH_API_KEY", None)
            os.environ.pop("LANGCHAIN_API_KEY", None)
            self.assertFalse(configure_tracing())
            ls_utils.get_env_var.cache_clear()  # LangSmith caches env reads
            try:
                self.assertFalse(_tracing_v2_is_enabled())
                with mock.patch.object(langsmith.Client, "__init__", _no_langsmith), \
                        mock.patch.object(requests.Session, "send", _no_langsmith), \
                        warnings.catch_warnings(record=True) as caught, \
                        self.assertNoLogs("langsmith", level=logging.WARNING):
                    warnings.simplefilter("always")
                    events = test_stream.run_real()
            finally:
                ls_utils.get_env_var.cache_clear()
        self.assertEqual(events[-1]["type"], "final")
        self.assertEqual(events[-1]["status"], "done")
        self.assertEqual([str(w.message) for w in caught if "langsmith" in str(w.message).lower()], [])


class TraceMetadataTest(unittest.TestCase):
    def test_query_truncated_and_thread_kept(self):
        self.assertEqual(trace_metadata("x" * 150, "t1"), {"query": "x" * 100, "thread_id": "t1"})

    def test_no_thread(self):
        self.assertEqual(trace_metadata("fridge"), {"query": "fridge"})


class RecordingTracer(BaseCallbackHandler):
    """Stands in for the LangSmith tracer: records every run's name, parent, tags and metadata."""

    def __init__(self):
        self.runs = {}  # run_id -> {"name", "parent", "tags", "metadata", "kind"}

    def _record(self, kind, serialized, run_id, parent_run_id, tags, metadata, name):
        if name is None and serialized:
            name = serialized.get("name") or (serialized.get("id") or ["?"])[-1]
        self.runs[run_id] = {"name": name, "parent": parent_run_id, "tags": tags or [],
                             "metadata": metadata or {}, "kind": kind}

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        self._record("chain", serialized, run_id, parent_run_id, tags, metadata, kwargs.get("name"))

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, tags=None, metadata=None,
                            **kwargs):
        self._record("llm", serialized, run_id, parent_run_id, tags, metadata, kwargs.get("name"))

    def roots(self):
        return [r for r in self.runs.values() if r["parent"] is None]

    def names(self):
        return [r["name"] for r in self.runs.values()]


def traced_run(product_pages=True, spec_results=False):
    """A full mocked graph run, labelled the way the app labels it, with RecordingTracer attached."""
    handlers, search = test_token_load.answers(product_pages)
    if spec_results:  # the spec search finds a page, so Call D runs too
        retail_search = search
        search = lambda query, domains: (  # noqa: E731
            {"results": [{"url": "https://www.91mobiles.com/hp-15", "title": "HP 15 specs",
                          "content": "i5-1235U 8GB RAM", "score": 0.9}]}
            if domains == tools.SPEC_DOMAINS else retail_search(query, domains))
    fake = FakeGroq(handlers).install()
    tavily = FakeTavily(search=search).install()
    tracer = RecordingTracer()
    config = graph.make_config(thread_id=None, run_name=agent.RUN_NAME, tags=["streamlit", "product-research"],
                               metadata=trace_metadata("laptop for coding under 60000", "thread-1"))
    config["callbacks"] = [tracer]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            graph.build_graph().invoke({"user_query": "laptop for coding under 60000"}, config=config)
    finally:
        fake.uninstall()
        tavily.uninstall()
    return tracer


class RunConfigTest(unittest.TestCase):
    def test_make_config_adds_labels_only_when_given(self):
        self.assertNotIn("tags", graph.make_config())
        self.assertNotIn("metadata", graph.make_config())
        self.assertNotIn("run_name", graph.make_config())
        config = graph.make_config(thread_id="t", run_name="r", tags=["a"], metadata={"query": "q"})
        self.assertEqual((config["run_name"], config["tags"], config["metadata"]), ("r", ["a"], {"query": "q"}))
        self.assertEqual(config["configurable"], {"thread_id": "t"})

    def test_run_pipeline_stream_labels_the_run(self):
        seen = {}

        class ConfigGraph:
            def stream(self, inputs, config=None, stream_mode=None):
                seen.update(config)
                return iter([("updates", {"intake": {"clarify_question": "Budget?"}})])

        list(agent.run_pipeline_stream("q" * 120, graph=ConfigGraph(), thread_id="thread-9", tags=["x"]))
        self.assertEqual(seen["run_name"], agent.RUN_NAME)
        self.assertEqual(seen["tags"], ["x"])
        self.assertEqual(seen["metadata"], {"query": "q" * 100, "thread_id": "thread-9"})
        self.assertEqual(seen["configurable"], {"thread_id": "thread-9"})

    def test_app_passes_its_tags(self):
        calls = []

        def stream(user_query, previous_question=None, **kwargs):
            calls.append(kwargs)
            yield {"type": "final", "status": "clarify", "question": "Budget?"}

        at = test_app.run_app(stream, query="I want a fridge")
        self.assertFalse(at.exception)
        self.assertEqual(calls[0]["tags"], ["streamlit", "product-research"])
        self.assertTrue(calls[0]["thread_id"])

    def test_root_run_carries_name_tags_and_metadata(self):
        tracer = traced_run()
        [root] = tracer.roots()
        self.assertEqual(root["name"], agent.RUN_NAME)
        self.assertEqual(root["tags"], ["streamlit", "product-research"])
        self.assertEqual(root["metadata"]["query"], "laptop for coding under 60000")
        self.assertEqual(root["metadata"]["thread_id"], "thread-1")
        # tags and metadata are inherited, so every node and LLM call can be filtered by them
        for run in tracer.runs.values():
            self.assertIn("streamlit", run["tags"])
            self.assertEqual(run["metadata"].get("thread_id"), "thread-1")


class TraceNamesTest(unittest.TestCase):
    def test_nodes_and_llm_calls_have_readable_names(self):
        names = traced_run(spec_results=True).names()
        for name in ("intake", "search", "extract_candidates", "verify_specs", "check_prices", "score_fit",
                     "merge", "report", "Call A: clarify", "Call B: requirements", "Search query rewrite",
                     "Call C: candidates", "Call D: specs", "Call E: price", "Call F: fit", "Report"):
            self.assertIn(name, names)

    def test_skipped_call_c_shows_as_search_routed_back_to_search(self):
        tracer = traced_run(product_pages=False)
        names = tracer.names()
        self.assertNotIn("Call C: candidates", names)
        self.assertNotIn("extract_candidates", names)
        self.assertIn("route_after_search", names)

    def test_rate_limit_wait_is_a_named_step_under_the_retried_call(self):
        clock = test_retry.FakeClock()
        chain = test_retry.Chain(clock, [test_retry.rate_limit(6.105)])
        tracer = RecordingTracer()
        step = RunnableLambda(lambda _: tools.invoke_with_retry(chain, {}), name="Call X")
        with mock.patch.object(tools, "time", clock), mock.patch.object(tools, "emit_progress"), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(step.invoke({}, config={"callbacks": [tracer]}), "ok")
        self.assertEqual(clock.sleeps, [7.105])  # same wait as before
        [wait] = [r for r in tracer.runs.values() if r["name"] == "Groq rate-limit wait"]
        self.assertEqual(tracer.runs[wait["parent"]]["name"], "Call X")


class TracePrivacyTest(unittest.TestCase):
    def test_no_api_key_in_what_langsmith_would_receive(self):
        """The real LangChainTracer with a mocked client: every create/update payload, searched for the key values."""
        import json
        from langchain_core.tracers.langchain import LangChainTracer, wait_for_all_tracers

        client = mock.MagicMock()
        handlers, search = test_token_load.answers()
        fake = FakeGroq(handlers).install()
        tavily = FakeTavily(search=search).install()
        config = graph.make_config(run_name=agent.RUN_NAME, tags=["test"], metadata=trace_metadata("laptop"))
        config["callbacks"] = [LangChainTracer(client=client, project_name="test")]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                graph.build_graph().invoke({"user_query": "laptop for coding under 60000"}, config=config)
            wait_for_all_tracers()
        finally:
            fake.uninstall()
            tavily.uninstall()
        payload = json.dumps([call.kwargs for call in client.method_calls], default=str)
        self.assertIn("Call C: candidates", payload)
        for key in ("GROQ_API_KEY", "TAVILY_API_KEY"):
            self.assertNotIn(os.environ[key], payload)


if __name__ == "__main__":
    unittest.main()
