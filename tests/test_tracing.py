"""
LangSmith tracing: off without a key (no tracer, no network), on with LANGSMITH_TRACING=true and a key.
No test reaches LangSmith: the "on" side only checks the switch, the run config uses a recording tracer.
"""
import logging
import os
import unittest
import warnings
from unittest import mock

import langsmith
import requests
from langchain_core.tracers.context import _tracing_v2_is_enabled
from langsmith import utils as ls_utils

import support  # noqa: F401  (network guard, tracing forced off)

import test_stream
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


if __name__ == "__main__":
    unittest.main()
