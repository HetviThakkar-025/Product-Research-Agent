"""invoke_with_retry on 429s: first retry uses Groq's hint, later ones wait out the rest of the minute (max 60s)."""
import contextlib
import io
import unittest
from unittest import mock

import httpx
from groq import APIConnectionError, APIError, APIStatusError, RateLimitError

import support  # noqa: F401  (network guard)

import tools


def rate_limit(hint_seconds):
    response = httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"))
    return RateLimitError(f"Rate limit reached ... Please try again in {hint_seconds}s.", response=response, body=None)


class FakeClock:
    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(round(seconds, 3))
        self.now += seconds


class Chain:
    """Raises the given errors in order, then returns "ok"; `work` seconds pass on the clock per attempt."""

    def __init__(self, clock, errors, work=0.0):
        self.clock, self.errors, self.work = clock, list(errors), work

    def invoke(self, inputs):
        self.clock.now += self.work
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


def run(errors, work=0.0):
    clock = FakeClock()
    progress = []
    with mock.patch.object(tools, "time", clock), mock.patch.object(tools, "emit_progress", progress.append), \
            contextlib.redirect_stdout(io.StringIO()):
        result = tools.invoke_with_retry(Chain(clock, errors, work), {})
    return result, clock.sleeps, progress


class RateLimitWaitTest(unittest.TestCase):
    def test_first_retry_uses_groq_hint(self):
        result, sleeps, progress = run([rate_limit(6.105)])
        self.assertEqual(result, "ok")
        self.assertEqual(sleeps, [7.105])
        self.assertEqual(progress, ["Waiting 7s for Groq rate limit, retry 1/3"])

    def test_second_429_waits_rest_of_minute(self):
        # live check 2: hints of 4s, 3s, 2s were too short; now the 2nd wait covers the rest of the minute
        result, sleeps, progress = run([rate_limit(3), rate_limit(2)], work=1.0)
        self.assertEqual(result, "ok")
        # first wait 4s; second 429 comes 4s + 1s of work after the first -> 60 - 5 = 55s
        self.assertEqual(sleeps, [4.0, 55.0])
        self.assertEqual(progress, ["Waiting 4s for Groq rate limit, retry 1/3",
                                    "Waiting 55s for Groq rate limit, retry 2/3"])

    def test_longer_hint_still_wins_but_capped_at_60(self):
        _, sleeps, _ = run([rate_limit(1), rate_limit(58.5)])
        self.assertEqual(sleeps, [2.0, 59.5])
        _, sleeps, _ = run([rate_limit(1), rate_limit(90)])
        self.assertEqual(sleeps, [2.0, 60])

    def test_after_the_minute_has_passed_hint_is_used_again(self):
        _, sleeps, _ = run([rate_limit(1), rate_limit(1), rate_limit(5)], work=1.0)
        # waits: 2 (hint); then 60 - 3 = 57; the 3rd 429 is 61s after the first, so the 6s hint is used
        self.assertEqual(sleeps, [2.0, 57.0, 6.0])

    def test_gives_up_after_three_retries(self):
        with self.assertRaises(tools.LLMRetriesExhausted):
            run([rate_limit(1)] * 4)

    def test_each_call_starts_a_fresh_minute(self):
        _, first_call, _ = run([rate_limit(2)])
        _, second_call, _ = run([rate_limit(2)])
        self.assertEqual(first_call, [3.0])
        self.assertEqual(second_call, [3.0])


TOOL_VALIDATION = ("Tool call validation failed: parameters for tool Call-B did not match schema: errors: "
                   "[`/negotiable_specs/brand`: expected string, but got null, `/negotiable_specs/color`: expected "
                   "string, but got null]")


def streamed_tool_error():
    # the shape of the live error: a plain groq.APIError raised while reading the stream (no status code)
    return APIError(TOOL_VALIDATION, httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"), body=None)


class ToolCallErrorTest(unittest.TestCase):
    def test_streamed_validation_error_retried_immediately(self):
        result, sleeps, progress = run([streamed_tool_error()])
        self.assertEqual(result, "ok")
        self.assertEqual((sleeps, progress), ([], []))

    def test_gives_up_after_two_retries_with_the_real_error_text(self):
        clock = FakeClock()
        chain = Chain(clock, [streamed_tool_error()] * 3)
        with mock.patch.object(tools, "time", clock), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(tools.LLMRetriesExhausted) as raised:
                tools.invoke_with_retry(chain, {})
        self.assertIn("after 2 retries", str(raised.exception))
        self.assertIn("/negotiable_specs/brand`: expected string, but got null", str(raised.exception))
        self.assertEqual(chain.errors, [])  # 1 call + 2 retries

    def test_400_tool_use_failed_still_retried(self):
        response = httpx.Response(400, request=httpx.Request("POST", "https://api.groq.com"))
        error = APIStatusError("Error code: 400 - {'error': {'code': 'tool_use_failed'}}", response=response, body=None)
        result, sleeps, _ = run([error, error])
        self.assertEqual((result, sleeps), ("ok", []))

    def test_other_errors_without_status_are_not_retried(self):
        request = httpx.Request("POST", "https://api.groq.com")
        for error in (APIConnectionError(request=request), APIError("stream ended unexpectedly", request, body=None)):
            clock = FakeClock()
            chain = Chain(clock, [error])
            with mock.patch.object(tools, "time", clock), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(type(error)):
                    tools.invoke_with_retry(chain, {})


if __name__ == "__main__":
    unittest.main()
