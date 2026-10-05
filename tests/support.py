"""
Shared setup for the mocked tests: repo on sys.path, dummy API keys, and a network guard.
Every Groq and Tavily call is intercepted, so no test can reach either service.
Import this module before any project module.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")

from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_groq import ChatGroq  # noqa: E402
from langchain_tavily import TavilyExtract, TavilySearch  # noqa: E402


class NetworkCallBlocked(AssertionError):
    pass


def _blocked(*args, **kwargs):
    raise NetworkCallBlocked("a test tried to reach Groq or Tavily without a mock")


ChatGroq._generate = _blocked
TavilySearch._run = _blocked
TavilyExtract._run = _blocked


def load_fixture(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class FakeGroq:
    """
    Stands in for ChatGroq._generate. Structured calls are answered by handlers[tool name]
    (e.g. "Candidate-Extraction"), plain-text calls by handlers["text"]; a handler gets the
    prompt text and returns the tool args (dict) or the text (str).
    """

    def __init__(self, handlers, prompt_tokens=100, completion_tokens=10):
        self.handlers = handlers
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.calls = []  # (tool name or "text", prompt text)

    def install(self):
        fake = self

        def _generate(model, messages, stop=None, run_manager=None, **kwargs):
            prompt = "\n".join(str(m.content) for m in messages)
            tools = kwargs.get("tools") or []
            name = tools[0]["function"]["name"] if tools else "text"
            fake.calls.append((name, prompt))
            answer = fake.handlers[name](prompt)
            if name == "text":
                message = AIMessage(content=answer)
            else:
                message = AIMessage(content="", tool_calls=[{"name": name, "args": answer, "id": f"call_{len(fake.calls)}"}])
            usage = {"prompt_tokens": fake.prompt_tokens, "completion_tokens": fake.completion_tokens}
            return ChatResult(generations=[ChatGeneration(message=message)], llm_output={"token_usage": usage})

        ChatGroq._generate = _generate
        return self

    def uninstall(self):
        ChatGroq._generate = _blocked

    def names(self):
        return [name for name, _ in self.calls]


class FakeTavily:
    """Stands in for TavilySearch._run / TavilyExtract._run; search(query, include_domains) -> response dict."""

    def __init__(self, search=None, extract=None):
        self.search = search or (lambda query, domains: {"results": []})
        self.extract = extract or (lambda urls: {"results": []})
        self.search_calls = []
        self.extract_calls = []

    def install(self):
        fake = self

        def _search(tool, query, *args, **kwargs):
            domains = kwargs.get("include_domains") or tool.include_domains
            fake.search_calls.append((query, tool.max_results, domains))
            return fake.search(query, domains)

        def _extract(tool, urls, *args, **kwargs):
            fake.extract_calls.append(urls)
            return fake.extract(urls)

        TavilySearch._run = _search
        TavilyExtract._run = _extract
        return self

    def uninstall(self):
        TavilySearch._run = _blocked
        TavilyExtract._run = _blocked
