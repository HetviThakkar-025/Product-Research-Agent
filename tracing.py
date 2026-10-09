"""
LangSmith tracing switch. Tracing is on only when LANGSMITH_TRACING=true and LANGSMITH_API_KEY are both set
(from the environment or .env); LANGSMITH_PROJECT defaults to DEFAULT_PROJECT.
"""
import os

DEFAULT_PROJECT = "product-research-agent"
QUERY_METADATA_CHARS = 100


def configure_tracing(environ=None):
    """Called once after load_dotenv(); returns whether tracing is on."""
    environ = os.environ if environ is None else environ
    wanted = environ.get("LANGSMITH_TRACING", "").strip().lower() == "true"
    if wanted and environ.get("LANGSMITH_API_KEY", "").strip():
        environ.setdefault("LANGSMITH_PROJECT", DEFAULT_PROJECT)
        return True
    # without a key LangChain would still try to send traces and log warnings, so switch tracing off explicitly;
    # LANGCHAIN_TRACING_V2 is the older name LangChain also reads
    environ["LANGSMITH_TRACING"] = "false"
    environ["LANGCHAIN_TRACING_V2"] = "false"
    return False


def trace_metadata(user_query, thread_id=None):
    """Metadata that makes a run easy to find in LangSmith: the thread and the start of the user's query."""
    metadata = {"query": (user_query or "")[:QUERY_METADATA_CHARS]}
    if thread_id:
        metadata["thread_id"] = thread_id
    return metadata
