"""
Persistence for the Streamlit app: one SQLite file (data/agent.db) holding LangGraph's checkpoints (each search's
graph state, keyed by thread_id) and a small sessions table that feeds the past-searches sidebar.
The file lives on local disk: on Streamlit Community Cloud it is lost when the app restarts or is redeployed.
"""
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

DB_PATH = Path(os.getenv("AGENT_DB_PATH") or Path(__file__).resolve().parent / "data" / "agent.db")
TITLE_MAX_CHARS = 60


def open_connection(path=DB_PATH):
    """One connection shared by every Streamlit session (each runs in its own thread), hence check_same_thread=False."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(path), check_same_thread=False)


def make_checkpointer(conn):
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


def session_title(first_message):
    """The user's first message on one line, cut to about 60 characters."""
    text = " ".join(first_message.split())
    return text if len(text) <= TITLE_MAX_CHARS else text[:TITLE_MAX_CHARS - 1].rstrip() + "…"


def load_saved_state(graph, thread_id):
    """A thread's saved graph state from its last checkpoint ({} if none); runs nothing."""
    return dict(graph.get_state({"configurable": {"thread_id": thread_id}}).values or {})


class SessionStore:
    """
    The sessions table: thread_id, title, created_at, status. status is "running" while a search runs, then
    "clarifying" (waiting for the user's answer), "done" or "error". Shares the checkpointer's connection and lock.
    """

    def __init__(self, conn, lock):
        self.conn = conn
        self.lock = lock
        with self.lock, self.conn:
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS sessions ("
                " thread_id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at TEXT NOT NULL, status TEXT NOT NULL)")

    def add(self, thread_id, first_message):
        """Inserts a session (status "running"); a thread that already has one (a clarify answer) keeps it."""
        with self.lock, self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO sessions (thread_id, title, created_at, status) VALUES (?, ?, ?, 'running')",
                (thread_id, session_title(first_message), datetime.now(timezone.utc).isoformat(timespec="seconds")))

    def set_status(self, thread_id, status):
        with self.lock, self.conn:
            self.conn.execute("UPDATE sessions SET status = ? WHERE thread_id = ?", (status, thread_id))

    def get(self, thread_id):
        rows = self._select("WHERE thread_id = ?", (thread_id,))
        return rows[0] if rows else None

    def list(self):
        """All sessions, newest first."""
        return self._select("ORDER BY created_at DESC, rowid DESC", ())

    def delete(self, thread_id, checkpointer):
        """Removes the session row and every checkpoint and write of its thread."""
        with self.lock, self.conn:
            self.conn.execute("DELETE FROM sessions WHERE thread_id = ?", (thread_id,))
        checkpointer.delete_thread(thread_id)

    def _select(self, where, params):
        with self.lock:
            rows = self.conn.execute(
                f"SELECT thread_id, title, created_at, status FROM sessions {where}", params).fetchall()
        return [dict(zip(("thread_id", "title", "created_at", "status"), row)) for row in rows]


def track_session(store, thread_id, first_message, events):
    """
    Wraps a run_pipeline_stream generator: adds the session row before the run and sets its status from the final
    event ("clarifying" or "done"), or "error" if the run raises (the exception still propagates).
    """
    store.add(thread_id, first_message)
    try:
        for event in events:
            if event.get("type") == "final":
                store.set_status(thread_id, "clarifying" if event.get("status") == "clarify" else "done")
            yield event
    except Exception:  # not GeneratorExit: a caller that stops reading after the final event is not an error
        store.set_status(thread_id, "error")
        raise
