"""
Persistence: sessions table CRUD, checkpointed runs on the real graph (fake Groq/Tavily), reopening a saved search
with zero pipeline calls, a clarify exchange kept in one thread, the error status, and delete.
"""
import contextlib
import io
import json
import sqlite3
import tempfile
import unittest
import uuid
from pathlib import Path

from support import FakeGroq, FakeTavily

import agent
import graph
import storage
import test_token_load
from tools import DailyQuotaExceeded

QUERY = "double door frost free refrigerator 250 litre 3 star under 50000"


def blocked(prompt):
    raise AssertionError("a reopened session must not call Groq")


class PersistenceCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "nested" / "agent.db"
        self.open()

    def open(self):
        self.conn = storage.open_connection(self.db)
        self.saver = storage.make_checkpointer(self.conn)
        self.store = storage.SessionStore(self.conn, self.saver.lock)
        self.graph = graph.build_graph(checkpointer=self.saver)

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def search(self, query, thread_id, handlers=None, previous_question=None, first_message=None):
        default_handlers, search = test_token_load.answers()
        fake = FakeGroq({**default_handlers, **(handlers or {})}).install()
        tavily = FakeTavily(search=search).install()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                events = list(storage.track_session(
                    self.store, thread_id, first_message or query,
                    agent.run_pipeline_stream(query, previous_question, graph=self.graph, thread_id=thread_id)))
        finally:
            fake.uninstall()
            tavily.uninstall()
        return events


class SessionStoreTest(PersistenceCase):
    def test_creates_db_directory_and_table(self):
        self.assertTrue(self.db.exists())
        tables = {row[0] for row in self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertTrue({"sessions", "checkpoints", "writes"} <= tables)

    def test_crud_and_order(self):
        self.store.add("t1", "first search")
        self.store.add("t2", "second search")
        self.conn.execute("UPDATE sessions SET created_at = '2026-10-05T10:00:00+00:00' WHERE thread_id = 't1'")
        self.conn.execute("UPDATE sessions SET created_at = '2026-10-05T11:00:00+00:00' WHERE thread_id = 't2'")
        self.assertEqual([s["thread_id"] for s in self.store.list()], ["t2", "t1"])
        self.assertEqual(self.store.get("t1")["status"], "running")
        self.store.set_status("t1", "done")
        self.store.add("t1", "a clarify answer must not replace the title")
        self.assertEqual((self.store.get("t1")["title"], self.store.get("t1")["status"]), ("first search", "done"))
        self.store.delete("t2", self.saver)
        self.assertIsNone(self.store.get("t2"))
        self.assertEqual(len(self.store.list()), 1)

    def test_title_is_first_message_truncated(self):
        self.assertEqual(storage.session_title("  fridge\n under   50000 "), "fridge under 50000")
        long = "double door frost free refrigerator 250 litre 3 star under 50000 for a family of five"
        title = storage.session_title(long)
        self.assertLessEqual(len(title), 60)
        self.assertTrue(title.endswith("…"))
        self.assertTrue(long.startswith(title[:-1]))


class CheckpointedRunTest(PersistenceCase):
    def test_full_run_writes_checkpoint_and_session(self):
        thread_id = uuid.uuid4().hex
        events = self.search(QUERY, thread_id)
        self.assertEqual(events[-1]["status"], "done")
        session = self.store.get(thread_id)
        self.assertEqual((session["title"], session["status"]), (storage.session_title(QUERY), "done"))
        state = storage.load_saved_state(self.graph, thread_id)
        self.assertEqual(state["report"], events[-1]["report"])
        self.assertTrue(state["recommendation_headline"])
        self.assertEqual(state["user_query"], QUERY)
        count = self.conn.execute("SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", (thread_id,)).fetchone()[0]
        self.assertGreater(count, 0)

    def test_saved_state_is_plain_json(self):
        thread_id = uuid.uuid4().hex
        self.search(QUERY, thread_id)
        json.dumps(storage.load_saved_state(self.graph, thread_id))  # no callbacks, sets or other objects

    def test_reopening_makes_no_pipeline_calls(self):
        thread_id = uuid.uuid4().hex
        report = self.search(QUERY, thread_id)[-1]["report"]
        fake = FakeGroq({"text": blocked, "Call-A": blocked}).install()
        tavily = FakeTavily(search=lambda q, d: self.fail("Tavily called"), extract=lambda u: self.fail("extract")).install()
        try:
            state = storage.load_saved_state(self.graph, thread_id)
        finally:
            fake.uninstall()
            tavily.uninstall()
        self.assertEqual(state["report"], report)
        self.assertEqual(fake.calls, [])
        self.assertEqual(tavily.search_calls, [])

    def test_survives_a_restart(self):
        thread_id = uuid.uuid4().hex
        report = self.search(QUERY, thread_id)[-1]["report"]
        self.conn.close()
        self.open()  # a new process: new connection, saver and graph on the same file
        self.assertEqual(self.store.get(thread_id)["status"], "done")
        self.assertEqual(storage.load_saved_state(self.graph, thread_id)["report"], report)

    def test_clarify_exchange_stays_in_one_thread(self):
        thread_id = uuid.uuid4().hex
        question = "What will you mainly use the laptop for?"
        events = self.search("I want a laptop under 60000", thread_id,
                             {"Call-A": lambda p: {"status": "unclear", "question": question, "budget": 60000,
                                                   "category": "laptop", "usecase": None}})
        self.assertEqual(events, [{"type": "final", "status": "clarify", "question": question}])
        self.assertEqual(self.store.get(thread_id)["status"], "clarifying")
        self.assertEqual(storage.load_saved_state(self.graph, thread_id)["clarify_question"], question)

        events = self.search("I want a laptop under 60000. coding", thread_id, previous_question=question,
                             first_message="I want a laptop under 60000")
        self.assertEqual(events[-1]["status"], "done")
        state = storage.load_saved_state(self.graph, thread_id)
        self.assertIsNone(state["clarify_question"])
        self.assertEqual(state["report"], events[-1]["report"])
        self.assertEqual(len(self.store.list()), 1)
        self.assertEqual(self.store.get(thread_id)["status"], "done")

    def test_error_sets_status(self):
        thread_id = uuid.uuid4().hex

        def quota(prompt):
            raise DailyQuotaExceeded("daily limit")
        with self.assertRaises(DailyQuotaExceeded):
            self.search(QUERY, thread_id, {"Fit-Evaluation": quota})
        self.assertEqual(self.store.get(thread_id)["status"], "error")

    def test_delete_removes_session_and_checkpoints(self):
        keep, gone = uuid.uuid4().hex, uuid.uuid4().hex
        self.search(QUERY, keep)
        self.search(QUERY, gone)
        self.store.delete(gone, self.saver)
        self.assertIsNone(self.store.get(gone))
        self.assertEqual(storage.load_saved_state(self.graph, gone), {})
        for table in ("checkpoints", "writes"):
            self.assertEqual(self.conn.execute(f"SELECT COUNT(*) FROM {table} WHERE thread_id = ?", (gone,)).fetchone()[0], 0)
        self.assertTrue(storage.load_saved_state(self.graph, keep)["report"])

    def test_plain_graph_still_runs_without_checkpointer(self):
        handlers, search = test_token_load.answers()
        fake = FakeGroq(handlers).install()
        tavily = FakeTavily(search=search).install()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                result = agent.run_pipeline("laptop for coding under 60000")
        finally:
            fake.uninstall()
            tavily.uninstall()
        self.assertEqual(result["status"], "done")

    def test_make_config_has_no_callback_unless_given(self):
        self.assertEqual(graph.make_config(thread_id="t")["configurable"], {"thread_id": "t"})
        self.assertEqual(graph.make_config()["configurable"], {})


if __name__ == "__main__":
    unittest.main()
