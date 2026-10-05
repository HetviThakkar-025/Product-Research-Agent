"""
The past-searches sidebar (Streamlit AppTest on a temporary SQLite file): list, open read-only with no pipeline
calls, clarifying/error sessions, delete, New search, and a search from the app saving its thread.
"""
import contextlib
import io
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

from support import ROOT, FakeGroq, FakeTavily

from streamlit.testing.v1 import AppTest

import agent
import graph
import storage
import test_token_load

APP = str(ROOT / "app.py")
QUERY = "double door frost free refrigerator 250 litre 3 star under 50000"


def no_pipeline(*args, **kwargs):
    raise AssertionError("opening a saved search must not run the pipeline")


class SidebarTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "agent.db")
        self.env = mock.patch.dict(os.environ, {"AGENT_DB_PATH": self.db})
        self.env.start()
        self.conn = storage.open_connection(self.db)
        self.saver = storage.make_checkpointer(self.conn)
        self.store = storage.SessionStore(self.conn, self.saver.lock)
        self.graph = graph.build_graph(checkpointer=self.saver)

    def tearDown(self):
        self.env.stop()
        self.conn.close()
        self.tmp.cleanup()

    def save_search(self, query=QUERY, created_at=None):
        """A real (fake-LLM) search saved to the temporary file, as the app would save it."""
        thread_id = uuid.uuid4().hex
        handlers, search = test_token_load.answers()
        fake = FakeGroq(handlers).install()
        tavily = FakeTavily(search=search).install()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                events = list(storage.track_session(self.store, thread_id, query, agent.run_pipeline_stream(
                    query, graph=self.graph, thread_id=thread_id)))
        finally:
            fake.uninstall()
            tavily.uninstall()
        if created_at:
            with self.saver.lock, self.conn:
                self.conn.execute("UPDATE sessions SET created_at = ? WHERE thread_id = ?", (created_at, thread_id))
        return thread_id, events[-1]

    def app(self):
        at = AppTest.from_file(APP, default_timeout=30)
        at.run()
        return at

    def session_buttons(self, at):
        return [b for b in at.sidebar.button if b.key and b.key.startswith("open_")]

    def test_sidebar_lists_sessions_newest_first_with_status_and_note(self):
        old, _ = self.save_search("older fridge search", "2026-10-05T09:00:00+00:00")
        new, _ = self.save_search("newer fridge search", "2026-10-05T10:00:00+00:00")
        self.store.add("t-clarify", "I want a laptop")
        self.store.set_status("t-clarify", "clarifying")
        with self.saver.lock, self.conn:
            self.conn.execute("UPDATE sessions SET created_at = '2026-10-05T08:00:00+00:00' WHERE thread_id = 't-clarify'")
        at = self.app()
        self.assertEqual(at.sidebar.button[0].label, "New search")
        self.assertEqual([b.label for b in self.session_buttons(at)],
                         ["newer fridge search", "older fridge search", "I want a laptop (clarifying)"])
        self.assertIn("History resets when the app restarts.", [c.value for c in at.sidebar.caption])

    def test_open_shows_saved_report_without_running_the_pipeline(self):
        thread_id, final = self.save_search()
        at = self.app()
        fake = FakeGroq({}).install()  # any Groq call would fail: no handlers
        try:
            with mock.patch.object(agent, "run_pipeline_stream", no_pipeline):
                at.sidebar.button(key=f"open_{thread_id}").click().run()
        finally:
            fake.uninstall()
        self.assertFalse(at.exception)
        self.assertEqual(fake.calls, [])
        texts = [m.value for m in at.markdown]
        self.assertIn(QUERY, texts)
        self.assertIn(final["report"], texts)
        state = storage.load_saved_state(self.graph, thread_id)
        self.assertIn(f"**{state['recommendation_headline']}**", texts)
        self.assertTrue(any("read-only" in c.value for c in at.caption))

    def test_clarifying_and_error_sessions_show_status_not_report(self):
        self.store.add("t-err", "fridge search that failed")
        self.store.set_status("t-err", "error")
        self.store.add("t-clar", "I want a fridge")
        self.store.set_status("t-clar", "clarifying")
        at = self.app()
        at.sidebar.button(key="open_t-err").click().run()
        self.assertEqual([w.value for w in at.warning], ["This search ended with an error, so there is no report."])
        at.sidebar.button(key="open_t-clar").click().run()
        self.assertTrue(at.info[0].value.startswith("This search stopped at a clarifying question"))

    def test_delete_removes_session_row_and_checkpoints(self):
        keep, _ = self.save_search("keep this one")
        gone, _ = self.save_search("delete this one")
        at = self.app()
        at.sidebar.button(key=f"open_{gone}").click().run()
        at.sidebar.button(key=f"delete_{gone}").click().run()
        self.assertFalse(at.exception)
        self.assertEqual([b.label for b in self.session_buttons(at)], ["keep this one"])
        self.assertIsNone(self.store.get(gone))
        count = self.conn.execute("SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", (gone,)).fetchone()[0]
        self.assertEqual(count, 0)
        self.assertIsNone(at.session_state.viewing)
        self.assertTrue(storage.load_saved_state(self.graph, keep)["report"])

    def test_new_search_makes_a_new_thread(self):
        thread_id, _ = self.save_search()
        at = self.app()
        first = at.session_state.thread_id
        at.sidebar.button(key=f"open_{thread_id}").click().run()
        at.sidebar.button(key="new_search").click().run()
        self.assertNotEqual(at.session_state.thread_id, first)
        self.assertIsNone(at.session_state.viewing)
        self.assertEqual(at.session_state.messages, [])

    def test_search_from_the_app_saves_its_thread_and_session(self):
        handlers, search = test_token_load.answers()
        fake = FakeGroq(handlers).install()
        tavily = FakeTavily(search=search).install()
        try:
            at = self.app()
            at.chat_input[0].set_value("laptop for coding under 60000").run()
            first_thread = at.session_state.thread_id
            at.chat_input[0].set_value("laptop for gaming under 90000").run()
        finally:
            fake.uninstall()
            tavily.uninstall()
        self.assertFalse(at.exception)
        second_thread = at.session_state.thread_id
        self.assertNotEqual(first_thread, second_thread)  # a new request after a finished search is a new thread
        self.assertEqual([(s["title"], s["status"]) for s in self.store.list()],
                         [("laptop for gaming under 90000", "done"), ("laptop for coding under 60000", "done")])
        self.assertTrue(storage.load_saved_state(self.graph, first_thread)["report"])
        self.assertEqual([b.label for b in self.session_buttons(at)],
                         ["laptop for gaming under 90000", "laptop for coding under 60000"])

    def test_clarify_answer_stays_in_the_same_thread(self):
        question = "What will you mainly use the laptop for?"
        handlers, search = test_token_load.answers()
        calls = []

        def call_a(prompt):
            calls.append(prompt)
            if len(calls) == 1:
                return {"status": "unclear", "question": question, "budget": 60000, "category": "laptop", "usecase": None}
            return {"status": "clear", "question": None, "budget": 60000, "category": "laptop", "usecase": "coding"}
        handlers["Call-A"] = call_a
        fake = FakeGroq(handlers).install()
        tavily = FakeTavily(search=search).install()
        try:
            at = self.app()
            at.chat_input[0].set_value("I want a laptop under 60000").run()
            thread_after_question = at.session_state.thread_id
            at.chat_input[0].set_value("coding").run()
        finally:
            fake.uninstall()
            tavily.uninstall()
        self.assertEqual(at.session_state.thread_id, thread_after_question)
        self.assertEqual([(s["title"], s["status"]) for s in self.store.list()], [("I want a laptop under 60000", "done")])
        state = storage.load_saved_state(self.graph, thread_after_question)
        self.assertEqual(state["user_query"], "I want a laptop under 60000. coding")
        self.assertTrue(state["report"])


if __name__ == "__main__":
    unittest.main()
