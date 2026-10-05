import streamlit as st
import os
import traceback
import uuid
from agent import run_pipeline_stream
from graph import build_graph
from storage import DB_PATH, SessionStore, load_saved_state, make_checkpointer, open_connection, track_session
from tools import DailyQuotaExceeded

try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    if "TAVILY_API_KEY" in st.secrets:
        os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
except FileNotFoundError:
    pass

st.set_page_config(page_title="Product Research Agent", layout="centered")

# ---------- minimal styling: clean look + fixes tables getting clipped on mobile ----------
st.markdown("""
<style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 760px;
    }
    /* markdown tables were getting cut off / not rendering fully on narrow screens —
       force them to scroll horizontally instead of overflowing hidden */
    .stMarkdown table {
        display: block;
        overflow-x: auto;
        white-space: nowrap;
        max-width: 100%;
    }
    .stMarkdown table td, .stMarkdown table th {
        white-space: normal;
        min-width: 100px;
    }
    div[data-testid="stChatMessage"] {
        padding: 0.4rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_persistence(db_path):
    """One SQLite connection, checkpointer, sessions table and checkpointed graph shared by every session of the app."""
    conn = open_connection(db_path)
    saver = make_checkpointer(conn)
    return build_graph(checkpointer=saver), SessionStore(conn, saver.lock), saver


graph, sessions, checkpointer = get_persistence(os.getenv("AGENT_DB_PATH") or str(DB_PATH))

FOOTER = (
    "\n\n---\n"
    "*This search is complete — I won't treat anything you type next as a follow-up. "
    "Just type a new request below, or hit **New search** in the sidebar to start fresh.*"
)


def start_new_search():
    st.session_state.thread_id = uuid.uuid4().hex
    st.session_state.messages = []
    st.session_state.context = ""
    st.session_state.awaiting_clarification = False
    st.session_state.last_question = None
    st.session_state.viewing = None


def open_saved(thread_id):
    st.session_state.viewing = thread_id


def delete_saved(thread_id):
    sessions.delete(thread_id, checkpointer)
    if st.session_state.viewing == thread_id:
        st.session_state.viewing = None
    if st.session_state.thread_id == thread_id:
        start_new_search()


def show_saved(thread_id):
    """A past search, read-only, from its saved checkpoint: no pipeline run, no Groq or Tavily calls."""
    session = sessions.get(thread_id)
    if session is None:
        st.session_state.viewing = None
        return
    state = load_saved_state(graph, thread_id)
    st.caption(f"Saved search from {session['created_at'][:16].replace('T', ' ')} UTC — read-only. "
               "Type below to start a new search.")
    with st.chat_message("user"):
        st.markdown(state.get("user_query") or session["title"])
    with st.chat_message("assistant"):
        if session["status"] == "done" and state.get("report"):
            if state.get("recommendation_headline"):
                st.markdown(f"**{state['recommendation_headline']}**")
            st.markdown(state["report"], unsafe_allow_html=True)
        elif session["status"] == "clarifying":
            st.info(f"This search stopped at a clarifying question: {state.get('clarify_question') or ''}")
        elif session["status"] == "error":
            st.warning("This search ended with an error, so there is no report.")
        else:
            st.info("This search did not finish, so there is no report.")


if "thread_id" not in st.session_state:
    start_new_search()

st.title("Product Research Agent")

if st.session_state.viewing:
    show_saved(st.session_state.viewing)
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

user_input = st.chat_input("What are you looking to buy?")

if user_input:
    if st.session_state.viewing:
        start_new_search()  # saved searches are read-only: typing starts a fresh one
    if not st.session_state.awaiting_clarification:
        # every new request is its own search (thread); a clarify answer stays in the thread that asked
        st.session_state.thread_id = uuid.uuid4().hex
        st.session_state.context = user_input
        st.session_state.first_message = user_input
    else:
        st.session_state.context += ". " + user_input

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        status_box = st.status("Researching...", expanded=True)
        final = {}

        def report_text(events, first_token=""):
            """Only the report's text, for st.write_stream; progress and the final result are handled on the way."""
            if first_token:
                yield first_token
            for event in events:
                if event["type"] == "token":
                    yield event["text"]
                elif event["type"] == "progress":
                    status_box.write(event["text"])  # e.g. a rate-limit wait while the report is written
                elif event["type"] == "final":
                    final.update(event)

        try:
            previous_question = st.session_state.last_question if st.session_state.awaiting_clarification else None
            thread_id = st.session_state.thread_id
            events = track_session(
                sessions, thread_id, st.session_state.first_message,
                run_pipeline_stream(st.session_state.context, previous_question=previous_question,
                                    graph=graph, thread_id=thread_id))
            headline, first_token = None, ""
            with status_box:
                for event in events:
                    if event["type"] == "progress":
                        st.write(event["text"])
                    elif event["type"] == "headline":
                        headline = event["text"]
                        break
                    elif event["type"] == "token":
                        first_token = event["text"]
                        break
                    elif event["type"] == "final":
                        final.update(event)
                        break

            if final.get("status") == "clarify":
                status_box.update(label="Need one more detail", state="complete", expanded=False)
                reply = final["question"]
                st.markdown(reply, unsafe_allow_html=True)
                st.session_state.awaiting_clarification = True
                st.session_state.last_question = reply
            else:
                status_box.update(label="Writing the report...", state="running", expanded=False)
                if headline:
                    st.markdown(f"**{headline}**")
                streamed = st.write_stream(report_text(events, first_token))
                report = final.get("report") or streamed
                if not streamed:
                    st.markdown(report, unsafe_allow_html=True)
                status_box.update(label="Research complete", state="complete", expanded=False)
                st.markdown(FOOTER)
                reply = (f"**{headline}**\n\n" if headline else "") + report + FOOTER
                st.session_state.awaiting_clarification = False

        except DailyQuotaExceeded:
            status_box.update(label="Daily usage limit reached", state="error", expanded=False)
            reply = "I've hit today's free usage limit for the search/AI service. Please come back after it resets (usually within a few hours) and try again."
            st.markdown(reply, unsafe_allow_html=True)
            st.session_state.awaiting_clarification = False

        except Exception:
            traceback.print_exc()
            status_box.update(label="Something went wrong", state="error", expanded=False)
            reply = "Sorry, something went wrong while researching this — please try again in a moment."
            st.markdown(reply, unsafe_allow_html=True)
            st.session_state.awaiting_clarification = False

    st.session_state.messages.append({"role": "assistant", "content": reply})

# ---------- sidebar: past searches (drawn last, so a search that just finished is already listed) ----------
with st.sidebar:
    st.button("New search", on_click=start_new_search, use_container_width=True, key="new_search")
    st.subheader("Past searches")
    saved = sessions.list()
    if not saved:
        st.caption("No saved searches yet.")
    for session in saved:
        label = session["title"] if session["status"] == "done" else f"{session['title']} ({session['status']})"
        open_col, delete_col = st.columns([5, 1])
        open_col.button(label, key=f"open_{session['thread_id']}", on_click=open_saved, args=(session["thread_id"],),
                        help=f"Saved {session['created_at'][:16].replace('T', ' ')} UTC", use_container_width=True)
        delete_col.button("🗑", key=f"delete_{session['thread_id']}", on_click=delete_saved,
                          args=(session["thread_id"],), help="Delete this search")
    st.caption("History resets when the app restarts.")
