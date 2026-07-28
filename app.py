import streamlit as st
import os
from agent import run_pipeline
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

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.context = ""
    st.session_state.awaiting_clarification = False

# ---------- header row: title + reset button, no sidebar ----------
header_col1, header_col2 = st.columns([4, 1])
with header_col1:
    st.title("Product Research Agent")
with header_col2:
    st.write("")  # small vertical spacer to align button with title
    st.write("")
    if st.button("New search", use_container_width=True):
        st.session_state.messages = []
        st.session_state.context = ""
        st.session_state.awaiting_clarification = False
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("What are you looking to buy?")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if st.session_state.awaiting_clarification:
        st.session_state.context += ". " + user_input
    else:
        st.session_state.context = user_input

    with st.chat_message("assistant"):
        status = st.empty()

        def update_status(msg):
            status.write(msg)

        try:
            result = run_pipeline(st.session_state.context,
                                  progress_callback=update_status)
            status.empty()

            if result['status'] == 'clarify':
                reply = result['question']
                st.session_state.awaiting_clarification = True
            else:
                reply = result['report']
                reply += (
                    "\n\n---\n"
                    "*This search is complete — I won't treat anything you type next as a follow-up. "
                    "Just type a new request below, or hit **New search** above to start fresh.*"
                )
                st.session_state.awaiting_clarification = False

        except DailyQuotaExceeded:
            status.empty()
            reply = "I've hit today's free usage limit for the search/AI service. Please come back after it resets (usually within a few hours) and try again."
            st.session_state.awaiting_clarification = False

        except Exception as e:
            status.empty()
            reply = "Sorry, something went wrong while researching this — please try again in a moment."
            st.session_state.awaiting_clarification = False

        st.markdown(reply, unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": reply})
