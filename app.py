import streamlit as st
from agent import run_pipeline
from tools import DailyQuotaExceeded

st.set_page_config(page_title="Product Research Agent")
st.title("Product Research Agent")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.context = ""
    st.session_state.awaiting_clarification = False

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
                st.session_state.awaiting_clarification = False

        except DailyQuotaExceeded:
            status.empty()
            reply = "I've hit today's free usage limit for the search/AI service. Please come back after it resets (usually within a few hours) and try again."
            st.session_state.awaiting_clarification = False

        except Exception as e:
            status.empty()
            reply = "Sorry, something went wrong while researching this — please try again in a moment."
            st.session_state.awaiting_clarification = False

        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})

if st.session_state.messages:
    if st.sidebar.button("New search"):
        st.session_state.messages = []
        st.session_state.context = ""
        st.session_state.awaiting_clarification = False
        st.rerun()
