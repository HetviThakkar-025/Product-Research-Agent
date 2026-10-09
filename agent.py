from graph import build_graph, make_config
from tools import section_1_start
from tracing import trace_metadata

_graph = build_graph()

# "custom" carries the progress/headline events that nodes send with get_stream_writer
STREAM_MODES = ["updates", "messages", "custom"]
# the root run's name in a LangSmith trace
RUN_NAME = "product-research"


def _result(final_state):
    if final_state.get("clarify_question"):
        return {'status': 'clarify', 'question': final_state['clarify_question']}

    return {
        'status': 'done',
        'report': final_state['report'],
        'candidates': final_state['report_candidates'],
        'is_degraded': final_state['is_degraded']
    }


def run_pipeline(user_query, progress_callback=None):
    """
    Runs the full agent pipeline for a single user query.
    progress_callback(str) is called with status updates, if provided — for Streamlit to show live progress.
    Returns a dict: {'status': 'clarify', 'question': str} or {'status': 'done', 'report': str, 'candidates': list}
    Raises DailyQuotaExceeded if Groq's daily token limit is hit — callers should catch this specifically.
    """
    final_state = _graph.invoke({"user_query": user_query},
                                config=make_config(progress_callback))
    return _result(final_state)


def run_pipeline_stream(user_query, previous_question=None, graph=None, thread_id=None, tags=None):
    """
    Same pipeline as run_pipeline, streamed. Yields, in order of arrival:
      {"type": "progress", "text": str}  the progress wording nodes already use, plus Groq rate-limit waits
      {"type": "headline", "text": str}  the Python-built Final Recommendation sentence, just before the report
      {"type": "token", "text": str}     report text as it is generated (report node only, never reasoning),
                                         starting at the section 1 heading like the stored report
      {"type": "final", **result}        last; result is the dict run_pipeline returns
    previous_question is the clarifying question user_query answers, if any (stops the same question being re-asked).
    graph/thread_id: the app passes its checkpointed graph and the search's thread, so the state is saved per thread;
    without them the module's plain graph runs with nothing saved.
    tags label the LangSmith trace, which also gets the thread_id and the start of user_query as metadata.
    Exceptions (e.g. DailyQuotaExceeded) propagate to the caller.
    """
    state = {"user_query": user_query}
    if previous_question:
        state["previous_question"] = previous_question
    held_back, report_started = "", False  # report text is held until its section 1 heading appears
    graph = graph or _graph
    config = make_config(thread_id=thread_id, run_name=RUN_NAME, tags=tags,
                         metadata=trace_metadata(user_query, thread_id))
    for mode, chunk in graph.stream(dict(state), config=config, stream_mode=STREAM_MODES):
        if mode == "custom":
            yield chunk
        elif mode == "messages":
            message, metadata = chunk
            if metadata.get("langgraph_node") == "report" and isinstance(message.content, str) and message.content:
                if report_started:
                    yield {"type": "token", "text": message.content}
                    continue
                held_back += message.content
                start = section_1_start(held_back)
                if start is not None:
                    report_started = True
                    yield {"type": "token", "text": held_back[start:]}
        elif mode == "updates":
            for update in chunk.values():
                state.update(update or {})  # every state key uses the default overwrite reducer
    if held_back and not report_started:
        yield {"type": "token", "text": held_back}  # no section 1 heading: send the report as written
    yield {"type": "final", **_result(state)}


if __name__ == "__main__":
    # keep manual testing possible: python agent.py
    result = run_pipeline(
        "I want to buy a referigerator, budget 60000, family use", progress_callback=print)
    print(result.get('report') or result.get('question'))
