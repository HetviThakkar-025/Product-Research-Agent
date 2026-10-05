from graph import build_graph, make_config
from tools import section_1_start

_graph = build_graph()

# "custom" carries the progress/headline events that nodes send with get_stream_writer
STREAM_MODES = ["updates", "messages", "custom"]


def _result(final_state):
    if "clarify_question" in final_state:
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


def run_pipeline_stream(user_query):
    """
    Same pipeline as run_pipeline, streamed. Yields, in order of arrival:
      {"type": "progress", "text": str}  the progress wording nodes already use, plus Groq rate-limit waits
      {"type": "headline", "text": str}  the Python-built Final Recommendation sentence, just before the report
      {"type": "token", "text": str}     report text as it is generated (report node only, never reasoning),
                                         starting at the section 1 heading like the stored report
      {"type": "final", **result}        last; result is the dict run_pipeline returns
    Exceptions (e.g. DailyQuotaExceeded) propagate to the caller.
    """
    state = {"user_query": user_query}
    held_back, report_started = "", False  # report text is held until its section 1 heading appears
    for mode, chunk in _graph.stream({"user_query": user_query}, config=make_config(), stream_mode=STREAM_MODES):
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
