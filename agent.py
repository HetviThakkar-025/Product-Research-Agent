from graph import build_graph, make_config

_graph = build_graph()


def run_pipeline(user_query, progress_callback=None):
    """
    Runs the full agent pipeline for a single user query.
    progress_callback(str) is called with status updates, if provided — for Streamlit to show live progress.
    Returns a dict: {'status': 'clarify', 'question': str} or {'status': 'done', 'report': str, 'candidates': list}
    Raises DailyQuotaExceeded if Groq's daily token limit is hit — callers should catch this specifically.
    """
    final_state = _graph.invoke({"user_query": user_query},
                                config=make_config(progress_callback))

    if "clarify_question" in final_state:
        return {'status': 'clarify', 'question': final_state['clarify_question']}

    return {
        'status': 'done',
        'report': final_state['report'],
        'candidates': final_state['report_candidates'],
        'is_degraded': final_state['is_degraded']
    }


if __name__ == "__main__":
    # keep manual testing possible: python agent.py
    result = run_pipeline(
        "I want to buy a referigerator, budget 60000, family use", progress_callback=print)
    print(result.get('report') or result.get('question'))
