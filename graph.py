import copy
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from prompts import final_chain, prompt3, str_model_call_c, prompt4, str_model_call_d, prompt5, str_model_call_e, prompt6, str_model_call_f, report_chain
from tools import search_tool, get_official_specs, build_query, filter_hallucinated_candidates, cap_results, select_report_candidates, suggest_realistic_budget, filter_by_domain, trim_results, invoke_with_retry, extract_price, search_price_fallback, RETAIL_DOMAINS, DailyQuotaExceeded

MAX_ITERATIONS = 4
MIN_QUALIFIED = 2
SEARCH_ATTEMPTS = 2

# worst case is ~38 node runs: intake + report + 4 iterations x
# (start_iteration + 2 x (search + extract_candidates) + verify/price/score/merge)
RECURSION_LIMIT = 50


class AgentState(TypedDict, total=False):
    user_query: str
    requirements: dict             # Call B output
    clarify_question: str          # set only when Call A says the query is unclear
    iteration: int
    search_attempt: int            # 1..SEARCH_ATTEMPTS within the current iteration
    search_results: dict           # trimmed/capped results, fed to Call C and the hallucination check
    new_candidates: list
    all_candidates: list
    report: str
    report_candidates: list
    is_degraded: bool


def is_qualified(candidate):
    return (
        candidate.get('specs_found')
        and candidate.get('within_budget') is True
        and candidate.get('fit_score', 0) >= 7
    )


def dedupe_candidates(existing, new):
    known_names = {c['product_name'].lower() for c in existing}
    for c in new:
        if c['product_name'].lower() not in known_names:
            existing.append(c)
            known_names.add(c['product_name'].lower())
    return existing


def _log(config, msg):
    callback = (config or {}).get("configurable", {}).get("progress_callback")
    if callback:
        callback(msg)


# ---------- nodes ----------

def intake(state, config):
    call_b_output = invoke_with_retry(final_chain, {"query": state["user_query"]})

    if isinstance(call_b_output, str):
        return {"clarify_question": call_b_output}

    return {"requirements": call_b_output, "iteration": 0, "all_candidates": []}


def start_iteration(state, config):
    iteration = state["iteration"] + 1
    qualified_count = sum(1 for c in state["all_candidates"] if is_qualified(c))
    _log(config, f"Iteration {iteration}/{MAX_ITERATIONS} — {qualified_count} qualified so far")
    _log(config, "Searching Indian retail sites...")
    return {"iteration": iteration, "search_attempt": 0, "new_candidates": []}


def search(state, config):
    attempt = state["search_attempt"] + 1
    # iteration-based relaxation: negotiable specs only nudge the query in iterations 1-2
    use_negotiable = state["iteration"] < 3

    query = build_query(call_b_result=state["requirements"],
                        include_negotiable=use_negotiable)

    search_tool.max_results = 4 + (attempt - 1) * 3
    result = search_tool.invoke({"query": query})
    result['results'] = filter_by_domain(result, RETAIL_DOMAINS)
    result = trim_results(result)
    result = cap_results(result, max_for_llm=5)

    return {"search_attempt": attempt, "search_results": result, "new_candidates": []}


def extract_candidates(state, config):
    result = state["search_results"]
    call_c_chain = prompt3 | str_model_call_c
    candidates_result = invoke_with_retry(call_c_chain, {
        "required_specs": state["requirements"]['non_negotiable_specs'],
        "raw_results": result
    })

    new_candidates = filter_hallucinated_candidates(
        candidates_result["candidates"], result)
    return {"new_candidates": new_candidates}


def verify_specs(state, config):
    requirements = state["requirements"]
    new_candidates = copy.deepcopy(state["new_candidates"])

    _log(config, f"Verifying specs for {len(new_candidates)} candidate(s)...")
    for candidate in new_candidates:
        if candidate.get('specs_found'):
            continue
        follow_up_results = get_official_specs(candidate["product_name"])

        call_d_chain = prompt4 | str_model_call_d
        newspecs = invoke_with_retry(call_d_chain, {
            'product_name': candidate["product_name"],
            'known_specs': candidate["known_specs"],
            'required_specs': requirements['non_negotiable_specs'],
            'follow_up_text': follow_up_results
        })

        # non-destructive merge: never overwrite a spec we already have
        for key, value in newspecs['new_specs'].items():
            if key not in candidate['known_specs']:
                candidate['known_specs'][key] = value

        candidate['specs_found'] = all(
            spec in candidate['known_specs'] for spec in requirements['non_negotiable_specs']
        )

    return {"new_candidates": new_candidates}


def check_prices(state, config):
    requirements = state["requirements"]
    new_candidates = copy.deepcopy(state["new_candidates"])

    _log(config, f"Checking prices for {len(new_candidates)} candidate(s)...")
    for candidate in new_candidates:
        try:
            page_content = extract_price(candidate['source_url'])

            if page_content:
                call_e_chain = prompt5 | str_model_call_e
                price_result = invoke_with_retry(call_e_chain, {
                    'product_name': candidate['product_name'],
                    'page_content': page_content
                })
            else:
                price_result = {'price': None, 'availability': 'unknown'}

            if price_result['price'] is None:
                fallback_result = search_price_fallback(
                    candidate['product_name'], candidate['source_url'])
                call_e_chain = prompt5 | str_model_call_e
                price_result = invoke_with_retry(call_e_chain, {
                    'product_name': candidate['product_name'],
                    'page_content': fallback_result
                })
        except DailyQuotaExceeded:
            raise
        except Exception as e:
            price_result = {'price': None, 'availability': 'unknown'}

        candidate['price'] = price_result['price']
        candidate['availability'] = price_result['availability']

        # Python-side budget check
        if candidate['price'] is None:
            candidate['within_budget'] = "unknown"
        elif requirements['budget'] is None:
            candidate['within_budget'] = True
        else:
            candidate['within_budget'] = candidate['price'] <= requirements['budget']

    return {"new_candidates": new_candidates}


def score_fit(state, config):
    requirements = state["requirements"]
    new_candidates = copy.deepcopy(state["new_candidates"])

    _log(config, f"Scoring fit for {len(new_candidates)} candidate(s)...")
    call_f_chain = prompt6 | str_model_call_f
    for candidate in new_candidates:
        if 'fit_score' in candidate:
            continue

        fit_result = invoke_with_retry(call_f_chain, {
            'non_negotiable_specs': requirements['non_negotiable_specs'],
            'negotiable_specs': requirements['negotiable_specs'],
            'known_specs': candidate['known_specs']
        })
        candidate['fit_score'] = fit_result['fit_score']
        candidate['reasoning'] = fit_result['reasoning']
        candidate['missing_or_weak_specs'] = fit_result['missing_or_weak_specs']

    return {"new_candidates": new_candidates}


def merge(state, config):
    all_candidates = dedupe_candidates(
        list(state["all_candidates"]), state["new_candidates"])
    return {"all_candidates": all_candidates}


def report(state, config):
    requirements = state["requirements"]
    all_candidates = state["all_candidates"]
    final_qualified = [c for c in all_candidates if is_qualified(c)]

    _log(config, "Generating final report...")
    report_candidates = select_report_candidates(all_candidates, top_n=3)
    is_degraded = len(final_qualified) < MIN_QUALIFIED

    realistic_budget = None
    if is_degraded:
        realistic_budget = suggest_realistic_budget(report_candidates)

    candidates_summary = []
    for c in report_candidates:
        candidates_summary.append({
            'product_name': c['product_name'],
            'price': c.get('price'),
            'within_budget': c.get('within_budget'),
            'fit_score': c.get('fit_score'),
            'known_specs': c.get('known_specs'),
            'missing_or_weak_specs': c.get('missing_or_weak_specs'),
            'reasoning': c.get('reasoning'),
            'source_url': c.get('source_url')
        })

    report_text = invoke_with_retry(report_chain, {
        'category': requirements['category'],
        'usecase': requirements['usecase'],
        'budget': requirements['budget'],
        'non_negotiable_specs': requirements['non_negotiable_specs'],
        'negotiable_specs': requirements['negotiable_specs'],
        'candidates': candidates_summary,
        'is_degraded': is_degraded,
        'realistic_budget': realistic_budget
    })

    return {
        "report": report_text,
        "report_candidates": report_candidates,
        "is_degraded": is_degraded
    }


# ---------- conditional edges ----------

def route_after_intake(state):
    return END if "clarify_question" in state else "start_iteration"


def route_next_iteration(state, config):
    """Step 8 stop rule: iteration cap first, then 2+ qualified products."""
    if state["iteration"] >= MAX_ITERATIONS:
        return "report"
    if sum(1 for c in state["all_candidates"] if is_qualified(c)) >= MIN_QUALIFIED:
        _log(config, "Enough qualified candidates found.")
        return "report"
    return "start_iteration"


def _retry_search_or_move_on(state, config):
    """Empty-search retry: search again, or give up on this iteration."""
    if state["search_attempt"] < SEARCH_ATTEMPTS:
        return "search"
    _log(config, "No new candidates found this iteration.")
    return route_next_iteration(state, config)


def route_after_search(state, config):
    if state["search_results"]['results']:
        return "extract_candidates"
    return _retry_search_or_move_on(state, config)


def route_after_extract(state, config):
    if state["new_candidates"]:
        return "verify_specs"
    return _retry_search_or_move_on(state, config)


def build_graph(checkpointer=None):
    builder = StateGraph(AgentState)

    builder.add_node("intake", intake)
    builder.add_node("start_iteration", start_iteration)
    builder.add_node("search", search)
    builder.add_node("extract_candidates", extract_candidates)
    builder.add_node("verify_specs", verify_specs)
    builder.add_node("check_prices", check_prices)
    builder.add_node("score_fit", score_fit)
    builder.add_node("merge", merge)
    builder.add_node("report", report)

    builder.add_edge(START, "intake")
    builder.add_conditional_edges(
        "intake", route_after_intake, ["start_iteration", END])
    builder.add_edge("start_iteration", "search")
    builder.add_conditional_edges(
        "search", route_after_search,
        ["extract_candidates", "search", "start_iteration", "report"])
    builder.add_conditional_edges(
        "extract_candidates", route_after_extract,
        ["verify_specs", "search", "start_iteration", "report"])
    builder.add_edge("verify_specs", "check_prices")
    builder.add_edge("check_prices", "score_fit")
    builder.add_edge("score_fit", "merge")
    builder.add_conditional_edges(
        "merge", route_next_iteration, ["start_iteration", "report"])
    builder.add_edge("report", END)

    return builder.compile(checkpointer=checkpointer)


def make_config(progress_callback=None):
    return {
        "recursion_limit": RECURSION_LIMIT,
        "configurable": {"progress_callback": progress_callback},
    }
