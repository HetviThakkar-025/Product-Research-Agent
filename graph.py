import copy
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from prompts import call_a_chain, branch_chain, prompt3, str_model_call_c, prompt4, str_model_call_d, prompt5, str_model_call_e, prompt6, str_model_call_f, report_chain
from tools import search_tool, get_official_specs, build_query, drop_spec_from_query, request_keywords, drop_budget_specs, drop_empty_specs, filter_hallucinated_candidates, cap_results, select_report_candidates, recommendation_headline, budget_note, realistic_budget_text, format_inr, spec_status, gap_advice, report_issues, trim_to_section_1, suggest_realistic_budget, filter_by_domain, trim_results, drop_repeated_title, invoke_with_retry, extract_price, search_price_fallback, extract_price_snippets, rupee_amounts, attribute_prices, find_search_snippet, is_product_page_url, normalize_product_url, model_numbers, product_match, RETAIL_DOMAINS, DailyQuotaExceeded, emit_event, emit_progress

MAX_ITERATIONS = 4
MIN_QUALIFIED = 2
SEARCH_ATTEMPTS = 2
SEARCH_MAX_RESULTS = 7  # first attempt; each retry asks for 3 more

# worst case is ~38 node runs: intake + report + 4 iterations x
# (start_iteration + 2 x (search + extract_candidates) + verify/price/score/merge)
RECURSION_LIMIT = 50

# a price outside this multiple of the budget is taken to belong to another product (or be an EMI amount)
PLAUSIBLE_PRICE_RANGE = (0.25, 3)


class AgentState(TypedDict, total=False):
    user_query: str
    previous_question: str         # the clarifying question this query answers, if any (input)
    requirements: dict             # Call B output
    clarify_question: str          # set only when Call A says the query is unclear
    iteration: int
    search_attempt: int            # 1..SEARCH_ATTEMPTS within the current iteration
    iteration_query: str           # this iteration's LLM query rewrite, reused by its later search attempts
    search_query: str              # query text actually searched by the latest attempt
    search_results: dict           # trimmed/capped results, fed to Call C and the hallucination check
    search_snippets: dict          # clean url -> full (untrimmed) search content, the primary price source
    new_candidates: list
    all_candidates: list
    report: str
    report_candidates: list
    is_degraded: bool
    recommendation_headline: str   # the Python-built Final Recommendation sentence, kept for reopening a saved search


def is_qualified(candidate):
    return (
        candidate.get('specs_found')
        and candidate.get('within_budget') is True
        and candidate.get('fit_score', 0) >= 7
    )


def spec_key_names(requirements):
    """Call B's spec keys (non-negotiable then negotiable), which Calls C and D are told to reuse verbatim."""
    return list(requirements['non_negotiable_specs']) + list(requirements.get('negotiable_specs') or {})


def duplicate_of(candidate, known):
    """(existing candidate, reason) if candidate is the same product as one in known, else None."""
    url_key = normalize_product_url(candidate['source_url'])
    models = model_numbers(candidate['product_name'])
    name = candidate['product_name'].lower()

    for other in known:
        if normalize_product_url(other['source_url']) == url_key:
            return other, 'url'
        if models & model_numbers(other['product_name']):
            return other, 'model number'
        if other['product_name'].lower() == name:
            return other, 'name'
    return None


def dedupe_candidates(existing, new):
    for c in new:
        if not duplicate_of(c, existing):
            existing.append(c)
    return existing


def _log(config, msg):
    callback = (config or {}).get("configurable", {}).get("progress_callback")
    if callback:
        callback(msg)
    emit_progress(msg)  # progress event for run_pipeline_stream; no-op under invoke


# ---------- nodes ----------

# assumed when a clarifying question gets a budget but no use case; prompt A itself counts it as a stated use case
DEFAULT_USECASE = "general everyday use"


def intake(state, config):
    # Call A, then the branch (question, or Call B): the same steps as final_chain, split so Call A's output can be checked
    call_a_output = invoke_with_retry(call_a_chain, {"query": state["user_query"]})

    # Answering a clarifying question with only a budget made Call A ask the same use-case question again
    if (state.get("previous_question") and call_a_output.get("status") == "unclear"
            and call_a_output.get("budget") is not None and not call_a_output.get("usecase")
            and call_a_output.get("category")):
        print(f"Clarify loop avoided: no use case given after {state['previous_question']!r}; "
              f"assuming {DEFAULT_USECASE!r}")
        call_a_output = {**call_a_output, "status": "clear", "usecase": DEFAULT_USECASE}

    # Call B also gets the user's own words, so values they state (250 L, double door) are kept as given
    call_b_output = invoke_with_retry(branch_chain, {**call_a_output, "request": state["user_query"]})

    if isinstance(call_b_output, str):
        return {"clarify_question": call_b_output}
    if isinstance(call_b_output, dict) and "non_negotiable_specs" in call_b_output:
        # null/empty spec values first, before anything (budget filter, search, scoring) uses the specs
        call_b_output = drop_budget_specs(drop_empty_specs(call_b_output))

    # clarify_question cleared: with a checkpointer, a clarify answer reuses the thread whose state still holds the question
    return {"requirements": call_b_output, "iteration": 0, "all_candidates": [], "clarify_question": None}


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

    if attempt == 1:
        # one rewrite per iteration; later attempts reuse it with one core spec dropped
        iteration_query = build_query(call_b_result=state["requirements"],
                                      include_negotiable=use_negotiable, user_query=state["user_query"])
        query = iteration_query
    else:
        iteration_query = state["iteration_query"]
        # the user's own words stay in every retry
        user_words = request_keywords(state["user_query"], state["requirements"]["category"])
        query, change = drop_spec_from_query(iteration_query, state["requirements"], keep_words=user_words)
        print(f"Search attempt {attempt}: {change} in {iteration_query!r} -> {query!r}")

    search_tool.max_results = SEARCH_MAX_RESULTS + (attempt - 1) * 3
    result = search_tool.invoke({"query": query})
    result['results'] = filter_by_domain(result, RETAIL_DOMAINS)
    search_snippets = {r['url'].split('?')[0]: r.get('content') or '' for r in result['results']}
    result = trim_results(result)
    result = drop_repeated_title(result)
    result = cap_results(result, max_for_llm=5)

    return {"search_attempt": attempt, "search_results": result, "search_snippets": search_snippets,
            "iteration_query": iteration_query, "search_query": query, "new_candidates": []}


def extract_candidates(state, config):
    result = state["search_results"]
    call_c_chain = prompt3 | str_model_call_c
    candidates_result = invoke_with_retry(call_c_chain, {
        "required_specs": state["requirements"]['non_negotiable_specs'],
        "raw_results": result,
        "spec_keys": spec_key_names(state["requirements"])
    })

    verified = filter_hallucinated_candidates(
        candidates_result["candidates"], result)

    # dedupe before verify/price/score so a product already seen costs no further calls
    all_candidates = copy.deepcopy(state["all_candidates"])
    new_candidates = []
    for candidate in verified:
        match = duplicate_of(candidate, all_candidates + new_candidates)
        if match:
            other, reason = match
            print(f"Dropped duplicate candidate: {candidate['product_name']} (same {reason} as {other['product_name']})")
            # keep the duplicate's model numbers on the kept candidate, for matching price sources to it
            other['model_numbers'] = sorted(set(other.get('model_numbers', [])) | model_numbers(candidate['product_name']))
            continue
        new_candidates.append(candidate)

    for candidate in new_candidates:
        candidate['search_snippet'] = find_search_snippet(
            candidate['source_url'], state.get("search_snippets", {}))
        # model numbers in the candidate's own result title are its own, for telling its price from others' on the page
        own_title = find_search_snippet(candidate['source_url'], {r['url']: r.get('title', '') for r in result['results']})
        if model_numbers(own_title):
            candidate['model_numbers'] = sorted(set(candidate.get('model_numbers', [])) | model_numbers(own_title))

    return {"new_candidates": new_candidates, "all_candidates": all_candidates}


def verify_specs(state, config):
    requirements = state["requirements"]
    new_candidates = copy.deepcopy(state["new_candidates"])

    _log(config, f"Verifying specs for {len(new_candidates)} candidate(s)...")
    for candidate in new_candidates:
        if candidate.get('specs_found'):
            continue
        follow_up_results = get_official_specs(candidate["product_name"])

        if follow_up_results['results']:
            call_d_chain = prompt4 | str_model_call_d
            newspecs = invoke_with_retry(call_d_chain, {
                'product_name': candidate["product_name"],
                'known_specs': candidate["known_specs"],
                'required_specs': requirements['non_negotiable_specs'],
                'follow_up_text': follow_up_results,
                'spec_keys': spec_key_names(requirements)
            })

            # non-destructive merge: never overwrite a spec we already have, whatever its key's casing
            known_keys = {key.lower() for key in candidate['known_specs']}
            for key, value in newspecs['new_specs'].items():
                if key.lower() not in known_keys:
                    candidate['known_specs'][key] = value
                    known_keys.add(key.lower())
        else:
            print(f"Skipped Call D for {candidate['product_name']}: spec search returned nothing")

        # case-insensitive, so "Processor" from Call C/D still counts for Call B's "processor"
        found_keys = {key.lower() for key in candidate['known_specs']}
        candidate['specs_found'] = all(
            spec.lower() in found_keys for spec in requirements['non_negotiable_specs']
        )

    return {"new_candidates": new_candidates}


def _price_sources(candidate):
    """(label, fetch) in priority order; fetch() returns [(url, text)] and only runs if earlier sources found no price."""
    source_url = candidate['source_url']
    return [
        ('search_snippet', lambda: [(source_url, candidate.get('search_snippet', ''))]),
        ('page_extract', lambda: [(source_url, extract_price(source_url))]),
        ('fallback_search', lambda: [(r['url'], r['content']) for r in _matching_fallback_results(candidate)]),
    ]


def _attributed_sources(candidate, label, texts):
    """
    [(url, ₹ snippets, amounts)] built only from the ₹ amounts attribute_prices ties to this candidate.
    Rejected amounts are kept in full on candidate['price_rejections'] (for the run file) and logged as one summary line.
    """
    sources = []
    rejections = []
    for url, text in texts:
        kept, rejected = attribute_prices(candidate, text)
        for amount, reason in dict.fromkeys(rejected):  # "₹X₹X" repeats are counted once
            rejections.append({'source': label, 'url': url, 'amount': amount, 'reason': reason})
        if kept:
            amounts = set().union(*(rupee_amounts(m.group()) for m in kept))
            sources.append((url, extract_price_snippets(text, matches=kept), amounts))

    if rejections:
        candidate.setdefault('price_rejections', []).extend(rejections)
        print(f"Rejected prices for {candidate['product_name']} ({label}): {summarize_rejections(rejections)}")
    return sources


def summarize_rejections(rejections, examples=3):
    """'12 product not named nearby (e.g. 71800, 71790, 85000), 2 other model nearby (e.g. 299, 69409)'."""
    by_reason = {}
    for r in rejections:
        by_reason.setdefault(r['reason'].split(' (')[0], []).append(r['amount'])
    return ", ".join(f"{len(amounts)} {reason} (e.g. {', '.join(str(a) for a in amounts[:examples])})"
                     for reason, amounts in by_reason.items())


def _matching_fallback_results(candidate):
    """Fallback search results tied to this candidate by product_match; priced ones that aren't are logged and skipped."""
    matching = []
    for r in search_price_fallback(candidate['product_name']):
        if product_match(candidate, r['url'], r['title'], r['content']):
            matching.append(r)
        elif rupee_amounts(r['content']):
            print(f"Rejected price: product mismatch for {candidate['product_name']}: {r['url']} ({r['title'][:80]})")
    return matching


def is_plausible_price(price, budget):
    if budget is None:
        return True
    low, high = PLAUSIBLE_PRICE_RANGE
    return low * budget <= price <= high * budget


def _price_from_snippets(product_name, sources):
    """
    Call E on the ₹ snippets from sources [(url, snippets, amounts)]. The price is accepted only if it is one of
    the ₹ amounts attributed to the candidate in a source, so the LLM can never invent one or pick another product's.
    Returns (E result with price possibly nulled, url the price came from).
    """
    call_e_chain = prompt5 | str_model_call_e
    price_result = invoke_with_retry(call_e_chain, {
        'product_name': product_name,
        'page_content': "\n---\n".join(snippets for _, snippets, _ in sources)
    })

    price = price_result.get('price')
    if price is None:
        return price_result, None

    for url, _, amounts in sources:
        if price in amounts:
            return price_result, url

    print(f"Rejected price {price} for {product_name}: not one of the ₹ amounts attributed to it")
    return {**price_result, 'price': None}, None


def check_prices(state, config):
    requirements = state["requirements"]
    new_candidates = copy.deepcopy(state["new_candidates"])

    _log(config, f"Checking prices for {len(new_candidates)} candidate(s)...")
    for candidate in new_candidates:
        price_result = {'price': None, 'availability': 'unknown'}
        price_source, price_source_url = None, None
        tried = []

        # each source is tried on its own, so a failing extract no longer skips the fallback
        for label, fetch in _price_sources(candidate):
            tried.append(label)
            try:
                sources = _attributed_sources(candidate, label, [(url, text) for url, text in fetch() if text])
                if not sources:
                    continue
                result, url = _price_from_snippets(candidate['product_name'], sources)
            except DailyQuotaExceeded:
                raise
            except Exception as e:
                print(f"Price source '{label}' failed for {candidate['product_name']}: {type(e).__name__}: {e}")
                continue

            price_result = result
            if result['price'] is not None and not is_plausible_price(result['price'], requirements['budget']):
                low, high = PLAUSIBLE_PRICE_RANGE
                print(f"Rejected implausible price {result['price']} for {candidate['product_name']}: "
                      f"outside {low}x-{high}x of budget {requirements['budget']} ({label}: {url})")
                price_result = {**result, 'price': None}
                continue
            if result['price'] is not None:
                price_source, price_source_url = label, url
                break

        print(f"Price source for {candidate['product_name']}: {price_source or 'none'} "
              f"(price={price_result['price']}, tried: {', '.join(tried)})")

        candidate['price'] = price_result['price']
        candidate['availability'] = price_result['availability']
        candidate['price_source'] = price_source
        candidate['price_source_url'] = price_source_url

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
    report_candidates = select_report_candidates(all_candidates, top_n=4)
    is_degraded = len(final_qualified) < MIN_QUALIFIED

    # only from candidates meeting every required spec; None means the report must not state a budget figure
    realistic_budget = suggest_realistic_budget(all_candidates) if is_degraded else None

    # decided in Python from verified prices, so the model can't crown an unpriced candidate
    headline = recommendation_headline(all_candidates, requirements['budget'])
    # budget problem vs search problem, also decided in Python
    gap_kind, advice = gap_advice(all_candidates, requirements['budget'])
    if is_degraded:
        print(f"Budget gap: {gap_kind}")
    emit_event({"type": "headline", "text": headline})  # shown before the report text streams

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
            'source_url': c.get('source_url'),
            'price_source': c.get('price_source'),
            'price_source_url': c.get('price_source_url'),
            'budget_note': budget_note(c, requirements['budget']),
            'spec_status': spec_status(c)
        })

    report_text = invoke_with_retry(report_chain, {
        'category': requirements['category'],
        'usecase': requirements['usecase'],
        'budget': format_inr(requirements['budget']) if requirements['budget'] is not None else "no budget limit",
        'non_negotiable_specs': requirements['non_negotiable_specs'],
        'negotiable_specs': requirements['negotiable_specs'],
        'candidates': candidates_summary,
        'is_degraded': is_degraded,
        'realistic_budget': realistic_budget_text(realistic_budget, requirements['budget']) if is_degraded else "not needed",
        'gap_advice': advice if is_degraded else "not needed",
        'recommendation_headline': headline
    })
    for issue in report_issues(report_text, headline):
        print(f"Report check: {issue}")
    # the app shows the headline above the report, so nothing may precede section 1 (e.g. a repeated headline)
    report_text = trim_to_section_1(report_text)

    return {
        "report": report_text,
        "report_candidates": report_candidates,
        "is_degraded": is_degraded,
        "recommendation_headline": headline
    }


# ---------- conditional edges ----------

def route_after_intake(state):
    return END if state.get("clarify_question") else "start_iteration"


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
    results = state["search_results"]['results']
    if any(is_product_page_url(r['url']) for r in results):
        return "extract_candidates"
    if results:
        # Call C would only find listing-page candidates, which the hallucination check drops anyway
        print(f"Skipped Call C: none of the {len(results)} search results is a product page")
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


def make_config(progress_callback=None, thread_id=None, run_name=None, tags=None, metadata=None):
    """
    Run config; thread_id selects the checkpointer thread. The callback is only added when given (it isn't serializable).
    run_name/tags/metadata label the run's LangSmith trace; they don't change what the graph does.
    """
    configurable = {}
    if progress_callback:
        configurable["progress_callback"] = progress_callback
    if thread_id:
        configurable["thread_id"] = thread_id
    config = {"recursion_limit": RECURSION_LIMIT, "configurable": configurable}
    if run_name:
        config["run_name"] = run_name
    if tags:
        config["tags"] = list(tags)
    if metadata:
        config["metadata"] = dict(metadata)
    return config
