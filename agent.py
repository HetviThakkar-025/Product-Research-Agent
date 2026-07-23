from prompts import final_chain, prompt3, str_model_call_c, prompt4, str_model_call_d, prompt5, str_model_call_e, prompt6, str_model_call_f, report_chain
from tools import search_tool, get_official_specs, build_query, filter_hallucinated_candidates, cap_results, select_report_candidates, suggest_realistic_budget, filter_by_domain, trim_results, invoke_with_retry, extract_price, search_price_fallback, RETAIL_DOMAINS, DailyQuotaExceeded

MAX_ITERATIONS = 4
MIN_QUALIFIED = 2


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


def run_pipeline(user_query, progress_callback=None):
    """
    Runs the full agent pipeline for a single user query.
    progress_callback(str) is called with status updates, if provided — for Streamlit to show live progress.
    Returns a dict: {'status': 'clarify', 'question': str} or {'status': 'done', 'report': str, 'candidates': list}
    Raises DailyQuotaExceeded if Groq's daily token limit is hit — callers should catch this specifically.
    """
    def log(msg):
        if progress_callback:
            progress_callback(msg)

    call_b_output = invoke_with_retry(final_chain, {"query": user_query})

    if isinstance(call_b_output, str):
        return {'status': 'clarify', 'question': call_b_output}

    all_candidates = []
    iteration = 0

    while iteration < MAX_ITERATIONS:
        iteration += 1
        qualified_count = sum(1 for c in all_candidates if is_qualified(c))
        log(f"Iteration {iteration}/{MAX_ITERATIONS} — {qualified_count} qualified so far")

        if qualified_count >= MIN_QUALIFIED:
            log("Enough qualified candidates found.")
            break

        use_negotiable = iteration < 3

        log("Searching Indian retail sites...")
        new_candidates = []
        search_attempts = 2

        for attempt in range(1, search_attempts + 1):
            query = build_query(call_b_result=call_b_output,
                                include_negotiable=use_negotiable)

            search_tool.max_results = 4 + (attempt - 1) * 3
            result = search_tool.invoke({"query": query})
            result['results'] = filter_by_domain(result, RETAIL_DOMAINS)
            result = trim_results(result)
            result = cap_results(result, max_for_llm=5)

            if not result['results']:
                continue

            call_c_chain = prompt3 | str_model_call_c
            candidates_result = invoke_with_retry(call_c_chain, {
                "required_specs": call_b_output['non_negotiable_specs'],
                "raw_results": result
            })

            new_candidates = filter_hallucinated_candidates(
                candidates_result["candidates"], result)

            if new_candidates:
                break

        if not new_candidates:
            log("No new candidates found this iteration.")
            continue

        all_candidates = dedupe_candidates(all_candidates, new_candidates)

        log(f"Verifying specs for {len(new_candidates)} candidate(s)...")
        for candidate in new_candidates:
            if candidate.get('specs_found'):
                continue
            follow_up_results = get_official_specs(candidate["product_name"])

            call_d_chain = prompt4 | str_model_call_d
            newspecs = invoke_with_retry(call_d_chain, {
                'product_name': candidate["product_name"],
                'known_specs': candidate["known_specs"],
                'required_specs': call_b_output['non_negotiable_specs'],
                'follow_up_text': follow_up_results
            })

            for key, value in newspecs['new_specs'].items():
                if key not in candidate['known_specs']:
                    candidate['known_specs'][key] = value

            candidate['specs_found'] = all(
                spec in candidate['known_specs'] for spec in call_b_output['non_negotiable_specs']
            )

        log(f"Checking prices for {len(new_candidates)} candidate(s)...")
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

            if candidate['price'] is None:
                candidate['within_budget'] = "unknown"
            elif call_b_output['budget'] is None:
                candidate['within_budget'] = True
            else:
                candidate['within_budget'] = candidate['price'] <= call_b_output['budget']

        log(f"Scoring fit for {len(new_candidates)} candidate(s)...")
        call_f_chain = prompt6 | str_model_call_f
        for candidate in new_candidates:
            if 'fit_score' in candidate:
                continue

            fit_result = invoke_with_retry(call_f_chain, {
                'non_negotiable_specs': call_b_output['non_negotiable_specs'],
                'negotiable_specs': call_b_output['negotiable_specs'],
                'known_specs': candidate['known_specs']
            })
            candidate['fit_score'] = fit_result['fit_score']
            candidate['reasoning'] = fit_result['reasoning']
            candidate['missing_or_weak_specs'] = fit_result['missing_or_weak_specs']

    final_qualified = [c for c in all_candidates if is_qualified(c)]

    log("Generating final report...")
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

    report = invoke_with_retry(report_chain, {
        'category': call_b_output['category'],
        'usecase': call_b_output['usecase'],
        'budget': call_b_output['budget'],
        'non_negotiable_specs': call_b_output['non_negotiable_specs'],
        'negotiable_specs': call_b_output['negotiable_specs'],
        'candidates': candidates_summary,
        'is_degraded': is_degraded,
        'realistic_budget': realistic_budget
    })

    return {
        'status': 'done',
        'report': report,
        'candidates': report_candidates,
        'is_degraded': is_degraded
    }


if __name__ == "__main__":
    # keep manual testing possible: python agent.py
    result = run_pipeline(
        "I want to buy a laptop for coding, budget around 50000 INR", progress_callback=print)
    print(result.get('report') or result.get('question'))
