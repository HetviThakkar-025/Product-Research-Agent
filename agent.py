from prompts import final_chain, prompt3, str_model_call_c, prompt4, str_model_call_d
from tools import search_tool, get_official_specs, build_query, filter_hallucinated_candidates, filter_by_domain, trim_results, invoke_with_retry, RETAIL_DOMAINS

user_query = "I want to buy a laptop for gaming, budget around 80000 INR"

# ---------- Step 1 ----------
call_b_output = invoke_with_retry(final_chain, {"query": user_query})
print("=== STEP 1: Requirements ===")
print(call_b_output)
print()

if isinstance(call_b_output, str):
    print("Clarifying question needed:", call_b_output)
else:
    # ---------- Step 2 ----------
    query = build_query(call_b_result=call_b_output)
    print("=== STEP 2: Tavily Query ===")
    print(query)
    print()

    result = search_tool.invoke({"query": query})
    result['results'] = filter_by_domain(result, RETAIL_DOMAINS)
    result = trim_results(result)
    print("=== STEP 2: Filtered Raw Results ===")
    for r in result['results']:
        print(r['url'])
    print()

    # ---------- Step 3 ----------
    call_c_chain = prompt3 | str_model_call_c
    candidates_result = invoke_with_retry(call_c_chain, {
        "required_specs": call_b_output['non_negotiable_specs'],
        "raw_results": result
    })
    print("=== STEP 3: Extracted Candidates (before hallucination filter) ===")
    print(candidates_result)
    print()

    candidates = filter_hallucinated_candidates(
        candidates_result["candidates"], result)
    print(
        f"=== STEP 3: {len(candidates)} candidates survived hallucination filter ===")
    print()

    # ---------- Step 4 ----------
    print("=== STEP 4: Filling incomplete specs ===")
    for candidate in candidates:
        print(f"--- {candidate['product_name']} ---")
        print("Before:", candidate['known_specs'],
              "| complete:", candidate['specs_complete'])

        if not candidate["specs_complete"]:
            follow_up_results = get_official_specs(candidate["product_name"])

            call_d_chain = prompt4 | str_model_call_d
            newspecs = invoke_with_retry(call_d_chain, {
                'product_name': candidate["product_name"],
                'known_specs': candidate["known_specs"],
                'required_specs': call_b_output['non_negotiable_specs'],
                'follow_up_text': follow_up_results
            })

            candidate['known_specs'].update(newspecs['new_specs'])
            candidate['specs_complete'] = all(
                spec in candidate['known_specs'] for spec in call_b_output['non_negotiable_specs']
            )

        print("After: ", candidate['known_specs'],
              "| complete:", candidate['specs_complete'])
        print()
