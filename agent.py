from prompts import final_chain, prompt3, str_model_call_c, prompt4, str_model_call_d, prompt5, str_model_call_e
from tools import search_tool, get_official_specs, build_query, filter_hallucinated_candidates, filter_by_domain, trim_results, invoke_with_retry, extract_price, search_price_fallback, RETAIL_DOMAINS

user_query = "I want to buy a laptop for coding, budget around 50000 INR"

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

    if not candidates:
        print("No usable candidates found in this search. Consider broadening the query or trying a different search.")
    else:
        for candidate in candidates:
            print(f"--- {candidate['product_name']} ---")
            print("Before:", candidate['known_specs'],
                  "| complete:", candidate['specs_found'])

            if not candidate["specs_found"]:
                follow_up_results = get_official_specs(
                    candidate["product_name"])

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

            print("After: ", candidate['known_specs'],
                  "| complete:", candidate['specs_found'])
            print()

print("=== STEP 5: Price checking ===")
for candidate in candidates:
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
        print(
            f"No price found via extract for {candidate['product_name']}, trying fallback search...")
        fallback_result = search_price_fallback(
            candidate['product_name'], candidate['source_url'])
        call_e_chain = prompt5 | str_model_call_e
        price_result = invoke_with_retry(call_e_chain, {
            'product_name': candidate['product_name'],
            'page_content': fallback_result
        })

    candidate['price'] = price_result['price']
    candidate['availability'] = price_result['availability']
    print(
        f"{candidate['product_name']}: ₹{candidate['price']} ({candidate['availability']})")
