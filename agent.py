from prompts import final_chain, prompt3, str_model_call_c
from tools import search_tool, get_official_specs, build_query

user_query = "I want to buy a laptop for Machine Learning work, budget around 80000 INR"

# Step 1 → LLM translates use case into specific technical requirements
call_b_output = final_chain.invoke({"query": user_query})

if isinstance(call_b_output, str):
    print("que")
else:
    query = build_query(call_b_result=call_b_output)

    #  Step 2 → Tool 1 searches Indian market for matching products (Tavily)
    result = search_tool.invoke({"query": query})

    #  Step 3 → LLM extracts specs from raw search results
    call_c_chain = prompt3 | str_model_call_c

    candidates_result = call_c_chain.invoke({"required_specs": call_b_output['non_negotiable_specs'],
                                            "raw_results": result})

    print(candidates_result)

    # Step 4 → If specs incomplete → targeted follow-up search for official specs
    candidates = candidates_result["candidates"]

    for candidate in candidates:
        if not candidate["specs_complete"]:
            results = get_official_specs(candidate["product_name"])
        print(results)
