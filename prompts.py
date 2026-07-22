from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
# Call A (status/question), Call E (price/availability)
llm_small = ChatGroq(model="openai/gpt-oss-120b", max_tokens=500)
# Call B, Call D
llm_medium = ChatGroq(model="openai/gpt-oss-120b",
                      max_tokens=1200)
# Call C (up to 4 full candidates)
llm_large = ChatGroq(model="openai/gpt-oss-120b", max_tokens=2000)

parser = StrOutputParser()

call_a = {
    "title": "Call-A",
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["unclear", "clear"],
            "description": "Status about whether user query specifies use case and budget of product or not, by default status is unclear, if both yes then status is clear, if both no or one of them is no then status is unclear"
        },
        "question": {
            "type": ["string", "null"],
            "description": "User query about product"
        },
        "usecase": {
            "type": ["string", "null"],
            "description": "Use case of product specified by user query"
        },
        "budget": {
            "type": ["integer", "null"],
            "description": "Budget in INR specified by user. Use null if user explicitly stated no budget limit / money is not a constraint."
        },
        "category": {
            "type": ["string", "null"],
            "description": "Product name given by user"
        }
    },
    "required": ["status"]
}

call_b = {
    "title": "Call-B",
    "type": "object",
    "properties": {
        "usecase": {
            "type": "string",
            "description": "Use case of product specified by user query"
        },
        "budget": {
            "type": ["integer", "null"],
            "description": "Budget in INR specified by user. Use null if user explicitly stated no budget limit / money is not a constraint."
        },
        "category": {
            "type": "string",
            "description": "Product category"
        },
        "non_negotiable_specs": {
            "type": "object",
            "additionalProperties": {"type": "string"},
            "description": "Dict of spec name to required value, for specs that must be met"
        },
        "negotiable_specs": {
            "type": ["object", "null"],
            "description": "Dict of spec name to required value, for specs that are less important which can be removed if there are budget constraints, null if all specs are important",
            "additionalProperties": {"type": "string"}
        }
    },
    "required": ["usecase", "budget", "non_negotiable_specs"]
}

# "candidate" means: a product that might be a good recommendation, extracted from messy search results,
# but not yet verified as complete or price-checked.
call_c = {  # Step 3: candidate extraction
    "title": "Candidate-Extraction",
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                        "description": "Exact, specific product name/model as it appears in results (brand + model number if available)"
                    },
                    "known_specs": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Any specs visible in the search result for this product"
                    },
                    "source_url": {
                        "type": "string",
                        "description": "URL where this product was found"
                    },
                    "specs_found": {  # renamed from specs_complete
                        "type": "boolean",
                        "description": "True only if a value exists for every required non-negotiable spec key for this product; does NOT mean the values meet the requirement, only that data was found."
                    },
                },
                "required": ["product_name", "known_specs", "specs_found"]
            }
        }
    },
    "required": ["candidates"]
}

call_d = {
    "title": "Spec-Merge",
    "type": "object",
    "properties": {
        "new_specs": {
            "type": "object",
            "additionalProperties": {"type": "string"},
            "description": "Dict of spec name to value, containing ONLY specs found in the follow-up text that are not already present in known_specs. Do not repeat specs already known."
        }
    },
    "required": ["new_specs"]
}

call_e = {
    "title": "Price-Extraction",
    "type": "object",
    "properties": {
        "price": {
            "type": ["integer", "null"],
            "description": "Price in INR only. Null if no INR price found in the text."
        },
        "availability": {
            "type": "string",
            "enum": ["in_stock", "out_of_stock", "unknown"],
            "description": "Stock status as stated in the text. Use 'unknown' if not mentioned."
        }
    },
    "required": ["price", "availability"]
}

call_f = {
    "title": "Fit-Evaluation",
    "type": "object",
    "properties": {
        "fit_score": {
            "type": "integer",
            "description": "Score from 1-10 rating how well this product's known specs match the required non-negotiable and negotiable specs. 10 = perfectly matches or exceeds all non-negotiable specs. Lower scores reflect missing or inferior specs."
        },
        "reasoning": {
            "type": "string",
            "description": "Brief explanation of the score - which specs matched well, which fell short or are unconfirmed, and how negotiable specs factored in."
        },
        "missing_or_weak_specs": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of specific non-negotiable specs that are missing entirely, or present but do not meet the requirement (e.g. 'CPU: found AMD Ryzen 5 7520U, weaker than required Ryzen 5 5600H')."
        }
    },
    "required": ["fit_score", "reasoning", "missing_or_weak_specs"]
}

str_model_call_a = llm_small.with_structured_output(call_a)
str_model_call_b = llm_medium.with_structured_output(call_b)
str_model_call_c = llm_large.with_structured_output(call_c)
str_model_call_d = llm_medium.with_structured_output(call_d)
str_model_call_e = llm_small.with_structured_output(call_e)
str_model_call_f = llm_medium.with_structured_output(call_f)

prompt1 = PromptTemplate(
    template="""Analyze the following user query -> {query},
    Do not respond to the user directly. Determine whether usecase and budget are both specified,
    if not specified ask user to specify, "asking a question" means filling in the question field, 
    not addressing the user directly.""",
    input_variables=['query']
)

prompt2 = PromptTemplate(
    template="""Extract technical specs/requirements for the product -> {category}, user's budget is -> {budget} and usecase is -> {usecase}

After extracting specs, categorise specs into negotiable and non-negotiable specs,
non-negotiable specs based on specs which are highly important and can't be neglected for given usecase, and
negotiable based on specs that are less important, which can be ignored if there are budget constraints.

IMPORTANT budget-realism constraint: if a budget is provided (not null), every non-negotiable spec you choose must be realistic and commonly available at that price point in the Indian market for this product category. Do not pick a spec tier that structurally forces the price far above the budget — for example, for a plain productivity/coding laptop with a budget under ₹60,000, do not require an H-series or HX-series processor (these are gaming/performance chips typically bundled with a dedicated GPU, pushing price well beyond that range) — prefer a U-series or equivalent power-efficient processor instead, since dedicated graphics are not needed for the stated usecase. Only require higher-tier, costlier specs as non-negotiable if the usecase genuinely cannot function without them (e.g. GPU is genuinely non-negotiable for gaming or ML, but not for general coding/productivity).

Respond only by populating the required schema fields. Do not write a conversational reply, markdown table, or explanation text,
output must go through the structured tool call only. You must respond only via the structured tool call, never as freeform conversational text.""",
    input_variables=['category', 'budget', 'usecase']
)

prompt3 = PromptTemplate(
    template="""You are given raw web search results for a product search, and the required specs.
    Required non-negotiable specs: {required_specs}
    Raw search results: {raw_results}
    Task:
    1. Identify distinct, specific products mentioned (ignore results that are generic articles, discussion forums, Q&A sites, listing pages with no single identifiable product, or reviews with no product data).
    2. For each distinct product, extract whatever specs are actually visible in the result.
    3. Mark specs_found as true only if a value is present for every required non-negotiable spec key for this product; this only means data was found, not that the values meet the requirement.
    4. Do not invent or assume specs that are not present in the text.
    5. Only extract products that are literally named in the provided raw_results text, and must not supplement with outside knowledge at all.
    6. Extract at most 4 candidates total. If more than 4 distinct products are found, choose the 4 whose visible specs most closely match the required non-negotiable specs.
    7. Ignore products that are clearly a different category than requested (e.g. a desktop PC when a laptop was requested), even if some specs overlap.
    8. Do not include a candidate if you cannot extract at least 2 concrete specs for it from the text. A product name alone, with no specs, is not a valid candidate — skip it entirely rather than including it with an empty known_specs.""",
    input_variables=['required_specs', 'raw_results']
)

prompt4 = PromptTemplate(
    template="""You are given a product name, the specs already known about it, and new raw search text about the same product.

    Product: {product_name}
    Already known specs: {known_specs}
    Required specs (for reference): {required_specs}
    Raw follow-up search text: {follow_up_text}

    Task:
    1. Read the raw follow-up text and identify any spec values for this exact product that are NOT already present in "Already known specs".
    2. Only extract specs that are literally stated in the follow-up text — do not invent or assume values.
    3. Only include specs relevant to the required_specs list; ignore irrelevant details (price, reviews, accessories, etc.) unless they match a required spec.
    4. Respond only through the structured tool call. Do not write conversational text.""",
    input_variables=['product_name', 'known_specs',
                     'required_specs', 'follow_up_text']
)

prompt5 = PromptTemplate(
    template="""You are given short text snippets extracted from a specific product's page. Each snippet is centered around a ₹ price mention, but not every ₹ amount is the actual product price — some may be for accessories, insurance add-ons, EMI breakdowns, or the original crossed-out M.R.P.

Product: {product_name}
Page content snippets: {page_content}

Task:
1. Identify the CURRENT SELLING PRICE of the product itself — this is the actual price a buyer pays right now.
2. Do NOT select: M.R.P./original/strikethrough prices (these are inflated reference prices, not the real price), EMI-per-month amounts, prices for accessories/insurance/add-on services, or prices for any other product mentioned in the text.
3. If multiple snippets show the same selling price repeated, that repetition is a strong signal it IS the correct price — prefer prices that appear more than once over one-off amounts.
4. Extract the price as a plain integer in INR only, with no currency symbol, commas, or decimals. If no clear current selling price can be identified, return null. Do not guess or average multiple different prices.
5. Determine availability/stock status if mentioned (e.g. "in stock", "only X left", "out of stock", "currently unavailable"); use "unknown" if not stated anywhere in the snippets.
6. Do not invent a price or availability status. Only extract what is literally present in the text.
7. Respond only through the structured tool call, never as conversational text or explanation.""",
    input_variables=['product_name', 'page_content']
)

prompt6 = PromptTemplate(
    template="""Evaluate how well this product fits the buyer's requirements.

Non-negotiable specs required: {non_negotiable_specs}
Negotiable specs (nice-to-have): {negotiable_specs}
Specs actually found for this product: {known_specs}

Task:
1. Compare each required non-negotiable spec against what was found. A spec only satisfies the requirement if it meets or exceeds what was asked - do not assume a similar-sounding spec is equivalent without judging it (e.g. AMD Ryzen 5 7520U is a weaker/newer-gen chip than Ryzen 5 5600H, not an automatic match).
2. If a required spec is entirely missing from known_specs, treat it as unmet, not neutral.
3. Give a fit_score from 1-10 based on how completely and well the non-negotiable specs are satisfied. Negotiable specs can add small bonus points but should not compensate for failing a non-negotiable spec.
4. List every non-negotiable spec that is missing or falls short in missing_or_weak_specs, with a short reason for each.
5. Respond only through the structured tool call, never as conversational text.""",
    input_variables=['non_negotiable_specs', 'negotiable_specs', 'known_specs']
)

prompt7 = PromptTemplate(
    template="""Generate a structured product recommendation report in Markdown, for a user shopping for a {category} in India.

Use case: {usecase}
Budget: {budget}
Non-negotiable requirements: {non_negotiable_specs}
Negotiable preferences: {negotiable_specs}

Candidates found (already sorted, best fit first):
{candidates}

Is this a degraded result (fewer than 2 candidates fully met all requirements within budget)? {is_degraded}
Realistic budget suggestion based on actual prices found (if degraded): {realistic_budget}

Write the report with these sections, in Markdown:
1. **Requirements Summary** - brief restatement of what was searched for.
2. **Top Recommendations** - list each candidate with product name, price, fit score, and a one-line reason.
3. **Comparison Table** - a Markdown table with columns: Product, Price, Fit Score, Key Specs, Within Budget.
4. **Fit Analysis** - for each candidate, note missing or weak specs from the data given.
5. **Final Recommendation** - pick the single best option and explain why, in plain language.
6. If is_degraded is true: add a **Budget Gap** section - explain honestly that no candidate fully met all requirements within budget, state the gap using the realistic_budget figure, and suggest the user either raise their budget close to that figure or relax a specific non-negotiable spec.
7. Highlight each product's source URL as a Markdown link.

Only use the data provided above - do not invent specs, prices, or products not listed in the candidates

If a spec is not present in known_specs for a candidate, you must state it is unknown/not listed — 
never suggest, estimate, or imply what the value probably is, even based on the brand or product line's typical specs""",
    input_variables=['category', 'usecase', 'budget', 'non_negotiable_specs',
                     'negotiable_specs', 'candidates', 'is_degraded', 'realistic_budget']
)

report_chain = prompt7 | llm_medium | parser

call_a_chain = prompt1 | str_model_call_a

branch_chain = RunnableBranch(
    (lambda x: x['status'] == "unclear",
     RunnableLambda(lambda x: x['question'])),
    (lambda x: x['status'] == "clear", prompt2 | str_model_call_b),
    RunnableLambda(lambda x: {"error": "Could not determine status", "raw": x})
)

final_chain = call_a_chain | branch_chain

# Case 1: everything present
# result1 = final_chain.invoke(
#     {"query": "I want to buy a laptop for coding, budget around 50000 INR"})
# print(result1)

# # Case 2: use case present, budget missing
# result2 = final_chain.invoke(
#     {"query": "I want to buy a laptop for Machine Learning work"})
# print(result2)

# # Case 3: budget present, use case missing
# result3 = final_chain.invoke(
#     {"query": "I want to buy a laptop, budget around 80000 INR"})
# print(result3)
