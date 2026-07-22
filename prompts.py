from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda

load_dotenv()
# Call A (status/question), Call E (price/availability)
llm_small = ChatGroq(model="openai/gpt-oss-120b", max_tokens=500)
# Call B, Call D
llm_medium = ChatGroq(model="openai/gpt-oss-120b",
                      max_tokens=1200)
# Call C (up to 4 full candidates)
llm_large = ChatGroq(model="openai/gpt-oss-120b", max_tokens=2000)

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

str_model_call_a = llm_small.with_structured_output(call_a)
str_model_call_b = llm_medium.with_structured_output(call_b)
str_model_call_c = llm_large.with_structured_output(call_c)
str_model_call_d = llm_medium.with_structured_output(call_d)
str_model_call_e = llm_small.with_structured_output(call_e)

prompt1 = PromptTemplate(
    template="""Analyze the following user query -> {query},
    Do not respond to the user directly. Determine whether usecase and budget are both specified,
    if not specified ask user to specify, "asking a question" means filling in the question field, 
    not addressing the user directly.""",
    input_variables=['query']
)

prompt2 = PromptTemplate(
    template="""Extract technical specs/requirements for the product -> {category}, user's budget is -> {budget} and usecase is -> {usecase} \n 
    After extracting specs, categorise specs into negotiable and non-negotiable specs, 
    non-negotiable specs based on specs which are highly important and can't be neglected for given usecase, and 
    negotiable based on specs that are less important, which can be ignored if there budget constraints
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
#     {"query": "I want to buy a laptop for Machine Learning work, budget around 80000 INR"})
# print(result1)

# # Case 2: use case present, budget missing
# result2 = final_chain.invoke(
#     {"query": "I want to buy a laptop for Machine Learning work"})
# print(result2)

# # Case 3: budget present, use case missing
# result3 = final_chain.invoke(
#     {"query": "I want to buy a laptop, budget around 80000 INR"})
# print(result3)
