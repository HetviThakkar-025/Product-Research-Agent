from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda

load_dotenv()
llm = ChatGroq(model="openai/gpt-oss-120b")

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
                    "specs_complete": {
                        "type": "boolean",
                        "description": "True only if all non-negotiable specs from requirements are confirmed for this product"
                    }
                },
                "required": ["product_name", "known_specs", "specs_complete"]
            }
        }
    },
    "required": ["candidates"]
}

str_model_call_a = llm.with_structured_output(call_a)
str_model_call_b = llm.with_structured_output(call_b)
str_model_call_c = llm.with_structured_output(call_c)

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
    negotiable based on specs that are less important, which can be ignored if there budget constraints""",
    input_variables=['category', 'budget', 'usecase']
)

prompt3 = PromptTemplate(
    template="""You are given raw web search results for a product search, and the required specs.
    Required non-negotiable specs: {required_specs}
    Raw search results: {raw_results} 
    Task:
    1. Identify distinct, specific products mentioned (ignore results that are generic articles, listing pages with no single identifiable product, or reviews with no product data).
    2. For each distinct product, extract whatever specs are actually visible in the result.
    3. Mark specs_complete as true only if every required non-negotiable spec is confirmed for that product; otherwise false.
    4. Do not invent or assume specs that are not present in the text.""",
    input_variables=['required_specs', 'raw_results']
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
result1 = final_chain.invoke(
    {"query": "I want to buy a laptop for Machine Learning work, budget around 80000 INR"})
print(result1)

# # Case 2: use case present, budget missing
# result2 = final_chain.invoke(
#     {"query": "I want to buy a laptop for Machine Learning work"})
# print(result2)

# # Case 3: budget present, use case missing
# result3 = final_chain.invoke(
#     {"query": "I want to buy a laptop, budget around 80000 INR"})
# print(result3)
