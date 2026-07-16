from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
llm = ChatGroq(model="openai/gpt-oss-120b")
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Shorten this given specs into 2-4 word keyword phrases {specs}, also output the result as a single space-separated line, not a list.",
    input_variables=['specs']
)


def build_query(call_b_result):
    category = call_b_result["category"]
    specs = call_b_result["non_negotiable_specs"] | call_b_result["negotiable_specs"]
    chain = prompt1 | llm | parser
    result = chain.invoke({"specs": specs})
    restriction = 'India price INR only, Flipkart Amazon.in or any Indian store'

    query = category + " " + result + " " + restriction

    return query


def get_official_specs(product_name):
    """Narrow, single-product search restricted to reliable spec sources."""
    spec_search_tool = TavilySearch(
        max_results=3,
        include_domains=["91mobiles.com", "smartprix.com",
                         "gsmarena.com", "notebookcheck.net"]
        # swap/expand this list depending on product category later
    )
    query = f"{product_name} full specifications"
    return spec_search_tool.invoke({"query": query})


call_b_output = {'budget': 80000, 'category': 'laptop', 'negotiable_specs': {'Battery': '6+ hours', 'Display': '15.6-inch Full HD (1920x1080), optional 1440p', 'Keyboard': 'Backlit keyboard', 'Ports': 'USB-C, HDMI, USB-A, etc.', 'Weight': 'Under 2.5 kg'}, 'non_negotiable_specs': {
    'CPU': 'Intel Core i7-13th gen or AMD Ryzen 7 7000 series', 'GPU': 'NVIDIA RTX 3060 (6GB VRAM) or better', 'OS': 'Windows 11 Pro', 'RAM': '16GB DDR4/DDR5', 'Storage': '512GB NVMe SSD'}, 'usecase': 'Machine Learning work'}

query = build_query(call_b_result=call_b_output)
print("QUERY : ", query)

search_tool = TavilySearch(max_results=3)

result = search_tool.invoke({"query": query})

print(result)
