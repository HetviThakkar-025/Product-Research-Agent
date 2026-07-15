from dotenv import load_dotenv
from langchain_tavily import TavilySearch

load_dotenv()

search_tool = TavilySearch(max_results=3, include_domains=["flipkart.com", "amazon.in"])

result = search_tool.invoke(
    {"query": "I want to buy laptop for Machine Learning work"})

print(result)
