from dotenv import load_dotenv
from langchain_tavily import TavilyExtract

load_dotenv()

extract_tool = TavilyExtract(extract_depth="advanced")

# use a real product URL from one of your actual past runs
result = extract_tool.invoke({"urls": ["https://www.amazon.in/dp/B0BWS9YNCX"]})

print(result)
