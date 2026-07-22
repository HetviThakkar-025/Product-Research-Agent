from dotenv import load_dotenv
from langchain_tavily import TavilyExtract
import re

load_dotenv()


def extract_price_snippets(raw_content, window=80, max_snippets=5):
    """
    Pull small text windows around every ₹ price mention,
    instead of sending the entire page to the LLM.
    """
    if not raw_content:
        return ""

    matches = list(re.finditer(r'₹[\d,]+(?:\.\d+)?', raw_content))
    if not matches:
        return ""

    snippets = []
    for m in matches[:max_snippets]:
        start = max(0, m.start() - window)
        end = min(len(raw_content), m.end() + window)
        snippets.append(raw_content[start:end].strip())

    return "\n---\n".join(snippets)


extract_tool = TavilyExtract(extract_depth="advanced")

# use a real product URL from one of your actual past runs
result = extract_tool.invoke(
    {"urls": ["https://www.amazon.in/hp-laptop-i5-11th-generation/s"]})

if not result.get('results'):
    print('null')
else:
    raw_content = result['results'][0].get('raw_content', '') or ''
    result = extract_price_snippets(raw_content)

print(result)
