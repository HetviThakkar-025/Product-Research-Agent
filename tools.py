from dotenv import load_dotenv
import time
from groq import RateLimitError
from langchain_tavily import TavilySearch, TavilyExtract
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
llm = ChatGroq(model="openai/gpt-oss-120b")
parser = StrOutputParser()

RETAIL_DOMAINS = ["flipkart.com", "amazon.in", "croma.com",
                  "reliancedigital.in", "vijaysales.com", "tatacliq.com"]
SPEC_DOMAINS = ["91mobiles.com", "smartprix.com",
                "gsmarena.com", "notebookcheck.net"]

prompt1 = PromptTemplate(
    template="Shorten this given specs into 2-4 word keyword phrases {specs}, also output the result as a single space-separated line, not a list.",
    input_variables=['specs']
)


def invoke_with_retry(chain, inputs, max_retries=3):
    for attempt in range(max_retries):
        try:
            return chain.invoke(inputs)
        except RateLimitError as e:
            wait_time = min(2 ** attempt, 10)  # cap at 10s
            print(f"Rate limit hit, waiting {wait_time}s before retry...")
            time.sleep(wait_time)
    raise Exception("Max retries exceeded for rate limit")


def build_query(call_b_result):
    category = call_b_result["category"]
    specs = call_b_result["non_negotiable_specs"]
    chain = prompt1 | llm | parser
    result = invoke_with_retry(chain, {"specs": specs})
    restriction = 'India price INR only, Indian markets only'

    query = category + " " + result + " " + restriction

    return query


def filter_by_domain(results, allowed_domains):
    verified_results = []
    for r in results['results']:
        url = r['url']
        is_allowed = any(domain in url for domain in allowed_domains)
        if is_allowed:
            verified_results.append(r)
        else:
            print(f"Dropped out-of-domain url: {url}")
    return verified_results


def trim_results(results):
    """Strip unused fields and tracking-parameter bloat before sending to an LLM."""
    trimmed = []
    for r in results['results']:
        clean_url = r['url'].split('?')[0]  # drop query-string tracking params
        trimmed.append({
            'url': clean_url,
            'title': r.get('title', ''),
            'content': r.get('content', '')
        })
    return {'results': trimmed}


def get_official_specs(product_name):
    """Narrow, single-product search restricted to reliable spec sources."""
    spec_search_tool = TavilySearch(
        max_results=3, include_domains=SPEC_DOMAINS)
    query = f"{product_name} full specifications"
    raw = spec_search_tool.invoke({"query": query})
    filtered = filter_by_domain(raw, SPEC_DOMAINS)
    return trim_results({'results': filtered})


def filter_hallucinated_candidates(candidates, raw_results):
    real_urls = [r['url'] for r in raw_results['results']]

    verified_candidates = []
    for candidate in candidates:
        source = candidate.get('source_url', '')
        is_real = any(
            source in real_url or real_url in source
            for real_url in real_urls
        )
        if is_real:
            verified_candidates.append(candidate)
        else:
            print(
                f"Dropped hallucinated candidate: {candidate['product_name']} (fake source: {source})")

    return verified_candidates


def extract_price(source_url):
    """Try to get full page content from the candidate's own source URL."""
    extract_tool = TavilyExtract(extract_depth="advanced")
    result = extract_tool.invoke({"urls": [source_url]})
    return result


def search_price_fallback(product_name, source_url):
    """Fallback: narrow search restricted to the same domain as source_url."""
    domain = source_url.split(
        '/')[2].replace('www.', '') 
    fallback_tool = TavilySearch(max_results=2, include_domains=[domain])
    query = f"{product_name} price"
    return fallback_tool.invoke({"query": query})


search_tool = TavilySearch(max_results=4, include_domains=RETAIL_DOMAINS)
