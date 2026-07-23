import re
from dotenv import load_dotenv
import time
from groq import RateLimitError, APIStatusError
from langchain_tavily import TavilySearch, TavilyExtract
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
llm = ChatGroq(model="openai/gpt-oss-120b", max_tokens=600)
parser = StrOutputParser()

RETAIL_DOMAINS = ["flipkart.com", "amazon.in", "croma.com",
                  "reliancedigital.in", "vijaysales.com", "tatacliq.com"]
SPEC_DOMAINS = ["91mobiles.com", "smartprix.com",
                "gsmarena.com", "notebookcheck.net"]

prompt1 = PromptTemplate(
    template="""Given these required specs: {specs}

Shorten them into 2-4 word keyword phrases suitable for a product search query.

Prioritize only the specs that meaningfully narrow a product search: CPU/processor, GPU/graphics, RAM, storage, display size/resolution, and operating system.

Skip specs that are minor or unusual for search purposes (e.g. keyboard type, port types, weight, battery life, build material) — even if they are technically required, they add noise to a search query rather than helping find matching products.

Output the result as a single space-separated line, not a list.""",
    input_variables=['specs']
)


def invoke_with_retry(chain, inputs, max_retries=3):
    for attempt in range(max_retries):
        try:
            return chain.invoke(inputs)
        except (RateLimitError, APIStatusError) as e:
            wait_time = min(2 ** attempt, 10)
            print(f"Rate/size limit hit ({type(e).__name__}): {e}")
            print(f"Waiting {wait_time}s before retry...")
            time.sleep(wait_time)
    raise Exception("Max retries exceeded for rate/size limit")


def build_query(call_b_result, include_negotiable=True):
    category = call_b_result["category"]
    # copy, don't mutate original
    specs = dict(call_b_result["non_negotiable_specs"])

    if include_negotiable and call_b_result.get("negotiable_specs"):
        # only add a couple, so negotiable specs nudge the search without dominating it
        negotiable_items = list(call_b_result["negotiable_specs"].items())[:2]
        for key, value in negotiable_items:
            specs[key] = value

    chain = prompt1 | llm | parser
    result = invoke_with_retry(chain, {"specs": specs})
    restriction = 'India price INR only, Indian markets only'

    query = category + " " + result + " " + restriction

    return query


def is_product_page_url(url):
    listing_patterns = ['/s?', '/s/', 'search', '/l/',
                        '/b', '/b/', '/c/', 'clp', 'collection']
    product_patterns = ['/dp/', '/p/itm', '/product/']

    url_lower = url.lower()

    if any(pattern in url_lower for pattern in product_patterns):
        return True
    if any(pattern in url_lower for pattern in listing_patterns):
        return False
    if url_lower.rstrip('/').endswith('/s'):
        return False

    return True


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


def trim_results(results, max_content_length=200):
    trimmed = []
    for r in results['results']:
        clean_url = r['url'].split('?')[0]
        trimmed.append({
            'url': clean_url,
            'title': r.get('title', ''),
            'content': (r.get('content') or '')[:max_content_length]
        })
    return {'results': trimmed}


def cap_results(results, max_for_llm=5):
    sorted_results = sorted(
        results['results'], key=lambda r: r.get('score', 0), reverse=True)
    return {'results': sorted_results[:max_for_llm]}


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

        if not source:
            print(
                f"Dropped candidate with missing source_url: {candidate.get('product_name')}")
            continue

        is_real = any(
            source in real_url or real_url in source for real_url in real_urls)
        if not is_real:
            print(
                f"Dropped hallucinated candidate: {candidate['product_name']} (fake source: {source})")
            continue

        if not is_product_page_url(source):
            print(
                f"Dropped listing-page candidate: {candidate['product_name']} (not a product page: {source})")
            continue

        verified_candidates.append(candidate)

    return verified_candidates


def extract_price(source_url):
    if not is_product_page_url(source_url):
        print(
            f"Skipping price extraction, looks like a listing page: {source_url}")
        return ""

    extract_tool = TavilyExtract(extract_depth="advanced")
    result = extract_tool.invoke({"urls": [source_url]})

    if not isinstance(result, dict) or not result.get('results'):
        print(f"Unexpected extract response shape for {source_url}")
        return ""

    raw_content = result['results'][0].get('raw_content', '') or ''
    return extract_price_snippets(raw_content)


def search_price_fallback(product_name, source_url):
    """Fallback: narrow search restricted to the same domain as source_url."""
    domain = source_url.split('/')[2].replace('www.', '')
    fallback_tool = TavilySearch(max_results=2, include_domains=[domain])
    query = f"{product_name} price"
    raw = fallback_tool.invoke({"query": query})

    combined_content = " ".join(r.get('content', '')
                                for r in raw.get('results', []))
    return extract_price_snippets(combined_content)


def select_report_candidates(all_candidates, top_n=3):
    def sort_key(c):
        # within_budget True sorts first (False=0 lower priority than True=1... need True first)
        budget_priority = 1 if c.get('within_budget') is True else 0
        return (budget_priority, c.get('fit_score', 0))

    sorted_candidates = sorted(all_candidates, key=sort_key, reverse=True)
    return sorted_candidates[:top_n]


def suggest_realistic_budget(candidates):
    prices = [c['price'] for c in candidates if c.get('price') is not None]
    return min(prices) if prices else None


search_tool = TavilySearch(max_results=4, include_domains=RETAIL_DOMAINS)
