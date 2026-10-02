import re
from dotenv import load_dotenv
import time
from groq import RateLimitError, APIStatusError
from langchain_tavily import TavilySearch, TavilyExtract
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from prompts import MODEL, REASONING_EFFORT

load_dotenv()
llm = ChatGroq(model=MODEL, max_tokens=600, reasoning_effort=REASONING_EFFORT, max_retries=0)
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


class DailyQuotaExceeded(Exception):
    """Raised when Groq's daily token quota (TPD) is hit — retrying won't help until reset."""
    pass


class LLMRetriesExhausted(Exception):
    """Raised when an LLM call still fails after all retries — carries the last real error text."""
    pass


TOOL_USE_FAILED_RETRIES = 2
RETRY_MARGIN_SECONDS = 1.0
MAX_RETRY_WAIT_SECONDS = 65


def parse_retry_after(error):
    """Seconds Groq asks us to wait, from 'try again in 1m2.5s' / '14.6s' / '450ms', else the retry-after header."""
    match = re.search(r'try again in ([\d.hms]+)', str(error))
    if match:
        units = {'h': 3600, 'm': 60, 's': 1, 'ms': 0.001}
        parts = re.findall(r'([\d.]+)(ms|h|m|s)', match.group(1))
        if parts:
            return sum(float(value) * units[unit] for value, unit in parts)

    response = getattr(error, 'response', None)
    header = response.headers.get('retry-after') if response is not None else None
    try:
        return float(header) if header else None
    except ValueError:
        return None


def invoke_with_retry(chain, inputs, max_retries=3):
    rate_limit_retries = 0
    tool_use_retries = 0
    other_retries = 0

    while True:
        try:
            return chain.invoke(inputs)
        except (RateLimitError, APIStatusError) as e:
            error_text = str(e)
            status_code = getattr(e, 'status_code', None)

            if "tokens per day" in error_text or "TPD" in error_text:
                # daily quota exhausted — waiting seconds won't help, fail fast
                print(f"Daily quota exceeded: {e}")
                raise DailyQuotaExceeded(
                    "Groq's daily free-tier token limit has been reached. Please try again after the daily reset."
                ) from e

            if isinstance(e, RateLimitError) or status_code == 429:
                rate_limit_retries += 1
                print(f"Rate limit hit (429): {e}")
                if rate_limit_retries > max_retries:
                    raise LLMRetriesExhausted(
                        f"Still rate limited after {max_retries} retries. Last error: {error_text}") from e
                retry_after = parse_retry_after(e)
                wait_time = min(retry_after + RETRY_MARGIN_SECONDS, MAX_RETRY_WAIT_SECONDS) \
                    if retry_after is not None else min(2 ** rate_limit_retries, 10)
                print(f"Waiting {wait_time:.1f}s before retry {rate_limit_retries}/{max_retries}...")
                time.sleep(wait_time)

            elif status_code == 400 and "tool_use_failed" in error_text:
                # bad/truncated tool call from the model — waiting won't help, just regenerate
                tool_use_retries += 1
                print(f"Tool call failed (400 tool_use_failed): {e}")
                if tool_use_retries > TOOL_USE_FAILED_RETRIES:
                    raise LLMRetriesExhausted(
                        f"Model kept failing the structured tool call after {TOOL_USE_FAILED_RETRIES} retries. Last error: {error_text}") from e
                print(f"Retrying immediately ({tool_use_retries}/{TOOL_USE_FAILED_RETRIES})...")

            else:
                # e.g. 413 request too large, 5xx — keep the old short backoff
                other_retries += 1
                print(f"Groq API error ({type(e).__name__}, status {status_code}): {e}")
                if other_retries > max_retries:
                    raise LLMRetriesExhausted(
                        f"Groq API error persisted after {max_retries} retries. Last error: {error_text}") from e
                wait_time = min(2 ** (other_retries - 1), 10)
                print(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)


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


PRICE_RE = r'₹[\d,]+(?:\.\d+)?'


def rupee_amounts(text):
    """Every ₹ amount literally present in text, as integers (₹59,499.00 -> 59499)."""
    amounts = set()
    for match in re.findall(PRICE_RE, text or ''):
        try:
            amounts.add(int(float(match.lstrip('₹').replace(',', ''))))
        except ValueError:
            pass
    return amounts


def find_search_snippet(source_url, snippets):
    """Full retail-search content for a candidate's source_url (same loose match as the hallucination check)."""
    if source_url in snippets:
        return snippets[source_url]
    for url, content in snippets.items():
        if source_url in url or url in source_url:
            return content
    return ''


def extract_price_snippets(raw_content, window=80, max_snippets=5):
    """
    Pull small text windows around every ₹ price mention,
    instead of sending the entire page to the LLM.
    """
    if not raw_content:
        return ""

    matches = list(re.finditer(PRICE_RE, raw_content))
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

    # langchain_tavily returns {"error": e} instead of raising when the API call itself fails
    if isinstance(result, dict) and 'error' in result:
        error = result['error']
        print(f"TavilyExtract error for {source_url}: {type(error).__name__}: {error}")
        return ""

    if not isinstance(result, dict) or not result.get('results'):
        print(f"Unexpected extract response shape for {source_url}: {str(result)[:300]}")
        return ""

    raw_content = result['results'][0].get('raw_content', '') or ''
    return extract_price_snippets(raw_content)


def search_price_fallback(product_name):
    """
    Fallback: price search across all retail domains (not just the candidate's own site).
    Returns [{'url', 'content'}] for product pages only, so listing-page prices of other products can't leak in.
    """
    fallback_tool = TavilySearch(max_results=5, include_domains=RETAIL_DOMAINS)
    query = f"{product_name} price"
    raw = fallback_tool.invoke({"query": query})

    results = []
    for r in raw.get('results', []):
        url = r['url'].split('?')[0]
        if not any(domain in url for domain in RETAIL_DOMAINS) or not is_product_page_url(url):
            continue
        results.append({'url': url, 'content': r.get('content') or ''})
    return results


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
