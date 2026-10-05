import re
from urllib.parse import urlparse
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

Use at most 4 specs in total: keep only the 3-4 that narrow the search the most.

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

    # no "India/INR only" phrase: include_domains already restricts to Indian retailers,
    # and the extra words pulled results toward listing/search pages
    query = category + " " + result.strip()

    return query


def drop_spec_from_query(query, call_b_result):
    """
    Search attempt 2's query: the iteration's rewrite with one core (non-negotiable) spec's words removed,
    so a retry never repeats the same text. Specs are tried last to first; if none of their words are in
    the query, the last word is dropped instead. Returns (query, dropped spec key or None).
    """
    words = query.split()
    keep = len(call_b_result["category"].split())  # the category prefix always stays
    head, tail = words[:keep], words[keep:]
    for key, value in reversed(list(call_b_result["non_negotiable_specs"].items())):
        spec_tokens = _spec_tokens(f"{key} {value}")
        remaining = [w for w in tail if not (_spec_tokens(w) and _spec_tokens(w) <= spec_tokens)]
        if remaining and len(remaining) < len(tail):
            return " ".join(head + remaining), key
    return " ".join(head + tail[:-1]), None


def has_path(url):
    """False for a bare domain or path-less URL such as https://www.vijaysales.com/ (scheme optional)."""
    parsed = urlparse(url if '://' in url else f"https://{url}")
    return bool(parsed.path.strip('/'))


def is_product_page_url(url):
    if not has_path(url):
        return False

    listing_patterns = ['/s?', '/s/', 'search', '/l/',
                        '/b', '/b/', '/c/', 'clp', 'collection']
    product_patterns = ['/dp/', '/p/itm', '/product/']

    # search/collection pages and blogs that the generic patterns below miss
    non_product_patterns = ['flipkart.com/q/', 'croma.com/unboxed/']

    url_lower = url.lower()

    if any(pattern in url_lower for pattern in non_product_patterns):
        return False
    if 'flipkart.com' in url_lower and url_lower.split('?')[0].rstrip('/').endswith('/pr'):
        return False
    if any(pattern in url_lower for pattern in product_patterns):
        return True
    if any(pattern in url_lower for pattern in listing_patterns):
        return False
    if url_lower.rstrip('/').endswith('/s'):
        return False

    return True


def normalize_product_url(url):
    """Key for "same product page": Amazon ASIN, Flipkart itm id (drops /hi/, slug and query), else host + path without query."""
    url = url.split('?')[0].split('#')[0].lower().rstrip('/')

    asin = re.search(r'amazon\.in/(?:.*/)?(?:dp|gp/product)/([a-z0-9]{10})', url)
    if asin:
        return f"amazon.in/dp/{asin.group(1)}"

    itm = re.search(r'flipkart\.com/.*/p/(itm[a-z0-9]+)', url)
    if itm:
        return f"flipkart.com/p/{itm.group(1)}"

    return re.sub(r'^https?://(www\.)?', '', url)


# chip families look like model numbers (dimensity9200, snapdragon8gen3...) but are shared across many products
NON_MODEL_PREFIXES = ('dimensity', 'snapdragon', 'helio', 'exynos', 'tensor', 'bionic',
                      'ryzen', 'core', 'ultra', 'rtx', 'gtx', 'radeon', 'windows')


def model_numbers(product_name):
    """
    Model-number-like tokens in a product name, e.g. RT38HG5A42S8HL, 82RK0085IN, fq5112tu.
    Requires 8+ chars with 2+ letters and 3+ digits, so CPU/GPU/RAM tokens (i5-1235U, RTX4050, 16GB) don't count.
    """
    return {token for token in re.findall(r'[a-z0-9]+', product_name.lower()) if _is_model_token(token)}


def _is_model_token(token):
    letters = sum(ch.isalpha() for ch in token)
    digits = sum(ch.isdigit() for ch in token)
    return len(token) >= 8 and letters >= 2 and digits >= 3 and not token.startswith(NON_MODEL_PREFIXES)


def _spec_tokens(text):
    """Lowercase word tokens with "8 GB" -> "8gb" and "i5 1235U" -> "i5-1235u", so names and page text compare alike."""
    text = re.sub(r'(\d)\s+(gb|tb)\b', r'\1\2', text.lower())
    text = re.sub(r'\b(i[3579])[\s-]+(\d{4,5}[a-z]{0,2})\b', r'\1-\2', text)
    return set(re.findall(r'[a-z0-9]+(?:[-.][a-z0-9]+)*', text))


# CPU models (i5-1235u, 7520u) and memory/storage sizes (8gb, 512gb, 1tb)
DISTINCTIVE_TOKEN_RE = re.compile(r'i[3579]-\d{4,5}[a-z]{0,2}|\d{4,5}[a-z]{1,2}|\d+(?:gb|tb)')


def product_match(candidate, url, title, content):
    """
    How a price source is tied to this candidate, else None. In order: one of the candidate's model numbers
    (its name, dropped duplicates, own result title) in the source title/content; the same product page URL;
    and, only when no model number is known, the brand plus every distinctive name token (CPU model,
    RAM/storage sizes) in the source title/content.
    """
    source_text = f"{title} {content}"
    models = model_numbers(candidate['product_name']) | set(candidate.get('model_numbers', []))
    if models & model_numbers(source_text):
        return 'model number'

    if normalize_product_url(url) == normalize_product_url(candidate.get('source_url', '')):
        return 'same url'
    if models:
        # a known model number must match; brand + specs would accept sibling models (HP 15 fd0070TU vs 15s fy5007TU)
        return None

    name_tokens = _spec_tokens(candidate['product_name'])
    brand = re.findall(r'[a-z0-9]+', candidate['product_name'].lower())[:1]
    distinctive = {t for t in name_tokens if DISTINCTIVE_TOKEN_RE.fullmatch(t)}
    source_tokens = _spec_tokens(source_text)
    if brand and distinctive and brand[0] in source_tokens and distinctive <= source_tokens:
        return 'brand and specs'
    return None


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


def _squash(text):
    return re.sub(r'\s+', ' ', text).strip().lower()


def drop_repeated_title(results):
    """Removes the title from the start of each result's content (Call C sees both), keeping the rest of the content."""
    trimmed = []
    for r in results['results']:
        # product part of the title, without a trailing " : Amazon.in: Electronics" / " - Buy ..." style suffix
        name = _squash(re.split(r'\s+[:|-]\s+(?:amazon|flipkart|croma|reliance|vijay|tata|buy)', r.get('title', ''),
                                flags=re.IGNORECASE)[0])
        content = r.get('content', '')
        body = _squash(content.lstrip('#*> \n'))
        if name and body.startswith(name):
            content = body[len(name):].lstrip(' ,.:;|-')
        elif name and body and name.startswith(body):
            content = ''
        trimmed.append({**r, 'content': content})
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


# window searched around a ₹ amount for the product it belongs to (names come before prices on retail pages)
PRICE_CONTEXT_BEFORE = 200
PRICE_CONTEXT_AFTER = 50
NAME_PREFIX_CHARS = 40


def own_product_ids(candidate):
    """Model numbers known to be the candidate's (name, dropped duplicates, own result title) plus its page's ASIN / itm id."""
    ids = model_numbers(candidate['product_name']) | set(candidate.get('model_numbers', []))
    page_id = re.search(r'/(?:dp|p)/([a-z0-9]+)$', normalize_product_url(candidate.get('source_url', '')))
    if page_id:
        ids.add(page_id.group(1))
    return ids


def attribute_prices(candidate, text):
    """
    Keeps only the ₹ amounts in text that belong to this candidate, so prices of other products on the same
    page (carousels, "similar items") are dropped. For each amount, the nearest model number in the window
    (the closest one before it, else the first after) decides: the candidate's own -> kept, another -> rejected.
    With no model number in the window, the amount is kept only if the start of the candidate's name is in it.
    Returns (kept ₹ matches, rejected [(amount, reason)]).
    """
    own = own_product_ids(candidate)
    name = _squash(candidate['product_name'])[:NAME_PREFIX_CHARS]
    kept, rejected = [], []
    for m in re.finditer(PRICE_RE, text or ''):
        amounts = rupee_amounts(m.group())
        if not amounts:
            continue
        start = max(0, m.start() - PRICE_CONTEXT_BEFORE)
        window = text[start:m.end() + PRICE_CONTEXT_AFTER]
        position = m.start() - start
        mentions = [(w.start(), w.group()) for w in re.finditer(r'[a-z0-9]+', window.lower()) if _is_model_token(w.group())]
        if mentions:
            before = [token for pos, token in mentions if pos < position]
            nearest = before[-1] if before else mentions[0][1]
            if nearest in own:
                kept.append(m)
            else:
                rejected.append((amounts.pop(), f"other model nearby ({nearest})"))
        elif name and name in _squash(window):
            kept.append(m)
        else:
            rejected.append((amounts.pop(), "product not named nearby"))
    return kept, rejected


def extract_price_snippets(raw_content, window=80, max_snippets=5, matches=None):
    """
    Pull small text windows around every ₹ price mention (or only around the given ₹ matches),
    instead of sending the entire page to the LLM.
    """
    if not raw_content:
        return ""

    if matches is None:
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

        # a bare domain is a substring of every URL on that site, so it can never count as a match
        is_real = has_path(source) and any(
            source in real_url or real_url in source for real_url in real_urls if has_path(real_url))
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
    """Full text of the candidate's own page (not just ₹ snippets), so prices can be checked against the product name."""
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

    return result['results'][0].get('raw_content', '') or ''


def search_price_fallback(product_name):
    """
    Fallback: price search across all retail domains (not just the candidate's own site).
    Returns [{'url', 'title', 'content'}] for product pages only, so listing-page prices of other products can't leak in.
    """
    fallback_tool = TavilySearch(max_results=5, include_domains=RETAIL_DOMAINS)
    query = f"{product_name} price"
    raw = fallback_tool.invoke({"query": query})

    results = []
    for r in raw.get('results', []):
        url = r['url'].split('?')[0]
        if not any(domain in url for domain in RETAIL_DOMAINS) or not is_product_page_url(url):
            continue
        results.append({'url': url, 'title': r.get('title') or '', 'content': r.get('content') or ''})
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


search_tool = TavilySearch(max_results=7, include_domains=RETAIL_DOMAINS)
