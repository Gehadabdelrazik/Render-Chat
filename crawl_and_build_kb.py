"""
Multi-tournament KB builder.

Given a tournament's website, this crawls it (staying on the same domain),
finds relevant pages and linked PDFs, extracts their text, and uses Gemini
to chunk everything into our standard KB format — saved under a unique
slug so multiple tournaments' knowledge bases can coexist.

Usage:
    python crawl_and_build_kb.py --url https://klsquash.org/REDTONE/ --slug redtone-2026

Output:
    kb_chunks_<slug>.json   (review this before precomputing embeddings)
"""

import os
import sys
import json
import argparse
import time
import re

# Prevent crashes when printing unusual characters (e.g. from scraped web content)
# under Windows consoles that default to a limited encoding like cp1252.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
import io
from playwright.sync_api import sync_playwright
from google import genai

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-3.6-flash"

MAX_PAGES = 15       # cap how many HTML pages we crawl, to avoid runaway crawls
MAX_DEPTH = 2         # how many link-hops deep from the starting page
REQUEST_DELAY = 0.5   # seconds between requests, polite to the target server

# Content-Type substrings that count as "a downloadable document" when a URL
# has no recognizable file extension (e.g. /download.php?id=42) — used as a
# fallback check when the URL's spelling alone doesn't tell us what it is.
FILE_CONTENT_TYPES = ["application/pdf", "application/msword", "application/vnd.openxmlformats"]
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


# ── Crawling ──────────────────────────────────────────────────────
def same_domain(url, base_domain):
    return urlparse(url).netloc == base_domain


def login(page, login_url, username, password, username_selector, password_selector, submit_selector):
    """Logs into a site before crawling, so login-gated pages (like a player or
    coach dashboard) become visible. Credentials are never printed or stored —
    they're passed in as environment variables by whoever runs this script."""
    print(f"Logging in at {login_url}...")
    page.goto(login_url, timeout=20000, wait_until="networkidle")
    page.fill(username_selector, username)
    page.fill(password_selector, password)
    page.click(submit_selector)
    page.wait_for_load_state("networkidle", timeout=20000)
    print("  Login submitted. Continuing crawl with this authenticated session.")


def find_onclick_links(soup, base_url):
    """Finds elements using onclick="window.open('...')" (or similar) instead of
    a real href, and extracts the real URL directly from the attribute text —
    no click simulation needed, since the address is already written down."""
    found = []
    for el in soup.find_all(attrs={"onclick": True}):
        onclick_value = el.get("onclick", "")
        match = re.search(r"window\.open\(\s*['\"]([^'\"]+)['\"]", onclick_value)
        if match:
            found.append(urljoin(base_url, match.group(1)))
    return found


def check_content_type(url: str):
    """For a URL with no recognizable file extension, checks what the server
    actually serves there via a lightweight HEAD request (falling back to a
    GET if HEAD isn't supported). Returns True if it looks like a document."""
    try:
        response = requests.head(url, headers=HEADERS, allow_redirects=True, timeout=8)
        content_type = response.headers.get("content-type", "").lower()
        if not content_type or response.status_code >= 400:
            response = requests.get(url, headers=HEADERS, stream=True, timeout=8)
            content_type = response.headers.get("content-type", "").lower()
        return any(ct in content_type for ct in FILE_CONTENT_TYPES)
    except Exception:
        return False


def crawl_site(start_url, max_pages=MAX_PAGES, max_depth=MAX_DEPTH, login_params=None):
    """Returns (list of (url, html_text), list of pdf_urls) found on the site.
    Uses Playwright (a real headless browser) instead of plain requests, so that
    JavaScript-rendered navigation and content actually load before we read the page —
    many modern sites build their menus/links dynamically via JS, which a plain
    HTTP request can't see.

    If login_params is provided (a dict with login_url, username, password, and
    the three field selectors), logs in first so login-gated pages become visible
    to the rest of the crawl."""
    base_domain = urlparse(start_url).netloc
    visited = set()
    to_visit = [(start_url, 0)]
    pages = []
    pdf_urls = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent=HEADERS["User-Agent"])

        if login_params:
            try:
                login(page, **login_params)
            except Exception as e:
                print(f"  WARNING: Login failed ({e}). Continuing without authentication — login-gated pages will be skipped.")

        while to_visit and len(pages) < max_pages:
            url, depth = to_visit.pop(0)
            if url in visited or depth > max_depth:
                continue
            visited.add(url)

            if url.lower().endswith(".pdf"):
                pdf_urls.add(url)
                continue

            try:
                response = page.goto(url, timeout=20000, wait_until="networkidle")
                if response is None:
                    print(f"  WARNING: No response for {url}")
                    continue
                content_type = response.headers.get("content-type", "")
                if "application/pdf" in content_type:
                    pdf_urls.add(url)
                    continue
                if "text/html" not in content_type:
                    continue
            except Exception as e:
                print(f"  WARNING: Could not fetch {url}: {e}")
                continue

            html = page.content()  # fully rendered HTML, after JavaScript has run
            pages.append((url, html))
            print(f"  Fetched page ({len(pages)}/{max_pages}): {url}")

            soup = BeautifulSoup(html, "html.parser")
            all_links = soup.find_all("a", href=True)
            new_links_added = 0
            for link in all_links:
                href = urljoin(url, link["href"])
                href = href.split("#")[0]
                if href.lower().endswith(".pdf"):
                    pdf_urls.add(href)
                elif same_domain(href, base_domain) and href not in visited:
                    to_visit.append((href, depth + 1))
                    new_links_added += 1
            print(f"    (found {len(all_links)} total <a> links on this page, {new_links_added} new same-domain pages queued)")

            # ── onclick="window.open(...)" extraction ──
            # Some sites write their real link destination directly inside an
            # onclick attribute instead of a normal href — no click simulation
            # needed, we can just read the URL straight out of the attribute text.
            onclick_urls = find_onclick_links(soup, url)
            for href in onclick_urls:
                if href.lower().endswith(".pdf"):
                    pdf_urls.add(href)
                elif same_domain(href, base_domain) and href not in visited:
                    to_visit.append((href, depth + 1))
            if onclick_urls:
                print(f"    (found {len(onclick_urls)} link(s) via onclick attributes)")

            # ── Content-type check for extension-less links ──
            # A link like /download.php?id=42 won't be recognized as a PDF by its
            # spelling alone. For same-domain links with no known extension, do a
            # quick check of what the server actually serves at that address.
            extensionless_checked = 0
            for link in all_links:
                href = urljoin(url, link["href"]).split("#")[0]
                if (same_domain(href, base_domain) and not href.lower().endswith(".pdf")
                        and "." not in href.rsplit("/", 1)[-1] and href not in pdf_urls):
                    content_type = check_content_type(href)
                    extensionless_checked += 1
                    if content_type:
                        pdf_urls.add(href)
                        print(f"    (found file by content-type check: {href} -> {content_type})")
            if extensionless_checked:
                print(f"    (checked {extensionless_checked} extension-less link(s) by content-type)")

            # ── Click-exploration pass ──
            # Some sites (accordions, dropdowns, single-page apps) only reveal their
            # real links/PDFs after a click, rather than having them in the page from
            # the start, and don't expose them via onclick either. This is the
            # slowest, least reliable method, so it's the last resort.
            revealed = explore_by_clicking(page, base_domain)
            for href in revealed:
                if href.lower().endswith(".pdf"):
                    pdf_urls.add(href)
                elif same_domain(href, base_domain) and href not in visited:
                    to_visit.append((href, depth + 1))
            if revealed:
                print(f"    (click-exploration revealed {len(revealed)} additional link(s))")

            time.sleep(REQUEST_DELAY)

        browser.close()

    return pages, list(pdf_urls)


def explore_by_clicking(page, base_domain, max_clicks=25):
    """Clicks on elements that look interactive (buttons, list items, elements with
    no real href) to reveal content that only appears after interaction — common
    with accordions, dropdown menus, or single-page apps. Returns any new URLs found
    (from newly-opened tabs, or new <a href> elements that appear after clicking)."""
    found_urls = set()

    # Candidates: things that look clickable but don't already act as a normal link
    selector = "a[href='#'], a[href=''], a:not([href]), button, [role='button'], [onclick]"
    try:
        candidates = page.query_selector_all(selector)
    except Exception:
        return found_urls

    for el in candidates[:max_clicks]:
        try:
            if not el.is_visible():
                continue

            html_before = page.content()

            new_tab_url = None
            try:
                with page.expect_popup(timeout=2000) as popup_info:
                    el.click(timeout=2000)
                new_tab = popup_info.value
                new_tab_url = new_tab.url
                new_tab.close()
            except Exception:
                # No popup opened — that's fine, it might have just expanded content in-page instead
                pass

            if new_tab_url:
                found_urls.add(new_tab_url)

            # Check if new links appeared in the page after this click (e.g. an accordion expanded)
            html_after = page.content()
            if html_after != html_before:
                soup_after = BeautifulSoup(html_after, "html.parser")
                for link in soup_after.find_all("a", href=True):
                    href = urljoin(page.url, link["href"])
                    href = href.split("#")[0]
                    if href.lower().endswith(".pdf") or same_domain(href, base_domain):
                        found_urls.add(href)

        except Exception:
            continue  # a failed click attempt shouldn't stop the whole crawl

    return found_urls


def extract_page_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def extract_pdf_text_from_url(pdf_url: str) -> str:
    try:
        resp = requests.get(pdf_url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        reader = PdfReader(io.BytesIO(resp.content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        print(f"  WARNING: Could not extract PDF {pdf_url}: {e}")
        return ""


# ── LLM-based chunking ────────────────────────────────────────────
CHUNKING_INSTRUCTIONS = """You are preparing a knowledge base for a chatbot that helps players and coaches
with squash tournament registration questions.

Below is raw text extracted from "{label}". Break it into self-contained,
topic-focused chunks suitable for retrieval-based Q&A.

Rules:
- Each chunk should cover ONE clear topic (e.g. entry fees, a specific registration step, withdrawal policy).
- Each chunk's text must make sense on its own, without needing surrounding chunks for context.
- Do not invent information that isn't in the source text.
- Skip navigation boilerplate, cookie notices, and content with no informational value.
- Assign an "audience" of "player", "coach", or "all" based on who the content is relevant to.
- Give each chunk a short, clear "topic" title.

Respond with ONLY a valid JSON array (no markdown fences, no explanation):
[
  {{"topic": "...", "audience": "player|coach|all", "text": "..."}}
]

If there is no substantive information in this text (e.g. it's just a navigation menu), respond with an empty array: []

--- SOURCE TEXT ---
{content}
"""


def chunk_with_llm(label: str, content: str) -> list:
    content = content[:15000]  # keep within a safe token budget per call
    if len(content.strip()) < 100:
        return []  # not enough real content to bother chunking

    prompt = CHUNKING_INSTRUCTIONS.format(label=label, content=content)
    try:
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    except Exception as e:
        print(f"  WARNING: Gemini call failed for '{label}': {e}")
        return []

    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"  WARNING: Could not parse Gemini's output for '{label}'. Skipping.")
        return []


def fetch_extra_page(url: str) -> str:
    """Simple direct fetch for a manually-provided page URL (not part of the crawl)."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"  WARNING: Could not fetch extra URL {url}: {e}")
        return ""


# ── Main pipeline ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Crawl a tournament website and build its knowledge base.")
    parser.add_argument("--url", required=True, help="Base URL of the tournament website")
    parser.add_argument("--slug", required=True, help="Short unique identifier for this tournament, e.g. 'redtone-2026'")
    parser.add_argument("--extra-urls", default="", help="Comma-separated list of additional known URLs (pages or PDFs) to include, for content the crawler can't auto-discover (e.g. links hidden behind a click/accordion)")
    parser.add_argument("--login-url", default="", help="If provided, logs in here before crawling, to access login-gated pages (e.g. a player/coach dashboard). Username/password/selectors are read from environment variables, never from the command line.")
    parser.add_argument("--username-selector", default="input[name='username']", help="CSS selector for the username field on the login page")
    parser.add_argument("--password-selector", default="input[name='password']", help="CSS selector for the password field on the login page")
    parser.add_argument("--submit-selector", default="button[type='submit']", help="CSS selector for the login form's submit button")
    args = parser.parse_args()

    login_params = None
    if args.login_url:
        username = os.environ.get("CRAWL_LOGIN_USERNAME")
        password = os.environ.get("CRAWL_LOGIN_PASSWORD")
        if not username or not password:
            print("ERROR: --login-url was given but CRAWL_LOGIN_USERNAME / CRAWL_LOGIN_PASSWORD environment variables are not set.")
            sys.exit(1)
        login_params = {
            "login_url": args.login_url,
            "username": username,
            "password": password,
            "username_selector": args.username_selector,
            "password_selector": args.password_selector,
            "submit_selector": args.submit_selector,
        }

    print(f"Crawling {args.url} (max {MAX_PAGES} pages, depth {MAX_DEPTH})...")
    pages, pdf_urls = crawl_site(args.url, login_params=login_params)
    print(f"\nFound {len(pages)} pages and {len(pdf_urls)} PDF(s) via crawling.\n")

    # Add any manually-specified extra URLs the crawler couldn't find on its own
    extra_urls = [u.strip() for u in args.extra_urls.split(",") if u.strip()]
    already_have = {u for u, _ in pages} | set(pdf_urls)
    for url in extra_urls:
        if url in already_have:
            continue
        if url.lower().endswith(".pdf"):
            pdf_urls.append(url)
            print(f"  Added extra PDF: {url}")
        else:
            html = fetch_extra_page(url)
            if html:
                pages.append((url, html))
                print(f"  Added extra page: {url}")

    if extra_urls:
        print(f"\nAfter adding extras: {len(pages)} pages and {len(pdf_urls)} PDF(s) total.\n")

    all_chunks = []
    chunk_counter = 1

    for url, html in pages:
        text = extract_page_text(html)
        title = urlparse(url).path or url
        print(f"Chunking page: {title}")
        raw_chunks = chunk_with_llm(f"Website page: {url}", text)
        for c in raw_chunks:
            all_chunks.append({
                "id": f"chunk_{chunk_counter:03d}",
                "source": f"Website: {url}",
                "topic": c.get("topic", "Untitled"),
                "audience": c.get("audience", "all"),
                "text": c.get("text", ""),
            })
            chunk_counter += 1
        print(f"  -> {len(raw_chunks)} chunks")

    for pdf_url in pdf_urls:
        print(f"Extracting and chunking PDF: {pdf_url}")
        text = extract_pdf_text_from_url(pdf_url)
        if not text.strip():
            continue
        raw_chunks = chunk_with_llm(f"PDF: {pdf_url}", text)
        for c in raw_chunks:
            all_chunks.append({
                "id": f"chunk_{chunk_counter:03d}",
                "source": f"PDF: {pdf_url}",
                "topic": c.get("topic", "Untitled"),
                "audience": c.get("audience", "all"),
                "text": c.get("text", ""),
            })
            chunk_counter += 1
        print(f"  -> {len(raw_chunks)} chunks")

    output_path = f"kb_chunks_{args.slug}.json"
    with open(output_path, "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"\nDone. {len(all_chunks)} total chunks written to {output_path}.")
    print("Review this file for accuracy before running precompute_embeddings.py on it.")
    print(f"(Tip: rename/copy it, or pass --slug through your embeddings/chat setup, to keep '{args.slug}' as this tournament's identifier throughout.)")


if __name__ == "__main__":
    main()
