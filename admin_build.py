"""
Admin tool for building a tournament's knowledge base — Python/Flask version,
matching build-chatbot.php's workflow.

Two steps: (1) enter a website URL and slug to crawl the site and stage its
chunks, then (2) click Publish to embed them and make them live.

Run this alongside (or instead of) app.py during setup:
    python admin_build.py
Then visit: http://localhost:5001

Folder expectations (same folder as your other scripts):
    kb-staging/kb_chunks_<slug>.json     (crawled, not yet published)
    kb/kb_embeddings_<slug>.json         (published, live — read by app.py)
"""

import os
import sys
import json
import subprocess
from flask import Flask, request, render_template_string
from jinja2 import DictLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STAGING_DIR = os.path.join(BASE_DIR, "kb-staging")
LIVE_KB_DIR = os.path.join(BASE_DIR, "kb")

os.makedirs(STAGING_DIR, exist_ok=True)
os.makedirs(LIVE_KB_DIR, exist_ok=True)

app = Flask(__name__)

PAGE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Build Chatbot</title>
<style>
    :root {
        --red: #c41230;
        --red-dark: #9e0e27;
        --ink: #14171a;
        --muted: #5f6b76;
        --line: #e3e6ea;
        --bg: #f5f6f8;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.5 -apple-system, "Segoe UI", Roboto, sans-serif; }
    .wrap { max-width: 640px; margin: 0 auto; padding: 48px 16px; }
    header { margin-bottom: 24px; }
    h1 { font-size: 26px; margin: 0 0 6px; }
    h2 { font-size: 16px; margin: 0 0 4px; }
    .lead { color: var(--muted); margin: 0; }
    .card { background: #fff; border: 1px solid var(--line); border-radius: 12px; padding: 28px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
    .card + .card { margin-top: 16px; }

    .field { margin-bottom: 20px; }
    label { display: block; font-weight: 600; font-size: 14px; margin-bottom: 6px; }
    .hint { color: var(--muted); font-size: 13px; margin-top: 6px; }
    input[type=text], input[type=password], textarea {
        width: 100%; padding: 10px 12px; font: inherit; color: inherit;
        border: 1px solid #cfd4da; border-radius: 8px; background: #fff;
    }
    input:focus, textarea:focus { outline: none; border-color: var(--red); box-shadow: 0 0 0 3px rgba(196,18,48,0.12); }
    textarea { min-height: 90px; resize: vertical; font-family: ui-monospace, Consolas, monospace; font-size: 13px; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    @media (max-width: 520px) { .row { grid-template-columns: 1fr; } }

    details.section { border: 1px solid var(--line); border-radius: 8px; margin-bottom: 14px; }
    details.section > summary { cursor: pointer; padding: 12px 14px; font-weight: 600; font-size: 14px; list-style: none; display: flex; justify-content: space-between; }
    details.section > summary::-webkit-details-marker { display: none; }
    details.section > summary::after { content: "+"; color: var(--muted); font-weight: 400; }
    details.section[open] > summary::after { content: "\\2212"; }
    details.section > summary .opt { color: var(--muted); font-weight: 400; margin-left: 6px; }
    .section-body { padding: 0 14px 14px; }
    .section-body .field:last-child { margin-bottom: 0; }

    button, .btn {
        display: inline-flex; align-items: center; justify-content: center; gap: 10px;
        width: 100%; margin-top: 8px; padding: 12px 18px; font: inherit; font-weight: 600;
        background: var(--red); color: #fff; border: 1px solid transparent; border-radius: 8px; cursor: pointer; text-decoration: none;
    }
    button:hover, .btn:hover { background: var(--red-dark); }
    button.secondary { background: var(--ink); }
    button.secondary:hover { background: #2b3036; }
    .btn.ghost { background: #fff; color: var(--ink); border-color: #cfd4da; }
    .btn.ghost:hover { background: var(--bg); }
    button:disabled { opacity: 0.75; cursor: wait; }
    .spinner { width: 16px; height: 16px; border: 2px solid rgba(255,255,255,0.4); border-top-color: #fff; border-radius: 50%; animation: spin 0.8s linear infinite; display: none; }
    button.loading .spinner { display: inline-block; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .inline { display: flex; gap: 10px; align-items: flex-start; }
    .inline input { flex: 1; }
    .inline button { width: auto; margin-top: 0; white-space: nowrap; }
    @media (max-width: 520px) { .inline { flex-direction: column; align-items: stretch; } .inline button { width: 100%; } }

    .alert { border-radius: 8px; padding: 14px 16px; margin-bottom: 20px; font-size: 14px; }
    .alert.error { background: #fdecef; border: 1px solid #f3b8c2; color: #7d0a1e; }
    .alert.success { background: #ecf8f0; border: 1px solid #b5e0c3; color: #14532d; }
    .alert.info { background: #fff8ec; border: 1px solid #ecc88a; color: #6b4507; }
    .alert strong { display: block; font-size: 15px; margin-bottom: 4px; }
    code, pre { font-family: ui-monospace, Consolas, monospace; font-size: 13px; }
    pre.snippet { background: var(--ink); color: #f5f6f8; padding: 12px 14px; border-radius: 8px; overflow-x: auto; margin: 10px 0 0; }
    pre.log { background: #fafbfc; border: 1px solid var(--line); padding: 12px; border-radius: 8px; max-height: 320px; overflow: auto; white-space: pre-wrap; margin: 0; }
    details.logbox { margin-top: 16px; }
    details.logbox > summary { cursor: pointer; color: var(--muted); font-size: 13px; margin-bottom: 8px; }
</style>
</head>
<body>
<div class="wrap">
    <header>
        <h1>Build Chatbot</h1>
        <p class="lead">Enter a website and we'll crawl it, build its knowledge base, and publish it to the live chatbot.</p>
    </header>

    {% if state == "published" %}
    <div class="card">
        <div class="alert success">
            <strong>Chatbot published</strong>
            <code>{{ slug }}</code> is now live with {{ chunk_count }} chunks. Add this to that tournament's page, before the chat widget:
            <pre class="snippet">&lt;script&gt;window.sqTournamentId = "{{ slug }}";&lt;/script&gt;</pre>
        </div>
        <p class="hint">To make it live on the hosted chatbot, commit and push the new <code>kb/kb_embeddings_{{ slug }}.json</code> file.</p>
        <a class="btn" href="/">Build another chatbot</a>
        {% include "log" %}
    </div>

    {% elif state == "staged" %}
    <div class="card">
        {% if error %}
        <div class="alert error"><strong>Publish failed</strong>{{ error }}</div>
        {% else %}
        <div class="alert info">
            <strong>{{ message or "Crawl complete" }}</strong>
            Found {{ chunk_count }} chunks for <code>{{ slug }}</code>. Publish to make them live in the chatbot.
        </div>
        {% endif %}
        <form method="post" action="/publish" data-loading="Publishing&hellip;">
            <input type="hidden" name="slug" value="{{ slug }}">
            <button type="submit"><span class="spinner"></span><span class="label">Publish to Live Chatbot</span></button>
        </form>
        <a class="btn ghost" href="/">Start over</a>
        {% include "log" %}
    </div>

    {% else %}
    <div class="card">
        {% if error %}
        <div class="alert error"><strong>Something went wrong</strong>{{ error }}</div>
        {% endif %}

        <form method="post" action="/build" data-loading="Crawling&hellip; this can take a few minutes">
            <div class="field">
                <label for="url">Website URL</label>
                <input type="text" id="url" name="url" value="{{ form.url }}" placeholder="https://klsquash.org/NEW-TOURNAMENT/" required>
            </div>
            <div class="field">
                <label for="slug">Slug</label>
                <input type="text" id="slug" name="slug" value="{{ form.slug }}" placeholder="newtourney-2027" required pattern="[a-zA-Z0-9_-]+">
                <div class="hint">A short ID for this chatbot: letters, numbers, dashes and underscores only.</div>
            </div>

            <details class="section" {% if form.extra_urls %}open{% endif %}>
                <summary>Extra documents <span class="opt">optional</span></summary>
                <div class="section-body">
                    <div class="field">
                        <label for="extra_urls">Extra URLs, one per line</label>
                        <textarea id="extra_urls" name="extra_urls" placeholder="https://example.com/player-guide.pdf">{{ form.extra_urls }}</textarea>
                        <div class="hint">For guides or rules PDFs the crawler can't find on its own.</div>
                    </div>
                </div>
            </details>

            <details class="section" {% if form.login_url %}open{% endif %}>
                <summary>Login-protected pages <span class="opt">optional</span></summary>
                <div class="section-body">
                    <div class="field">
                        <label for="login_url">Login page URL</label>
                        <input type="text" id="login_url" name="login_url" value="{{ form.login_url }}" placeholder="https://klsquash.org/REDTONE_APCA/login.php">
                    </div>
                    <div class="field row">
                        <div>
                            <label for="username">Test account username</label>
                            <input type="text" id="username" name="username" value="{{ form.username }}" autocomplete="off">
                        </div>
                        <div>
                            <label for="password">Password</label>
                            <input type="password" id="password" name="password" autocomplete="new-password">
                        </div>
                    </div>
                    <div class="hint">Lets the crawler see pages behind a login, like player or coach dashboards. Used only for this crawl and never saved.</div>
                </div>
            </details>

            <button type="submit"><span class="spinner"></span><span class="label">Build Knowledge Base</span></button>
        </form>
        {% include "log" %}
    </div>

    <div class="card">
        <h2>Already built a KB but didn't publish it yet?</h2>
        <p class="hint" style="margin: 0 0 12px;">Enter the slug of the staged build to pick up where you left off.</p>
        <form method="post" action="/resume">
            <label for="resume-slug">Slug of the staged build</label>
            <div class="inline">
                <input type="text" id="resume-slug" name="slug" placeholder="redtone-complete" required>
                <button type="submit" class="secondary">Resume</button>
            </div>
        </form>
    </div>
    {% endif %}
</div>

<script>
    document.querySelectorAll("form[data-loading]").forEach(function (form) {
        form.addEventListener("submit", function () {
            var btn = form.querySelector("button[type=submit]");
            btn.disabled = true;
            btn.classList.add("loading");
            btn.querySelector(".label").innerHTML = form.dataset.loading;
        });
    });
</script>
</body>
</html>
"""

LOG_TEMPLATE = """
{% if output %}
<details class="logbox">
    <summary>Show build log</summary>
    <pre class="log">{{ output }}</pre>
</details>
{% endif %}
"""

app.jinja_loader = DictLoader({"log": LOG_TEMPLATE})

EMPTY_FORM = {"url": "", "slug": "", "extra_urls": "", "login_url": "", "username": ""}


def render(state="form", form=None, **context):
    return render_template_string(PAGE_TEMPLATE, state=state, form=form or EMPTY_FORM, **context)


def clean_slug(value):
    return "".join(c for c in value if c.isalnum() or c in "_-")


def staged_path_for(slug):
    return os.path.join(STAGING_DIR, f"kb_chunks_{slug}.json")


def run_command(cmd_list, env=None):
    """Runs a command as a list of args (safer than a shell string — avoids the
    quoting/escaping issues we ran into on the PHP side with Windows)."""
    env = env or os.environ.copy()  # inherits GEMINI_API_KEY etc. from however this Flask app was started
    result = subprocess.run(cmd_list, cwd=BASE_DIR, env=env, capture_output=True, text=True)
    output = result.stdout + result.stderr
    return output, result.returncode


@app.route("/")
def index():
    return render()


@app.route("/build", methods=["POST"])
def build():
    form = {
        "url": request.form.get("url", "").strip(),
        "slug": clean_slug(request.form.get("slug", "")),
        "extra_urls": request.form.get("extra_urls", "").strip(),
        "login_url": request.form.get("login_url", "").strip(),
        "username": request.form.get("username", "").strip(),
    }
    login_password = request.form.get("password", "")  # not stripped — passwords can legitimately have leading/trailing spaces
    url, slug = form["url"], form["slug"]
    extra_urls_arg = ",".join(line.strip() for line in form["extra_urls"].splitlines() if line.strip())

    if not url or not slug:
        return render(form=form, error="Please provide a valid URL and slug.")

    cmd = [sys.executable, "crawl_and_build_kb.py", "--url", url, "--slug", slug, "--extra-urls", extra_urls_arg]

    env = os.environ.copy()
    if form["login_url"]:
        if not form["username"] or not login_password:
            return render(form=form, error="Login URL was provided but username/password is missing.")
        cmd += ["--login-url", form["login_url"]]
        env["CRAWL_LOGIN_USERNAME"] = form["username"]
        env["CRAWL_LOGIN_PASSWORD"] = login_password

    output, _ = run_command(cmd, env)

    generated_path = os.path.join(BASE_DIR, f"kb_chunks_{slug}.json")
    if not os.path.exists(generated_path):
        return render(form=form, error="The crawl didn't produce any chunks. Check the build log below.", output=output)

    with open(generated_path) as f:
        chunks = json.load(f)
    if not chunks:
        os.remove(generated_path)
        return render(form=form, error="The crawl finished but found no usable content on that site.", output=output)

    os.replace(generated_path, staged_path_for(slug))
    return render("staged", slug=slug, chunk_count=len(chunks), output=output)


@app.route("/resume", methods=["POST"])
def resume():
    slug = clean_slug(request.form.get("slug", ""))
    staged_path = staged_path_for(slug)
    if not os.path.exists(staged_path):
        return render(error=f"No staged build found for slug '{slug}'.")
    with open(staged_path) as f:
        chunks = json.load(f)
    return render("staged", slug=slug, chunk_count=len(chunks), message="Staged build found")


@app.route("/publish", methods=["POST"])
def publish():
    slug = clean_slug(request.form.get("slug", ""))
    staged_path = staged_path_for(slug)
    if not os.path.exists(staged_path):
        return render(error=f"No staged build found for slug '{slug}'.")
    with open(staged_path) as f:
        chunks = json.load(f)

    # precompute_embeddings.py reads kb_chunks_<slug>.json from BASE_DIR, so
    # put a temporary working copy there while it runs
    working_copy = os.path.join(BASE_DIR, f"kb_chunks_{slug}.json")
    with open(working_copy, "w") as f:
        json.dump(chunks, f, indent=2)
    try:
        output, _ = run_command([sys.executable, "precompute_embeddings.py", "--slug", slug])
    finally:
        os.remove(working_copy)

    embeddings_path = os.path.join(BASE_DIR, f"kb_embeddings_{slug}.json")
    if not os.path.exists(embeddings_path):
        return render("staged", slug=slug, chunk_count=len(chunks), output=output,
                      error="The embedding step failed. Check the build log below, then try publishing again.")

    os.replace(embeddings_path, os.path.join(LIVE_KB_DIR, f"kb_embeddings_{slug}.json"))
    return render("published", slug=slug, chunk_count=len(chunks), output=output)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
