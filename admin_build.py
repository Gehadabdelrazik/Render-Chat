"""
Admin tool for building a tournament's knowledge base — Python/Flask version,
matching build-chatbot.php's workflow.

Run this alongside (or instead of) App.py during setup:
    python admin_build.py
Then visit: http://localhost:5001

Folder expectations (same folder as your other scripts):
    kb-staging/kb_chunks_<slug>.json     (unpublished, reviewed chunks)
    kb/kb_embeddings_<slug>.json         (published, live — read by App.py)
"""

import os
import sys
import json
import subprocess
from flask import Flask, request, render_template_string

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
<title>Build Tournament Chatbot</title>
<style>
    body { font-family: -apple-system, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; color: #14171a; }
    h1 { font-size: 22px; }
    label { display: block; margin-top: 16px; font-weight: 600; font-size: 14px; }
    input[type=text] { width: 100%; padding: 8px; margin-top: 4px; box-sizing: border-box; }
    button { margin-top: 16px; padding: 10px 18px; background: #c41230; color: white; border: none; border-radius: 6px; cursor: pointer; }
    button.secondary { background: #14171a; }
    .message { background: #fff8ec; border: 1px solid #c48212; padding: 12px; margin-top: 16px; white-space: pre-wrap; font-size: 13px; }
    .chunk { border: 1px solid #ddd; border-radius: 6px; padding: 10px; margin-top: 10px; }
    .chunk-topic { font-weight: 600; }
    .chunk-audience { font-size: 11px; color: #666; text-transform: uppercase; }
    textarea { width: 100%; height: 400px; font-family: monospace; font-size: 12px; margin-top: 12px; }
</style>
</head>
<body>
<h1>Build Tournament Chatbot</h1>
<p>Step 1: enter a tournament's website URL and a short slug, then build. Step 2: review the generated chunks, edit if needed, then publish.</p>

{% if message %}
<div class="message">{{ message }}</div>
{% endif %}

{% if not chunks %}
<form method="post" action="/build">
    <label>Tournament website URL</label>
    <input type="text" name="url" placeholder="https://klsquash.org/NEW-TOURNAMENT/" required>
    <label>Slug (used internally, e.g. "newtourney-2027")</label>
    <input type="text" name="slug" placeholder="newtourney-2027" required pattern="[a-zA-Z0-9_-]+">
    <label>Known extra URLs (optional) — one per line. Use this for guide/rules PDFs the crawler can't find automatically.</label>
    <textarea name="extra_urls" style="height: 100px;" placeholder="https://example.com/player-guide.pdf"></textarea>

    <label>Login-gated pages? (optional) — provide a test account so the crawler can access player/coach dashboard content that's hidden behind login.</label>
    <input type="text" name="login_url" placeholder="Login page URL, e.g. https://klsquash.org/REDTONE_APCA/login.php">
    <input type="text" name="username" placeholder="Test account username" style="margin-top: 8px;">
    <input type="password" name="password" placeholder="Test account password" style="margin-top: 8px;">
    <small style="color: #666;">Leave blank if this tournament has no login-gated content worth crawling. Credentials are used only for this one crawl and are not saved anywhere.</small>

    <button type="submit">Build Knowledge Base</button>
</form>

<hr style="margin-top: 30px;">
<h3 style="font-size: 15px;">Already built a KB but didn't publish it yet?</h3>
<form method="post" action="/resume">
    <label>Slug of the staged build</label>
    <input type="text" name="slug" placeholder="redtone-complete" required>
    <button type="submit" class="secondary">Resume Review</button>
</form>

{% else %}
<h2>Review: {{ slug }} ({{ chunks|length }} chunks)</h2>
{% for c in chunks %}
<div class="chunk">
    <div class="chunk-topic">{{ c.topic }}</div>
    <div class="chunk-audience">{{ c.audience }} · {{ c.source }}</div>
    <div>{{ c.text }}</div>
</div>
{% endfor %}

<form method="post" action="/publish">
    <input type="hidden" name="slug" value="{{ slug }}">
    <label>Edit raw chunk JSON if needed, then publish:</label>
    <textarea name="edited_chunks">{{ chunks_json }}</textarea>
    <button type="submit">Publish to Live Chatbot</button>
</form>
<form method="get" action="/">
    <button type="submit" class="secondary">Start Over</button>
</form>
{% endif %}
</body>
</html>
"""


def run_command(cmd_list):
    """Runs a command as a list of args (safer than a shell string — avoids the
    quoting/escaping issues we ran into on the PHP side with Windows)."""
    env = os.environ.copy()  # inherits GEMINI_API_KEY etc. from however this Flask app was started
    result = subprocess.run(cmd_list, cwd=BASE_DIR, env=env, capture_output=True, text=True)
    output = result.stdout + result.stderr
    return output, result.returncode


@app.route("/")
def index():
    return render_template_string(PAGE_TEMPLATE, message="", chunks=None, slug="", chunks_json="")


@app.route("/build", methods=["POST"])
def build():
    url = request.form.get("url", "").strip()
    slug = "".join(c for c in request.form.get("slug", "") if c.isalnum() or c in "_-")
    extra_urls = request.form.get("extra_urls", "").strip()
    extra_urls_arg = ",".join(line.strip() for line in extra_urls.splitlines() if line.strip())

    login_url = request.form.get("login_url", "").strip()
    login_username = request.form.get("username", "").strip()
    login_password = request.form.get("password", "")  # not stripped — passwords can legitimately have leading/trailing spaces

    if not url or not slug:
        return render_template_string(PAGE_TEMPLATE, message="Please provide a valid URL and slug.", chunks=None, slug="", chunks_json="")

    cmd = [sys.executable, "crawl_and_build_kb.py", "--url", url, "--slug", slug, "--extra-urls", extra_urls_arg]

    env = os.environ.copy()
    if login_url:
        if not login_username or not login_password:
            return render_template_string(PAGE_TEMPLATE, message="Login URL was provided but username/password is missing.", chunks=None, slug="", chunks_json="")
        cmd += ["--login-url", login_url]
        env["CRAWL_LOGIN_USERNAME"] = login_username
        env["CRAWL_LOGIN_PASSWORD"] = login_password

    result = subprocess.run(cmd, cwd=BASE_DIR, env=env, capture_output=True, text=True)
    output = result.stdout + result.stderr

    generated_path = os.path.join(BASE_DIR, f"kb_chunks_{slug}.json")
    if os.path.exists(generated_path):
        staged_path = os.path.join(STAGING_DIR, f"kb_chunks_{slug}.json")
        os.replace(generated_path, staged_path)
        with open(staged_path) as f:
            chunks = json.load(f)
        message = f"Crawl complete. Review the {len(chunks)} chunks below before publishing.\n\n--- Script output ---\n{output}"
        return render_template_string(PAGE_TEMPLATE, message=message, chunks=chunks, slug=slug, chunks_json=json.dumps(chunks, indent=2))
    else:
        message = f"Build failed. Script output:\n{output}"
        return render_template_string(PAGE_TEMPLATE, message=message, chunks=None, slug="", chunks_json="")


@app.route("/resume", methods=["POST"])
def resume():
    slug = "".join(c for c in request.form.get("slug", "") if c.isalnum() or c in "_-")
    staged_path = os.path.join(STAGING_DIR, f"kb_chunks_{slug}.json")
    if os.path.exists(staged_path):
        with open(staged_path) as f:
            chunks = json.load(f)
        message = f"Resumed staged build for '{slug}' ({len(chunks)} chunks)."
        return render_template_string(PAGE_TEMPLATE, message=message, chunks=chunks, slug=slug, chunks_json=json.dumps(chunks, indent=2))
    else:
        message = f"No staged build found for slug '{slug}'."
        return render_template_string(PAGE_TEMPLATE, message=message, chunks=None, slug="", chunks_json="")


@app.route("/publish", methods=["POST"])
def publish():
    slug = "".join(c for c in request.form.get("slug", "") if c.isalnum() or c in "_-")
    edited_json = request.form.get("edited_chunks", "")

    try:
        decoded = json.loads(edited_json)
    except json.JSONDecodeError:
        message = "The chunk data isn't valid JSON — no changes were published. Please fix and try again."
        return render_template_string(PAGE_TEMPLATE, message=message, chunks=None, slug=slug, chunks_json=edited_json)

    staged_path = os.path.join(STAGING_DIR, f"kb_chunks_{slug}.json")
    with open(staged_path, "w") as f:
        json.dump(decoded, f, indent=2)

    # precompute_embeddings.py reads kb_chunks_<slug>.json from the current working
    # directory, so copy the staged (possibly edited) version there first
    working_copy = os.path.join(BASE_DIR, f"kb_chunks_{slug}.json")
    with open(working_copy, "w") as f:
        json.dump(decoded, f, indent=2)

    cmd = [sys.executable, "precompute_embeddings.py", "--slug", slug]
    output, returncode = run_command(cmd)

    embeddings_path = os.path.join(BASE_DIR, f"kb_embeddings_{slug}.json")
    if os.path.exists(embeddings_path):
        live_path = os.path.join(LIVE_KB_DIR, f"kb_embeddings_{slug}.json")
        os.replace(embeddings_path, live_path)
        message = f"Published. This tournament's chatbot is now live using slug '{slug}'. Set window.sqTournamentId = \"{slug}\"; on that tournament's page."
        return render_template_string(PAGE_TEMPLATE, message=message, chunks=None, slug="", chunks_json="")
    else:
        message = f"Embedding step failed. Script output:\n{output}"
        return render_template_string(PAGE_TEMPLATE, message=message, chunks=decoded, slug=slug, chunks_json=edited_json)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
