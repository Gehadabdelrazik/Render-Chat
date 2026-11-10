"""
Flask version of the chatbot backend — kept in sync with chat.php.

This reads the SAME kb_embeddings_<slug>.json files produced by
precompute_embeddings.py, so both the Python and PHP versions share one
knowledge base format. No Chroma here anymore — plain cosine similarity,
matching exactly what chat.php does, so behavior is consistent between
the two.

Folder expectations (same project folder as your other tools):
    kb/kb_embeddings_<slug>.json   — one file per tournament

Run:
    python app.py
"""



import os
import json
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.6-flash"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(BASE_DIR, "kb")  # one kb_embeddings_<slug>.json per tournament

SYSTEM_INSTRUCTIONS = """You are a helpful assistant answering questions about this website, using
information gathered from the website itself. Answer the user's question using ONLY the information
provided below for anything specific to this website or organization (details, policies, procedures,
pricing, dates, or anything else that could plausibly differ from one website to another). Be concise
and friendly. If a detail isn't in the provided information, say you're not sure and suggest the user
check the website directly or contact them — don't guess at anything that could vary by organization.
 
You may answer briefly and confidently from your own general knowledge only for well-established facts
that are clearly NOT specific to this website and have no reasonable chance of varying between
organizations (for example, a one-sentence definition of a well-known general term related to the
website's subject matter). Keep such answers to a sentence or two. If there's any real chance the answer
depends on this particular website's own details, policies, or context, say you're not sure instead
of guessing."""

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot / (norm_a * norm_b)


def load_kb(tournament: str):
    kb_path = os.path.join(KB_DIR, f"kb_embeddings_{tournament}.json")
    if not os.path.exists(kb_path):
        return None
    with open(kb_path, "r") as f:
        return json.load(f)


def answer_question(question: str, tournament: str, role: str = None) -> dict:
    kb = load_kb(tournament)
    if kb is None:
        return {"error": f"No knowledge base found for tournament '{tournament}'"}, 404

    last_error = None
    for attempt in range(3):
        try:
            # 1. Embed the question
            embed_result = client.models.embed_content(model=EMBEDDING_MODEL, contents=question)
            question_vector = embed_result.embeddings[0].values

            # 2. Score every chunk, optionally filtering by role
            scored = []
            for chunk in kb:
                if role in ("player", "coach") and chunk["audience"] not in ("all", role):
                    continue
                score = cosine_similarity(question_vector, chunk["embedding"])
                scored.append((score, chunk))

            if not scored:  # role-filtering removed everything — fall back to unfiltered
                for chunk in kb:
                    score = cosine_similarity(question_vector, chunk["embedding"])
                    scored.append((score, chunk))

            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[:3]

            context = "\n\n".join(c["text"] for _, c in top)
            sources = [c["topic"] for _, c in top]

            # 3. Role-aware prompt
            if role == "player":
                role_note = "The person asking is logged in as a PLAYER. Tailor instructions to what a player does on the site."
            elif role == "coach":
                role_note = "The person asking is logged in as a COACH/TEAM MANAGER. Tailor instructions to what a coach does on the site."
            else:
                role_note = "The person's role (player or coach) is not known. Cover both cases briefly if relevant."

            prompt = f"{SYSTEM_INSTRUCTIONS}\n\n{role_note}\n\n--- INFORMATION ---\n{context}\n\n--- QUESTION ---\n{question}"

            # 4. Generate the answer
            response = client.models.generate_content(model=CHAT_MODEL, contents=prompt)

            return {"answer": response.text, "sources": sources}, 200

        except Exception as e:
            last_error = str(e)
            time.sleep(1.5 * (attempt + 1))

    return {
        "answer": "I'm getting a lot of questions right now and can't respond at the moment. Please try again in a minute, or contact them directly if it's urgent.",
        "sources": [],
        "is_error": True,
        "error_detail": last_error,
    }, 200


app = Flask(__name__)
CORS(app)  # tighten to your real domain(s) before deploying anywhere public


@app.route("/", methods=["GET"])
def home():
    """Simple health check, so opening the server URL in a browser shows it's running."""
    sites = sorted(
        f[len("kb_embeddings_"):-len(".json")]
        for f in os.listdir(KB_DIR)
        if f.startswith("kb_embeddings_") and f.endswith(".json")
    ) if os.path.isdir(KB_DIR) else []
    return jsonify({
        "status": "Chatbot server is running",
        "sites": sites,
        "endpoints": {
            "POST /chat": "Ask a question: {question, tournament, role}",
            "GET /kb-status?tournament=<slug>": "Check a site's knowledge base is loaded",
        },
    })


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = (data.get("question") or "").strip()
    tournament = data.get("tournament")
    role = data.get("role")
    if role not in ("player", "coach"):
        role = None

    if not question:
        return jsonify({"error": "No question provided"}), 400
    if not tournament:
        return jsonify({"error": "Missing 'tournament' identifier"}), 400

    result, status = answer_question(question, tournament, role)
    return jsonify(result), status


@app.route("/kb-status", methods=["GET"])
def kb_status():
    tournament = request.args.get("tournament")
    if not tournament:
        return jsonify({"error": "Provide ?tournament=<slug>"}), 400
    kb = load_kb(tournament)
    if kb is None:
        return jsonify({"tournament": tournament, "found": False}), 404
    return jsonify({"tournament": tournament, "found": True, "chunks_loaded": len(kb)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)