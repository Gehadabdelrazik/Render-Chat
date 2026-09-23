"""
Run this once per tournament (and again any time that tournament's kb_chunks
file changes) to pre-compute embeddings. This replaces Chroma for the PHP
deployment: since each tournament's KB is small (~15-40 chunks), we can do a
simple brute-force similarity search in PHP instead of running a vector
database server.

Usage:
    python precompute_embeddings.py --slug redtone-2026
"""

import os
import json
import argparse
from google import genai

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
EMBEDDING_MODEL = "gemini-embedding-001"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", required=True, help="Tournament identifier, e.g. 'redtone-2026' — must match the kb_chunks_<slug>.json file")
    args = parser.parse_args()

    kb_path = f"kb_chunks_{args.slug}.json"
    output_path = f"kb_embeddings_{args.slug}.json"

    with open(kb_path, "r") as f:
        chunks = json.load(f)

    output = []
    for i, chunk in enumerate(chunks):
        print(f"Embedding chunk {i+1}/{len(chunks)}: {chunk['id']} ({chunk['topic']})")
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=chunk["text"],
        )
        embedding = result.embeddings[0].values

        output.append({
            "id": chunk["id"],
            "source": chunk["source"],
            "topic": chunk["topic"],
            "audience": chunk.get("audience", "all"),
            "text": chunk["text"],
            "embedding": embedding,
        })

    with open(output_path, "w") as f:
        json.dump(output, f)

    print(f"\nDone. Wrote {len(output)} chunks with embeddings to {output_path}.")
    print(f"Upload this file alongside chat.php on the server, in a folder identifying tournament '{args.slug}'.")


if __name__ == "__main__":
    main()
