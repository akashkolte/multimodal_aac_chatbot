import ollama
import json

# load once at import — not on every call
with open("data/users.json") as f:
    _profile_cache = {u["id"]: u for u in json.load(f)["users"]}

def get_profile(user_id):
    return _profile_cache.get(user_id)

def generate(query, memories, user_id):
    profile = get_profile(user_id)
    name      = profile["name"]
    style     = profile["style"]
    condition = profile["condition"]

    memory_text = "\n".join(f"- [{m['bucket']}] {m['text']}" for m in memories)

    prompt = f"""You are {name}, an AAC device user with {condition}.
Communication style: {style}

Memories:
{memory_text}

Question: {query}

Rules: Only use the memories. Speak in first person. Be brief — 1-3 sentences. If not in memories, say "I don't know."

Answer:"""

    response = ollama.chat(
        model="gpt-oss:120b-cloud",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]