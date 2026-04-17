"""Derive users.json from per-persona memory JSONs in memories/.

The memory JSONs are the source of truth for each persona's structured profile.
This script reads every memories/*.json and projects a thin index entry into
users.json for the frontend PersonaSelector and for session lookup.

Run after editing any persona JSON:
    cd data && python generate_users.py
"""

import json
from pathlib import Path


def main() -> None:
    memories_dir = Path("memories")
    if not memories_dir.is_dir():
        raise SystemExit(f"Expected directory not found: {memories_dir.resolve()}")

    user_index: list[dict] = []
    for path in sorted(memories_dir.glob("*.json")):
        with open(path) as f:
            persona = json.load(f)
        profile = persona["profile"]
        prefs = profile.get("stylistic_preferences") or {}
        access = (profile.get("access_needs") or {}).get("input_method")
        tone_summary = ", ".join(prefs.get("tone", []))

        user_index.append(
            {
                "id": profile["id"],
                "name": profile["name"],
                "age": profile.get("age"),
                "condition": profile["condition"],
                "access_method": access,
                "tone_summary": tone_summary,
                "file": f"memories/{profile['id']}.json",
            }
        )

    out_path = Path("users.json")
    with open(out_path, "w") as f:
        json.dump({"users": user_index}, f, indent=2, ensure_ascii=False)

    print(f"Indexed {len(user_index)} personas → {out_path}")
    for u in user_index:
        print(f"  {u['id']:25s} {u['name']:30s} {u['condition']}")


if __name__ == "__main__":
    main()
