import json
import os
from pathlib import Path
from typing import List, Dict
import requests  # DeepSeek uses simple HTTP API

# ===================== CONFIG =====================
INPUT_FOLDER = "processed_data"
OUTPUT_FOLDER = "data1_cleaned"
# Get your key from: https://platform.deepseek.com/api_keys
DEEPSEEK_API_KEY = "DEEPSEEK_API_KEY"  # Set in environment or replace here
# =====================

if not DEEPSEEK_API_KEY:
    raise ValueError("Please set DEEPSEEK_API_KEY in your environment!")

URL = "https://api.deepseek.com/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    "Content-Type": "application/json"
}

SYSTEM_PROMPT = """You are an expert in extracting family relationships from oral history testimonies.
From the narrator's ("I", "my") point of view, identify the real names of these family members.

Return ONLY a clean JSON object with exactly these keys when found (do NOT include nulls or missing ones):

{
  "mother": "Full Name",
  "father": "Full Name",
  "grandmother": "Full Name",           // prefer the most frequently mentioned one
  "grandfather": "Full Name",
  "sister": "Full Name",
  "brother": "Full Name",
  "daughter": "Full Name",
  "son": "Full Name"

Use the most common or full version of the name as it appears in the text.
If multiple names exist for one role (e.g. maiden + married), use the primary one used in the story.
Only include a key if you're 100% sure.
"""

def ask_deepseek(sentences: List[str]) -> Dict[str, str]:
    payload = {
        "model": "deepseek-chat",           # or "deepseek-coder" if you prefer
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Here is the full testimony:\n\n" + "\n".join(
                f"{i+1:3}. {s}" for i, s in enumerate(sentences)
            )}
        ],
        "temperature": 0.0,
        "max_tokens": 512,
        "response_format": { "type": "json_object" }  # forces valid JSON
    }

    try:
        response = requests.post(URL, json=payload, headers=HEADERS, timeout=60)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()

        # Sometimes it adds ```json ... ``` – clean it
        if content.startswith("```"):
            content = content.split("```", 2)[1]
            if content.lower().startswith("json"):
                content = content[4:]

        mapping = json.loads(content)
        # Simplify keys (in case it used maternal_grandmother etc.)
        simplified = {}
        for k, v in mapping.items():
            key = k.replace("maternal_", "").replace("paternal_", "").replace("grandma", "grandmother")
            simplified[key] = v.strip()
        return simplified

    except Exception as e:
        print(f"DeepSeek API error: {e}")
        if 'response' in locals():
            print("Response:", response.text)
        return {}

def replace_generic_subjects(data: dict, family_map: Dict[str, str]):
    """
    Replaces generic kinship terms ('mother', 'grandfather', etc.)
    with the actual names identified by DeepSeek.
    """

    # These keywords will be detected anywhere in the subject string
    role_keywords = {
        "mother": ["mother", "mom", "mum"],
        "father": ["father", "dad"],
        "grandmother": ["grandmother", "grandma", "granny"],
        "grandfather": ["grandfather", "grandpa"],
        "sister": ["sister"],
        "brother": ["brother"],
        "daughter": ["daughter"],
        "son": ["son"],
    }

    for item in data.get("extracted_relationships", []):
        subj_raw = item["subject"]["entity"].strip()
        subj_lower = subj_raw.lower()

        # If the subject already matches a real detected person, skip replacement
        if subj_raw in family_map.values():
            continue

        replaced = False

        for role, keywords in role_keywords.items():
            if role in family_map:  # DeepSeek detected a real name
                for kw in keywords:
                    # word-boundary match (avoids matching 'stepmother', etc.)
                    if f" {kw} " in f" {subj_lower} ":
                        item["subject"]["entity"] = family_map[role]
                        replaced = True
                        break
            if replaced:
                break

def process_file(filepath: Path):
    print(f"Processing → {filepath.name}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    sentences = [entry["sentence"] for entry in data.get("extracted_relationships", [])]
    if not sentences:
        print("  No sentences found.")
        return

    print("  Asking DeepSeek to extract family names...")
    family_map = ask_deepseek(sentences)

    if family_map:
        print("  Detected:", " | ".join(f"{k}: {v}" for k, v in family_map.items()))
    else:
        print("  No family members detected by DeepSeek.")

    replace_generic_subjects(data, family_map)

    out_path = Path(OUTPUT_FOLDER) / filepath.name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved → {out_path}\n")

def main():
    Path(OUTPUT_FOLDER).mkdir(exist_ok=True)
    files = list(Path(INPUT_FOLDER).glob("*.json"))
    print(f"Found {len(files)} files. Starting DeepSeek processing...\n")
    for f in files:
        process_file(f)
    print("All done! Cleaned files are in:", OUTPUT_FOLDER)

if __name__ == "__main__":
    main()