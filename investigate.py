## Import the necessary modules
import json
import os
import ollama

## Import the function from the module parse_data
from parse_data import load_items, get_unclaimed_items, save_result


## The Qwen model served by Ollama. Override with the OLLAMA_MODEL env var
## if a different tag (e.g. qwen2.5:7b) has been pulled locally.
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen2.5")

DATA_FILE = "found_items.json"
OUTPUT_FILE = os.path.join("output", "match_result.json")


## Build your prompt based on the description the user provides
## and the items that are available in the lost-and-found database.
## The model must follow the rules listed in the README file
## The function should return the system prompt and the user prompt.
## You may need to use json.dumps() to convert the available_items list into a JSON string.
def build_prompt(description, available_items):
    system_prompt = (
        "You are a campus lost-and-found assistant.\n"
        "Rules for the Model:\n"
        "- You must use only the given JSON list of items.\n"
        "- Not all the details of an item must match to be a possible match.\n"
        "- Only JSON must be returned, with exactly the following structure:\n"
        "{\n"
        '    "matches": ["ITEM_ID"],\n'
        '    "confidence": "LOW"\n'
        "}\n"
        '- "matches" contains all the possible matches (use the exact item IDs from the list).\n'
        '- "confidence" measures how confident you are about the matches. '
        'It must be exactly one of: LOW, MEDIUM, HIGH.\n'
        "- If there is no match, return an empty list for \"matches\".\n"
        "- Do not output any explanation, notes, or text outside the JSON object.\n"
    )

    user_prompt = (
        'The user describes the item they lost:\n"' + description + '"\n\n'
        "The items available in the lost-and-found database are:\n"
        + json.dumps(available_items, indent=2)
        + '\n\nReturn ONLY the JSON object with "matches" and "confidence".'
    )

    return system_prompt, user_prompt


## Logic to ask Qwen for all the possible matches based on the system prompt and user prompt.
## The function should return the response from Qwen.
def ask_qwen(system_prompt, user_prompt):
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response["message"]["content"]


## Logic to parse the response from Qwen and return the result.
## You may need to use json.loads() to convert the response string into a suitable Python data structure.
def parse_response(response_text):
    text = response_text.strip()

    # Some models wrap JSON in ```json ... ``` fences; strip them if present.
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop the opening fence line (``` or ```json)
        lines = lines[1:]
        # Drop the trailing fence line, if any
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Find the outermost JSON object in case extra prose sneaks in.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    return json.loads(text)


## Logic to validate the result returned by Qwen.
## It should check if the result is a dictionary, contains the keys "matches" and "confidence",
## and that the values are of the correct type.
## If everything is correct, then it should check if the item IDs in the "matches" list are valid IDs.
def validate_result(result, available_items):
    if not isinstance(result, dict):
        return False
    if "matches" not in result or "confidence" not in result:
        return False
    if not isinstance(result["matches"], list):
        return False
    if not isinstance(result["confidence"], str):
        return False
    if result["confidence"] not in ("LOW", "MEDIUM", "HIGH"):
        return False

    valid_ids = {item["id"] for item in available_items}
    for item_id in result["matches"]:
        if not isinstance(item_id, str) or item_id not in valid_ids:
            return False

    return True


## Logic to display the matches found by Qwen in a user-friendly format.
def display_matches(result, available_items):
    print("MATCH RESULT")
    print("-" * 50)
    print(f"Confidence: {result.get('confidence', 'UNKNOWN')}")
    print()

    matches = result.get("matches", [])
    if not matches:
        print("No matches were found in the lost-and-found database.")
        print()
        print("Possible matches:")
        print("(empty list)")
        return

    print("Possible matches:")
    print()
    items_by_id = {item["id"]: item for item in available_items}
    for item_id in matches:
        item = items_by_id.get(item_id)
        if item is None:
            continue
        print(f"ID: {item['id']}")
        print(f"Item: {item['item']}")
        print(f"Color: {item['color']}")
        print(f"Location: {item['location']}")
        print(f"Date found: {item['date']}")
        print()


## Control center for the entire program.
def main():
    print("CAMPUS LOST-AND-FOUND ASSISTANT")
    print("=" * 50)
    print()

    # Ask the user for the description of the lost item.
    description = input("Describe the item you lost: ").strip()
    if not description:
        description = "(no description provided)"

    # Load the database and keep only the unclaimed items.
    items = load_items(DATA_FILE)
    available_items = get_unclaimed_items(items)

    print()
    print("Searching for possible matches...")
    print()

    # Build the prompt, ask Qwen, parse and validate.
    system_prompt, user_prompt = build_prompt(description, available_items)
    response_text = ask_qwen(system_prompt, user_prompt)
    result = parse_response(response_text)

    if not validate_result(result, available_items):
        # Fall back to an empty result so the program still exits cleanly.
        result = {"matches": [], "confidence": "LOW"}
        print("Warning: model response was invalid; returning no matches.")
        print()

    # Show the result and persist it.
    display_matches(result, available_items)
    save_result(result, OUTPUT_FILE)
    print(f"Result saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
