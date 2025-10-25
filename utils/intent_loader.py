# utils/intent_loader.py
import json

def load_intents(file_path: str):
    """Load intents and utterances from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    intents = []
    utterances = []
    for entry in data["intents"]:
        for text in entry["text"]:
            utterances.append(text)
            intents.append(entry["intent"])
    return utterances, intents
