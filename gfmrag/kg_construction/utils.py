import json
import os
import re

KG_DELIMITER = ","


def processing_phrases(phrase: str | int | float | bool | None) -> str:
    if phrase is None:
        return ""
    if isinstance(phrase, (int, float, bool)):
        return str(phrase)  # deal with numeric / bool values
    if not isinstance(phrase, str):
        phrase = str(phrase)
    return re.sub("[^A-Za-z0-9 ]", " ", phrase.lower()).strip()


def directory_exists(path: str) -> None:
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


def extract_json_dict(text: str) -> str | dict:
    pattern = r"\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\})*)*\})*)*\}"
    match = re.search(pattern, text)

    if match:
        json_string = match.group()
        try:
            json_dict = json.loads(json_string)
            return json_dict
        except json.JSONDecodeError:
            return ""
    else:
        return ""
