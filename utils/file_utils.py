import json


def load_specific_positions_from_file(filename: str) -> dict:
    with open(filename, "r") as f:
        data = json.load(f)
    return data
