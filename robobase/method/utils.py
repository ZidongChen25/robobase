import re
from typing import Dict

import gymnasium as gym


def extract_from_spec(spec: gym.spaces.Dict, key, missing_ok: bool = False):
    if key not in list(spec.keys()):
        if missing_ok:
            return None
        raise ValueError(
            f"Couldn't find '{key}' in the space. "
            f"Available keys are: {list(spec.keys())}"
        )
    return spec[key]


def extract_many_from_spec(
    spec: gym.spaces.Dict, pattern: str, missing_ok: bool = False
):
    filtered_dict = {}
    regex = re.compile(pattern)
    for key, value in spec.items():
        if regex.search(key):
            filtered_dict[key] = value
    if len(filtered_dict) == 0 and not missing_ok:
        raise ValueError(
            f"Couldn't find the regex key '{pattern}' in the space. "
            f"Available keys are: {list(spec.keys())}"
        )
    return filtered_dict


def extract_many_from_batch(batch: Dict, pattern: str) -> Dict:
    filtered_dict = {}
    regex = re.compile(pattern)
    for key, value in batch.items():
        if regex.search(key):
            filtered_dict[key] = value
    if len(filtered_dict) == 0:
        raise ValueError(
            f"Couldn't find the regex key '{pattern}' in the batch. "
            f"Available keys are: {list(batch.keys())}"
        )
    return filtered_dict
