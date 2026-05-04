from typing import Any

import yaml


def load_yaml_config(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)
