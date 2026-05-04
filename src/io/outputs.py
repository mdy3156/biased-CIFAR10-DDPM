import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def prepare_output_dir(config: dict[str, Any], config_path: str | None = None) -> Path:
    experiment_name = config.get("experiment", "experiment")
    base_dir = Path(config.get("output_root", "results"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = config.get("run_name")
    dir_name = f"{timestamp}_{run_name}" if run_name else f"{timestamp}_{experiment_name}"
    out_dir = base_dir / dir_name
    counter = 1
    while out_dir.exists():
        out_dir = base_dir / f"{dir_name}_{counter}"
        counter += 1
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    if config_path:
        shutil.copy(config_path, out_dir / "config.original.yaml")
    return out_dir
