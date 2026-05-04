import argparse

from ..experiments.train_ddpm import run_train_ddpm
from ..io.config import load_yaml_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run biased CIFAR-10 DDPM experiments from a YAML config."
    )
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    experiment_type = config.get("experiment", "train_ddpm")

    if experiment_type == "train_ddpm":
        run_train_ddpm(config, config_path=args.config)
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")


if __name__ == "__main__":
    main()
