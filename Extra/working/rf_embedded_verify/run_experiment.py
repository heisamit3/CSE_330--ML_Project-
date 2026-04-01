from __future__ import annotations

import argparse
import random

import numpy as np

from src.eval.run_cv import run_cv
from src.experiment import ExperimentRunner
from src.utils.io import read_yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/quick_run.yaml")
    parser.add_argument("--mode", default=None)
    parser.add_argument("--run-cv", action="store_true")
    args = parser.parse_args()

    config = read_yaml(args.config)
    if args.mode:
        config["mode"] = args.mode
    set_seed(int(config.get("seed", 42)))

    if args.run_cv:
        outputs = run_cv(config)
        print(outputs["summary"])
        return

    runner = ExperimentRunner(config)
    data_bundle = runner.load_data_bundle()
    train_df = data_bundle["train"]
    test_df = data_bundle["test"]
    candidate_df = data_bundle["candidates"]

    if not test_df.empty:
        result = runner.run_mode(train_df, test_df, candidate_df, mode=config.get("mode"), prefix="test")
        print(result["paths"]["submission"])
    else:
        result = runner.run_mode(train_df, train_df, candidate_df, mode=config.get("mode"), prefix="train_eval")
        print({"metric": result["metric"], "submission_path": str(result["paths"]["submission"])})


if __name__ == "__main__":
    main()
