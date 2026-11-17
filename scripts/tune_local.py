"""Run a bounded, single-writer development search and save a confirmation config."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import run_local  # Reuse the existing macOS OpenMP setup.
import gci_pipeline as gci


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--study-dir", required=True)
    parser.add_argument("--model", choices=("lightgbm", "catboost"), required=True)
    parser.add_argument("--trials", type=int, default=20, help="Additional trials; run one writer per study")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--best-config", required=True)
    args = parser.parse_args()
    cfg = gci.Config(**json.loads(Path(args.config).read_text()))
    best = gci.tune_model(args.input, args.study_dir, cfg, args.model, args.trials, args.timeout)
    selected = asdict(cfg)
    selected.update(label=cfg.label + "_tuned", n_splits=5, models=[args.model])
    selected["model_params"][args.model] = best
    destination = Path(args.best_config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    gci.save_json(destination, selected)
    print(f"Five-fold confirmation configuration: {destination}")


if __name__ == "__main__":
    main()
