"""Compare feature groups on the existing fixed development folds; audit stays unused."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import run_local  # Reuse the local import/OpenMP setup without starting its CLI.
import gci_pipeline as gci

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--baseline", required=True, help="Manifest of the completed raw-feature development baseline")
parser.add_argument("--output", default=str(root / "output/screening"))
args = parser.parse_args()
baseline = json.loads(Path(args.baseline).read_text())
baseline_config = gci.Config(**baseline["config"])
if baseline_config.mode != "development" or baseline_config.feature_groups or baseline_config.evaluate_audit:
    raise ValueError("Use a raw-feature development baseline with audit evaluation disabled.")
if baseline["pipeline_sha256"] != gci.sha256(gci.__file__):
    raise ValueError("Baseline pipeline differs from this version. Re-run a comparable baseline first.")
_, _, _, hashes = gci.load_inputs(args.input)
if hashes != baseline["input_hashes"]:
    raise ValueError("Baseline input bytes differ from the screening inputs.")
output = Path(args.output)
baseline_score = baseline["model_metrics"]["lightgbm"]["oof_auc"]
screens = [("F01_scores", ("scores",)), ("F02_affordability", ("affordability",)),
           ("F03_missingness_tenure", ("missingness", "tenure"))]
completed = []
accepted = []
for label, groups in screens:
    cfg = replace(baseline_config, label=label, models=("lightgbm",), feature_groups=groups, blend_weights=None)
    manifest = gci.run_experiment(args.input, output, cfg, root / "Experiments/Runs")
    delta = manifest["selected_oof_auc"] - baseline_score
    completed.append({"label": label, "groups": groups, "oof_auc": manifest["selected_oof_auc"],
                      "delta_vs_lightgbm_baseline": delta, "manifest": str(Path(manifest["run_dir"]) / "manifest.json")})
    if delta >= 0.0005:
        accepted.extend(groups)
if len(accepted) > 1:
    cfg = replace(baseline_config, label="F04_accepted_combination", models=("lightgbm",),
                  feature_groups=tuple(accepted), blend_weights=None)
    manifest = gci.run_experiment(args.input, output, cfg, root / "Experiments/Runs")
    completed.append({"label": cfg.label, "groups": cfg.feature_groups, "oof_auc": manifest["selected_oof_auc"],
                      "delta_vs_lightgbm_baseline": manifest["selected_oof_auc"] - baseline_score,
                      "manifest": str(Path(manifest["run_dir"]) / "manifest.json")})
gci.save_json(output / "feature_screen.json", {"baseline_run": baseline["run_id"],
              "lightgbm_baseline": baseline_score, "screens": completed,
              "selection_rule": "development screening only; 0.0005 is a heuristic, not a significance test"})
print(json.dumps(completed, indent=2))
