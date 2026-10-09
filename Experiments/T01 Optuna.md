---
type: experiment
status: planned
created: 2026-10-09
tags: [gci, experiment]
---
# T01 Optuna

[[03 Experiments]] · [[06 Decisions]]

Hypothesis: Tuning regularization and tree complexity improves a validated feature set.

Configuration: `Keep the accepted groups fixed; enable RUN_OPTUNA.`

Start with affordability alone after its CatBoost/second-seed confirmation. Score combinations were deferred, and adding missingness/tenure reduced affordability's AUC. Use a new label, keep audit disabled, search three development folds, then confirm selected parameters on five folds. See [[research/FEATURE_SCREEN]].

Comparison: same input hashes, five folds, seed 42. Audit disabled during exploration.

- [ ] Run the controlled experiment.
- [ ] Link its generated report below.
- [ ] Record the decision and next step.

Report:
Decision:
