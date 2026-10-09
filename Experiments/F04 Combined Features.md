---
type: experiment
status: deferred
created: 2026-10-09
tags: [gci, experiment]
---
# F04 Combined Features

[[03 Experiments]] · [[06 Decisions]]

Hypothesis: Combining affordability with the independently improving missingness/tenure group will improve risk ranking further.

Configuration: `('affordability', 'missingness', 'tenure')`, LightGBM only, the same five development folds and seed 42. Audit disabled.

Report: [[Experiments/Runs/20261009T050127647521Z_F04_accepted_combination]]. OOF AUC **0.756631**, below affordability alone by **0.000506** despite exceeding the raw-feature baseline.

Decision: Defer the combination. Use affordability alone for the next confirmation. Standalone feature gains do not imply a gain when combined. [[research/FEATURE_SCREEN]].
