---
type: data_rules
tags: [gci, validation]
---
# Data and Rules

[[00 Dashboard]] · [[COMPETITION_PLAN]] · [[research/data_audit.json|Data audit]]

171,202 labeled rows. 61,500 test rows. 32 raw predictors. Default rate 8.0729%. ID is excluded from features. The input filenames and Drive file sizes match the local audit. Every training run verifies the exact input hashes before proceeding.

Use only provided train/test data. The supplied EXT_SOURCE fields are permitted. Do not download original Kaggle labels, relational tables or pretrained external weights. No manual conditional prediction overrides. Cite referenced code. Set seeds and preserve dependency versions.

Validation has a fixed 20% audit partition and five development folds. All learned categories, frequency/peer statistics and other fitted transformations stay within training folds. The audit is not used for tuning or early stopping. Final mode uses every labeled row after choices are frozen.

The default baseline retains all customers, preserves categories, and replaces the employment placeholder with missing plus an indicator. Ratio features guard zero denominators. Historical source filters are not copied blindly: test has an income of 117 million that an old filter would delete.

The metric is ROC-AUC. Public scoring uses a subset; the README says final scoring uses the entire test set and the last uploaded CSV. CSV schema is SK_ID_CURR,TARGET with no index column. Code disclosure must reproduce that same candidate.
