# Public Home Credit code study

These are code references, not additional training data. Exact versions and hashes are recorded in [the source manifest](/Users/ryokitano/Documents/Competitions/GCI_World/research/source_manifest.json). Downloaded notebook outputs were excluded. The extracted files preserve code-cell numbers, may contain notebook-only syntax, and are **not runnable GCI training scripts**. None has been executed.

## Will Koehrsen: Getting Started / A Gentle Introduction

Read [Start Here: A Gentle Introduction](https://www.kaggle.com/code/willkoehrsen/start-here-a-gentle-introduction) alongside the inspected [author-hosted notebook](https://github.com/WillKoehrsen/kaggle-credit-data-science-competition/blob/3ea52480c618a80de925b9ecccde3a1f4a024b96/notebooks/1-Getting%20Started.ipynb). The GitHub snapshot is the version inspected here, not a guarantee that every Kaggle revision is identical.

Open [the extracted code](/Users/ryokitano/Documents/Competitions/GCI_World/research/reference_code/koehrsen_getting_started.py). Cells 39/41 flag and remove the employment placeholder. Cell 67 constructs a small polynomial feature set from credit scores and age. Cells 76/77 add ratios. Cell 113's model function removes IDs, constructs folds, fills OOF predictions, and averages test predictions.

Adapt the feature hypotheses and fold-prediction pattern. Modernize deprecated `Imputer` and LightGBM fit arguments. Fit learned transforms inside folds, use stratification, and handle unknown categories without discarding columns through an inner alignment.

One naming trap: its `CREDIT_TERM` is annuity divided by credit. That is the inverse of the credit/annuity proxy used in the proposed plan. Neither ratio gives the exact contractual duration.

## Aguiar: Simple Features / 7th Place Solution

Reading links: [LightGBM with Simple Features](https://www.kaggle.com/code/jsaguiar/lightgbm-with-simple-features) and [LightGBM 7th place solution](https://www.kaggle.com/code/jsaguiar/lightgbm-7th-place-solution). The second identifies the author's repository. The inspected repository is a related solution version, not the exact original Simple Features notebook.

The [application pipeline](https://github.com/js-aguiar/home-credit-default-competition/blob/d846e3f2477c0bf466128ab18db1971f85eb2c91/pipeline/application_pipeline.py) combines application tables, cleans values, creates score interactions and ratios, computes group statistics, and encodes categories. It offers transferable hypotheses, but its full pipeline assumes many more columns and tables. Its fixed weights and bins need fresh validation. Fit group statistics inside training folds. Replace deprecated pandas usage and unsafe NaN equality checks.

Open [the local application source](/Users/ryokitano/Documents/Competitions/GCI_World/research/reference_code/aguiar_application_pipeline.py). Its global filters would delete a GCI test row with income 117 million and rare gender categories. Keep all GCI rows. Preserve the original category or add an unusual-value indicator. Document/house-history features and the broad drop list do not apply to the reduced schema.

The separate [model implementation](https://github.com/js-aguiar/home-credit-default-competition/blob/d846e3f2477c0bf466128ab18db1971f85eb2c91/model.py) saves OOF predictions, fold-averaged test predictions, and importances, and tunes hyperparameters with Hyperopt. Open [its local snapshot](/Users/ryokitano/Documents/Competitions/GCI_World/research/reference_code/aguiar_model.py). Reuse that experiment structure with a current callbacks API and your own Optuna search. Ordinary bagging requires a positive frequency. Read [current LightGBM parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html#bagging-freq) before transferring old sampling settings.

## Neptune / Minerva: Prediction Averaging

Inspect [the authors' blending notebook](https://github.com/minerva-ml/open-solution-home-credit/blob/0a0a92268974b2cd050ac9ecbf264111036034d7/notebooks/prediction_averaging.ipynb) and [its extracted code](/Users/ryokitano/Documents/Competitions/GCI_World/research/reference_code/minerva_prediction_averaging.py). Cell 6 normalizes predictions to ranks within folds. Cell 7 examines correlation. Cells 9–11 optimize weights and create submissions by averaging fold predictions.

Transfer OOF alignment, model-diversity diagnostics, and coarse blending experiments. Add separate meta-validation or an untouched audit set to assess weight selection. The same OOF rows used to optimize weights cannot provide an unbiased estimate of the resulting gain. Its old toolkit dependencies are unnecessary for a small modern implementation. Rank outputs are ranking scores, not calibrated probabilities.

## Will Koehrsen: Manual Feature Engineering

The [author's manual-feature notebook](https://github.com/WillKoehrsen/kaggle-credit-data-science-competition/blob/3ea52480c618a80de925b9ecccde3a1f4a024b96/notebooks/2-Introduction%20to%20Manual%20Feature%20Engineering.ipynb), also available on [Kaggle](https://www.kaggle.com/code/willkoehrsen/introduction-to-manual-feature-engineering), explains relationship-table aggregation and includes a cross-validation model function. Open [the extracted version](/Users/ryokitano/Documents/Competitions/GCI_World/research/reference_code/koehrsen_manual_features.py).

Most of its bureau-history and auxiliary-table features cannot be constructed from the GCI inputs. Study the aggregation reasoning, but implement peer comparisons from permitted application rows instead. The model function's historical fold and library arguments also need modernization. Original Kaggle leaderboard scores do not predict performance on GCI's reduced columns and modified split.

## A current neural alternative

The [TabM authors' repository](https://github.com/yandex-research/tabm) includes an end-to-end notebook and explains parameter-sharing ensembles. It is a useful second-stage experiment after strong tree baselines. Train from scratch on the allowed data. Follow its distinction between optimizing component losses during training and averaging probabilities during inference.

## Reading method for each code block

1. Identify its input columns, whether it uses labels, and whether any required table is unavailable.
2. Translate it into the proposed mechanism: affordability, combined score, missingness, peer comparison, or variance reduction.
3. Identify where it learns statistics and which rows it sees. Move learned statistics inside the appropriate folds.
4. Adapt it to the GCI schema with explicit missing/unknown cases and preserved IDs.
5. Run a single controlled ablation and record both gains and failures. Cite the author beside adapted code and in the final notebook bibliography.

Aguiar and Minerva source snapshots include their MIT licenses. Keep source attribution even when an implementation is rewritten. Do not use saved public predictions, engineered datasets, original Kaggle labels, or label-matching tricks.
