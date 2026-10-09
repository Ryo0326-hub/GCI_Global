# GCI World 2026: Home Credit competition plan

Prepared October 8, 2026, using the supplied materials, the local competition data, and inspected public source code. The leaderboard target of **0.79169** comes from the supplied screenshot. It is a snapshot, not a verified current or final winning score.

Prioritize **LightGBM and CatBoost using all 32 supplied features**, then improve their features and tune them with Optuna. Add XGBoost or a neural model when validation demonstrates useful additional signal. Choose the final model or blend by its ability to generalize. A larger ensemble does not automatically improve AUC.

October 9 progress: B01 CatBoost completed five development folds at **0.752544** OOF AUC, and you confirmed its exported Drive CSV earned **0.756**, rank **668**. A controlled affordability feature screen raised LightGBM from **0.748699** to **0.757137** OOF AUC. The next target is **0.80 ROC AUC**, with **0.85** as a stretch goal; neither is currently demonstrated. Confirm affordability with CatBoost and another seed, then tune the stronger models. See [[research/FEATURE_SCREEN]] and [[04 Submission Tracker]].

## Competition facts and constraints

| Item | Verified finding | Consequence |
| --- | --- | --- |
| Training data | 171,202 rows, 32 predictors, an ID, and a target | This is a manageable tabular classification problem. |
| Positive labels | 13,821 defaults, or 8.0729% | Stratify validation splits and optimize ROC-AUC. |
| Test data | 61,500 rows with the same predictor schema | Preserve every test ID and the sample submission order. |
| Metric | ROC-AUC | Submit continuous model probabilities. Threshold tuning does not improve AUC. |
| Public scoring | A subset of the test data | Treat public results as noisy feedback. |
| Final scoring | The README says the entire test set | Optimize generalization across the full population. |
| Final file | The **last submitted CSV** determines the final ranking | Restore the selected final file after any experimental submission. |
| External data | Prohibited at every stage | Do not import original Kaggle rows, labels, auxiliary tables, or pretrained model weights learned from external data. |
| Public reference code | Explicitly permitted, with attribution | Record the original author, URL, version, and adaptations. |
| Hand labeling | Prohibited, including conditional manual overrides | Generate every prediction through the reproducible model pipeline. |
| Code submission | Required for Honors/Outstanding consideration | Deliver code that reproduces the selected CSV, including dependencies and citations. |
| Deadline | November 20, 2026, 01:00 UTC | **November 19, 2026, 8:00 p.m. Toronto time (EST).** |

Rule source: [competition README](/Users/ryokitano/Downloads/README.ipynb), particularly Evaluation and Rules. The [tutorial slides](/Users/ryokitano/Downloads/competition_tutorial.pptx) also confirm the deadline and code-submission workflow. The README's detailed final-scoring description governs this plan.

Audited data: [training file](/Users/ryokitano/Downloads/Competition/input/train.csv), [test file](/Users/ryokitano/Downloads/Competition/input/test.csv), and [sample submission](/Users/ryokitano/Downloads/Competition/input/sample_submission.csv). The Drive competition folder and matching input bytes were verified during setup. Every run checks the saved hashes again.

The supplied `EXT_SOURCE_1`, `EXT_SOURCE_2`, and `EXT_SOURCE_3` columns are allowed competition inputs, despite the word “external” in their names.

## What the data and tutorial suggest

These observations inform experiments. They do not establish that a feature will improve validation AUC.

| Observation in your data | Experiment to prioritize |
| --- | --- |
| `EXT_SOURCE_1` is 69.47% missing, `EXT_SOURCE_3` 31.88%, and `OWN_CAR_AGE` 66.00% | Preserve missingness. Test missing flags and available-score summaries. |
| Rows missing `EXT_SOURCE_3` have an observed default rate of 8.86%, versus 7.71% when present | Missingness may carry signal beyond the score itself. Confirm conditional predictive value in validation. |
| `DAYS_EMPLOYED=365243` occurs in 30,898 training rows | Flag the placeholder and replace it with missing before calculating tenure ratios. |
| That employment flag exactly matches `ORGANIZATION_TYPE='XNA'` in training | The signal is partly redundant. Do not mistake a duplicate flag for new independent information. |
| Maximum income is 13.5 million in training but 117 million in test | Compare raw, log, and training-derived clipping treatments. Keep all test rows. |
| Test contains two zero employment durations | Guard divisions even when training has no zero denominator. |
| `FLAG_MOBIL` is 1 for all test rows and all but one training row | Test dropping this nearly constant column. |
| IDs are consecutive and occupy disjoint train/test ranges | Exclude IDs from prediction features. Consecutive IDs do not establish a chronological split. |

The tutorial notebook inside [tutorial.zip](/Users/ryokitano/Downloads/tutorial.zip) uses only five input features. Its count encoding removes categorical identity, its preprocessing is fitted before the validation split, and its final random forest remains fitted to the 70% training portion. A fresh pipeline can use all supplied predictors, fit learned preprocessing inside each fold, and train final models on the complete labeled data after selection.

The attached [submission](/Users/ryokitano/Downloads/submission.csv) has 61,500 unique IDs in the correct sample order and finite probabilities within [0, 1]. Its score and exact model provenance are unknown. Keep it as an existing artifact, not a performance benchmark.

## First 72 hours

1. **Freeze the inputs and validation design.** Save hashes, schema, seeds, and row IDs. Reserve a stratified 20% audit set. Create fixed stratified folds on the remaining development data. Save their assignments.
2. **Build two complete baselines.** Train LightGBM with categorical features and CatBoost with native categorical inputs. Use all 32 predictors, exclude ID, and use minimal cleaning. Save out-of-fold predictions, test predictions, fold AUCs, and training times.
3. **Verify the complete export.** Join predictions to the sample by ID, check every row, and generate a CSV without an index column. Check the existing submission's score in Omnicampus if known, then submit one stronger validated baseline manually.
4. **Test the first feature groups separately.** Start with credit-score combinations and affordability ratios. Compare them on identical folds before combining them.

The first milestone is a trustworthy measured baseline and a reproducible CSV. Do not spend the first days on an exhaustive grid search or a large neural architecture.

## Validation design

Use a stratified 80/20 development/audit split, with a fixed seed such as `20261009`. This produces 136,961 development rows and 34,241 audit rows when the test-size rounding follows scikit-learn. The audit set must stay out of early stopping, feature selection, hyperparameter search, and blend-weight tuning.

On development data:

- Use a fixed five-fold `StratifiedKFold` for feature and model comparisons. Every row receives a prediction from a model that did not train on that row. These are the **out-of-fold (OOF) predictions**.
- Fit imputers, scalers, quantile transforms, frequency encoders, group statistics, and category vocabularies using each fold's training rows. Apply the fitted transformation to validation and test. Define a fallback for unseen categories.
- Any target encoding needs an additional inner split to construct training features without their own labels. CatBoost's native handling is preferable to an improvised target encoder for the first baseline.
- Use three fixed development folds for the main Optuna search. Recheck the top few configurations on the five-fold comparison scheme.
- Repeat the strongest candidates using another fold seed. Compare pooled OOF AUC, per-fold changes, and stability by subgroup. Fold standard deviation alone is not a confidence interval or proof of significance.
- Diagnose train/test shift using predictor distributions and, if necessary, a classifier that predicts whether a row is from train or test. Remove ID from that classifier. An elevated score is a reason to investigate, not proof that adversarial weighting will help.

Preselect at most three finalists using development results. Freeze their designs and inspect the audit set once near the end. Compare their prediction differences using a paired, stratified bootstrap on audit rows. This estimates evaluation uncertainty conditional on fitted models, not all uncertainty from model selection or retraining.

A development improvement of roughly 0.0005 or more can be a useful screening threshold. It is a planning heuristic, not a claim of statistical significance. Smaller consistent gains can matter, particularly in ensembles. Public AUC and local AUC cannot be directly equated because they use different rows.

## Feature experiments

Use the meanings in [the supplied column dictionary](/Users/ryokitano/Downloads/HomeCredit_columns_description.xlsx). Test one group at a time, then combine groups that improve repeated validation. Keep the original columns alongside derived features unless an ablation supports removing them.

| Priority | Feature group | Concrete candidates |
| --- | --- | --- |
| 1 | Credit-score combinations | Available-score mean, minimum, maximum, standard deviation, number observed, pairwise products, and pairwise differences of `EXT_SOURCE_1/2/3`. |
| 1 | Affordability and financing | `AMT_CREDIT / AMT_INCOME_TOTAL`, `AMT_ANNUITY / AMT_INCOME_TOTAL`, `AMT_CREDIT / AMT_ANNUITY`, `AMT_CREDIT / AMT_GOODS_PRICE`, and `AMT_CREDIT - AMT_GOODS_PRICE`. |
| 2 | Household resources | Income, credit, and annuity per family member. Compare children count with family size rather than assuming that every non-child is an earning adult. |
| 2 | Employment and life stage | Age in years, cleaned tenure in years, tenure/age, registration recency/age, ID-document recency/age, and interactions with income type. |
| 2 | Missingness and unusual values | Per-column missing flags, number of missing predictors, score-availability pattern, employment-placeholder flag, zero-phone-change flag, and a car-age-at-least-60 flag. |
| 3 | Peer comparisons | Fold-trained median income by occupation, education, or organization. Add individual income/peer median and individual minus peer score mean. Use coarse groups, minimum support, and fallback to broader groups. |
| 3 | Categories and credit enquiries | A few supported category crosses such as contract × income type. Frequency encoding as an additional feature. Bureau enquiry indicators and smoothed ratios, such as MON/(YEAR+1), without pretending the available windows form a complete history. |
| 4 | Alternative representations | `log1p` of positive amounts, coarse age bands, and a small set of score × age or score × affordability interactions. Test targeted polynomial features only after the simpler groups. |

The public application-feature code reviewed for this plan contains useful score combinations, ratios, and peer statistics. These transfer as hypotheses; its fixed weights and bins are not established as optimal here. [Aguiar's application pipeline](https://github.com/js-aguiar/home-credit-default-competition/blob/d846e3f2477c0bf466128ab18db1971f85eb2c91/pipeline/application_pipeline.py)

Implementation details that matter:

- Define ratios only for valid denominators. Emit missing plus a denominator flag when undefined. Replace infinities with missing.
- `CREDIT/ANNUITY` is a repayment-period proxy, not the actual contractual term. Interest and payment frequency matter. `ANNUITY/INCOME` is a mathematical burden proxy until the amounts' periods are confirmed.
- Available-score summaries must remain missing when no score exists. Do not let a missing-skipping product silently turn an all-missing row into 1.
- Preserve rare categories such as `XNA` rather than dropping customers. Retain raw phone-change zero and old car ages initially. Compare proposed cleaning instead of assuming unusual values are errors.
- Use training-fold thresholds for any clipping. Apply the same transformation to every validation/test row.
- Fit peer statistics inside folds. Do not include validation rows in their own reference population. Keep train-only statistics as the first implementation.
- Do not download bureau, previous-application, installments, or credit-card tables from the original competition. They are absent from this permitted dataset.

## Models and tuning

Start with unweighted binary log loss as the training objective and AUC for validation/early stopping. An 8.1% default rate does not, by itself, require SMOTE or class balancing. Compare mild positive weighting as a later controlled experiment.

| Model | Role | Initial experiment budget |
| --- | --- | --- |
| LightGBM | Main model and feature-testing engine | 40–60 Optuna trials on three development folds. |
| CatBoost | Main alternative with different categorical handling | 20–30 trials, after a native-category baseline. |
| XGBoost with histogram training | Additional candidate for ensemble diversity | 10–20 trials if it adds useful validation signal. |
| Regularized logistic regression | Sanity check and possible small blend component | A few regularization settings. |
| Neural MLP / TabM | Optional competitor and ensemble component | Initially 8–12 trials, capped around 15% of total experiment compute. |

These are starting budgets, not a commitment to run every trial. Measure five pilot fits, estimate total time, and resize the budgets for the available CPU/GPU. CPU training is a suitable first route for the tree models. Use a GPU for neural work if available. Copy the allowed input files from Drive to Colab's local runtime before training and save completed artifacts back to Drive.

Prefer a seeded Optuna TPE search to a large grid. Many interacting parameters make an exhaustive grid wasteful. Early stopping handles the useful number of trees, while pruning and a wall-clock limit constrain weak trials. Preserve the study so interrupted runs can resume. [Optuna TPE documentation](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html)

Proposed LightGBM search ranges:

| Parameter | Starting range |
| --- | --- |
| `learning_rate` | 0.01–0.07, logarithmic |
| `num_leaves` | 15–63 |
| `max_depth` | 5–10, or an independently tested unrestricted variant |
| `min_child_samples` | 50–500 |
| `colsample_bytree` | 0.65–1.0 |
| `subsample` | 0.7–1.0, with `subsample_freq=1` for GBDT bagging |
| `reg_alpha`, `reg_lambda` | Explicit zero candidates plus logarithmic positive ranges up to about 20 |
| `n_estimators` | High cap such as 6,000, with AUC early stopping after 200 non-improving rounds |

Constrain leaves by finite depth. Start with GBDT, then test GOSS separately with compatible sampling parameters. Set the evaluation metric explicitly to AUC and use the current callbacks API. A subsample below 1 does not activate ordinary bagging when its frequency remains zero. These are experimental ranges, not transferred winning settings. [LightGBM tuning](https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html), [sampling parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html#bagging-freq)

For CatBoost, start with depth 4–8, learning rate 0.01–0.08, L2 leaf regularization 1–30, and validation AUC early stopping. For XGBoost, start with depth 3–7 and tune minimum child weight, row/column sampling, learning rate, and regularization. Expand a search only when the initial results identify a promising boundary. [CatBoost tuning](https://catboost.ai/docs/en/concepts/parameter-tuning)

Record every run's input hashes, source citations, feature version, fold assignments, seeds, parameters, software versions, runtime, fold AUCs, pooled OOF AUC, prediction file hashes, and decision. Include failed and rejected experiments so later work does not repeat them.

## Neural networks: when they are worth trying

**A neural network is worth a controlled experiment, but it is not the preferred first investment.** Your inputs are mixed numerical/categorical fields with substantial missingness, and there is no image, text, or sequence information to motivate a specialized deep architecture.

Tree-model benchmark evidence supports keeping boosted trees as serious baselines, but the frequently cited 2022 study focuses on medium-sized datasets around 10,000 samples. It does not settle the outcome on these 171,202 rows. [Grinsztajn, Oyallon, and Varoquaux](https://arxiv.org/abs/2207.08815)

Test a small MLP trained from scratch: two or three hidden layers, categorical one-hot inputs or embeddings, training-fold numeric imputation and scaling, missing indicators, dropout, and AdamW. Monitor validation AUC. Start with widths such as 128–64 or 256–128–64, dropout 0.1–0.4, and a small learning-rate search. Use the same development folds as the tree comparisons.

Then test TabM if a GPU and time are available. It is a modern architecture that efficiently shares parameters across multiple MLP-like predictions. Train its component predictions with separate losses and average probabilities at inference, following the authors' example. Initialize it from scratch on the allowed training data. [TabM authors' repository and example](https://github.com/yandex-research/tabm)

Promote a neural model when it wins as a standalone model **or** reliably improves a tree blend. A slightly weaker neural model may still be useful if its errors differ. Stop neural exploration when it fails both tests after the allocated budget. Do not assume that training longer or increasing depth will help.

## Ensemble decisions

Save OOF and test predictions keyed by ID for every candidate. Align all candidates to the same row and fold assignments. Compare a simple probability average first. Search only a coarse set of nonnegative weights that sum to one, including zero weight for each optional model.

Evaluate blend weights on separate OOF meta-validation splits: choose weights using one part and assess them on another. Confirm the selected blend on the untouched audit set. A gain measured on the same OOF rows used to optimize weights is optimistic. Prediction correlation is a diagnostic; measured incremental AUC decides membership.

Compare probability blending with rank normalization only as a secondary AUC experiment, applying the same method consistently. Rank scores need not be calibrated probabilities. Prefer the simpler probability blend when results are similar. The reviewed public notebook illustrates OOF alignment and rank blending. [Neptune/Minerva prediction averaging](https://github.com/minerva-ml/open-solution-home-credit/blob/0a0a92268974b2cd050ac9ecbf264111036034d7/notebooks/prediction_averaging.ipynb)

Start with LightGBM + CatBoost. Add XGBoost or the neural model only after a repeatable gain. Test two or three model seeds after model selection; seed averaging can reduce variance without adding a new architecture.

A learned stacker is optional. If attempted, use genuinely nested validation that regenerates base-model predictions inside each meta-training split, or judge it on a held-out dataset untouched by all training and selection. Merely cross-validating a stacker on one globally constructed OOF matrix does not remove every source of indirect leakage.

Once choices are frozen, train the selected base-model families using five folds on **all** 171,202 labeled rows. Average their test predictions across folds and approved seeds, then apply the already chosen blend. This final refit uses the former audit labels for training and must not be presented as a new independent audit result.

## Schedule to the deadline

| Dates, Toronto | Work | Completion criterion |
| --- | --- | --- |
| Oct 9–11 | Input identity, validation, complete baselines | Reproducible OOF results and a format-checked baseline CSV. |
| Oct 12–18 | Credit-score, affordability, missingness, and employment features | Accepted/rejected ablations on fixed folds. |
| Oct 19–25 | Optuna searches and peer features | Top configurations confirmed on five folds and a second seed. |
| Oct 26–Nov 1 | Model diversity and simple blending | Measured incremental value of CatBoost/XGBoost and candidate blends. |
| Nov 2–8 | Bounded neural experiments and robustness checks | Neural model kept or rejected using the same criteria. |
| Nov 9–15 | Freeze finalists and inspect audit results | Final architecture and blend chosen with documented evidence. |
| Nov 16–17 | Full-data training, clean reproduction, final CSV/code package | Reproduction succeeds and final candidate is submitted and scored. |
| Nov 18–19 | Verify last submission and retain time for corrections | Correct selected CSV and matching code ZIP confirmed before 8 p.m. Nov 19. |

Use public submissions to check major validated milestones, not hundreds of feature/weight variations. Track the public/local difference across a few candidates, but do not assume the difference is a constant transferable offset. The current leader may improve. A public score above 0.79169 is a milestone, not proof of victory.

## Final submission and reproducibility

Check that the selected CSV has exactly `SK_ID_CURR,TARGET`, 61,500 rows, exactly the expected ID set/order, no duplicates, no missing or infinite predictions, and values within [0, 1]. Verify every test row passes through the model, including extreme incomes and rare categories. Do not make conditional manual prediction overrides.

Run the final notebook or script from a clean session using only the allowed inputs. Save dependency versions, fold/seed configuration, feature definitions, blend weights, and source citations. Package all code dependencies needed to reproduce the CSV in the code ZIP. Save the CSV hash and match the uploaded file to it. Verify Omnicampus has finished scoring and displays the final upload timestamp. Because the last upload counts, check it again after any correction.

## Public code study

Read the accompanying [code study notes](/Users/ryokitano/Documents/Competitions/GCI_World/research/CODE_STUDY.md). They explain which public code blocks transfer, which require changes, and which depend on unavailable tables. Reference snapshots preserve original cell numbers, exclude notebook outputs, and have not been executed.

The recommended immediate implementation is the validation/export pipeline plus the two all-feature tree baselines. Subsequent work should follow measured ablations, not a predetermined belief that any model family must win.
