# What the strongest public solutions teach us

Research requested during the honours sprint on November 17. Our target is **>=0.80 public ROC AUC**. Published Kaggle scores are reference results on a different dataset and split, not a forecast for this competition. Our files contain only 32 predictors and no auxiliary loan-history tables.

## Sources inspected

| Source | Evidence and code inspected | Transfer to this project |
| --- | --- | --- |
| [Kazuki Onodera — second-place team](https://github.com/KazukiOnodera/Home-Credit-Default-Risk) | Author identifies his contribution as feature engineering. Inspected the current-application section of [`py/000.py`](https://github.com/KazukiOnodera/Home-Credit-Default-Risk/blob/41eb0c0679dbf6cc156f8d3e699b40be731d6e9a/py/000.py) and numeric-feature export in `py/001.py`. | Financial ratios, differences, household composition and repayment-duration proxies. Reimplement a small `loan_terms` candidate using our columns; preserve rows and guard invalid denominators. |
| [João Aguiar — seventh-place ensemble component](https://github.com/js-aguiar/home-credit-default-competition) | Repository reports 0.8044 private AUC for one component. Inspected [`application_pipeline.py`](https://github.com/js-aguiar/home-credit-default-competition/blob/d846e3f2477c0bf466128ab18db1971f85eb2c91/pipeline/application_pipeline.py), `model.py`, and `config.py`. | Financial ratios already helped F02. Test richer peer profiles: education/occupation/age and payment-ratio/income-type/education groups. The source pools train/test for statistics; our version learns them separately inside each training fold. Also test its GOSS modelling idea with current LightGBM settings. |
| [Paweł Godula — fifth-place solution](https://github.com/pawelgodula/kagglehomecredit) | Read the solution explanation and interest-feature methods in [`data_processors.py`](https://github.com/pawelgodula/kagglehomecredit/blob/f4033b9548f526960fd2257bcef80091425cd951/src/data_processors.py). | Loan pricing can encode lender risk assessment. Its learned interest/duration models require previous applications with payment counts, which we do not have. Its temporal CNN and nested loan-history models likewise cannot be reproduced from our files. |
| [Minerva/Neptune open solution](https://github.com/minerva-ml/open-solution-home-credit) | Repository reports 0.79804 private AUC for its final solution. Existing inspected [`prediction_averaging.ipynb`](https://github.com/minerva-ml/open-solution-home-credit/blob/0a0a92268974b2cd050ac9ecbf264111036034d7/notebooks/prediction_averaging.ipynb) demonstrates prediction diversity/averaging. | Compare ID-aligned out-of-fold predictions and coarse probability blends. Different algorithms are useful only if their errors complement one another. Retain the independent audit boundary. |

The Kaggle first-place write-up was located, but its full content was not readable through the research tool. No implementation decision here depends on an unverified summary of it.

## Code walkthrough and controlled experiments

1. **Loan terms.** A tree splits single columns; an explicit ratio gives it a useful ordering directly. In addition to our existing credit/income and credit/annuity ratios, test goods/income, goods/annuity, financing difference/income, income per adult, children/family, and an approximate age at repayment. The repayment value is a mathematical proxy, not an observed contract term or interest rate. Missing or impossible denominators remain missing.
2. **Peer profiles.** Build two demographic/loan groups. Store medians and standard deviations of income, credit, annuity, credit/annuity and mean external score from the training fold only. Compare each applicant with those medians. Unseen or small groups use training-wide fallbacks; no target averages are used. This makes a rich reference idea compatible with honest validation.
3. **Boosting diversity.** Compare a GOSS LightGBM variant with the tuned bagged model on the same development folds. Disable bagging for GOSS. Keep CatBoost's explicit GPU setting through tuning. A published parameter set is a starting hypothesis, not a selected configuration for our data.
4. **Keep only measured gains.** Screen candidates against the current affordability baseline, confirm promising groups on five folds, and check a second seed. Examine error correlation before freezing at most three finalists. Audit labels remain unavailable to feature selection and hyperparameter search.

These additions are authorized by the user's research request. Existing GPU trials continue with their original code snapshot; new feature groups get distinct configurations and source hashes. Nothing is imported from external datasets, predictions, labels or pretrained weights. Reference scripts are read as source material and never executed. Local source caches are excluded from Git; durable citations identify inspected versions.

## Neural networks

The fifth-place example supports neural networks as a complementary representation of rich histories. It does not establish that a neural network will beat boosting on our small set of application columns. Finish and validate the boosting candidates first. A neural experiment is justified only if time remains and it improves validation or adds useful diversity.

## Results

Initial five-fold results (seed 42, development rows only): tuned affordability LightGBM **0.758386**; additional loan terms **0.758191**; rich peer profiles **0.758555**; GOSS **0.757991**. Simple peer features scored **0.758534**. Loan terms and GOSS did not improve this comparison. The rich-profile gain is very small and needs confirmation; published success alone is not sufficient to keep a feature.

An unlabelled distribution check found no exact duplicate rows within development data, and no test matches to development using all predictors or the checked demographic/date signature. Cash-loan shares were 90.55% in development and 90.72% in test; the largest missingness shift was about 0.52 percentage points. The striking contract-type shift described in the fifth-place Kaggle write-up is therefore not present in these supplied files. We are not copying its distribution-specific adjustments.

The bounded comparisons are now complete; final selection is recorded below and in [[03 Experiments]].

## Deeper feature engineering requested during execution

The supplied README allows reproducible transformations of the provided inputs and prohibits external data and hand-labeling. It does not supply historical monthly observations. Deterministic augmentation can expose useful relationships to a model, but cannot restore missing factual information.

New controlled hypotheses:

- **Category combinations:** housing × family status, occupation × organization, education × occupation, income type × education/contract, age band × family status, and repayment-ratio band × contract. A combination can distinguish applicants whose separate categories look similar. Frequencies are fitted on training folds, and unseen validation categories stay unseen.
- **Credit-enquiry patterns:** compare reported monthly and quarterly counts with annual-rate proxies, retaining missingness. These are comparisons between reported windows, not a reconstructed chronology. Do not sum overlapping windows as if they were disjoint events.
- **Pricing scenarios:** for assumed 12/24/36/60-month amortizing terms, solve the rate implied by the supplied credit and annuity. Feasible scenarios are bounded at a hypothetical 0–6% monthly rate. Names explicitly say `RATE_HYPOTHESIS`; neither the true duration nor actual interest rate is known. Invalid credit/annuity values stay missing. This adapts the fifth-place team's scenario reasoning without its unavailable historical labels.
- **XGBoost:** native categorical histogram boosting offers a third set of errors to compare with CatBoost and LightGBM. Preserve explicit device settings, use the common folds, and verify saved inference. A third model is kept only if its own validation or its contribution to the blend is useful.

Existing `loan_terms` and GOSS did not improve their initial five-fold comparison. Therefore the new groups are tested separately, not added wholesale based on intuition. Audit labels stay closed until this bounded extension ends.

Measured extension results: pricing scenarios improved profile LightGBM from **0.758555 to 0.759832**, with **0.760046** on seed 20261117. Category crosses scored **0.748534** and enquiry-window comparisons **0.758370** in that LightGBM comparison, so neither is automatically retained. Native XGBoost with the pricing group reached **0.761092**, versus **0.758902** without pricing. The common-fold 25% LightGBM / 50% shallow CPU CatBoost / 25% XGBoost blend reached **0.763262** on development OOF predictions. This is model-selection evidence, not an independent audit result. Adding the tuned GPU CatBoost to an equal four-way blend did not improve this diagnostic.

Ryo's first submitted sprint candidate (the earlier shallow CPU CatBoost alone) received public AUC **0.76365**, rank **2,918**. It is not the newer pricing ensemble. See [[04 Submission Tracker]] for exact file hashes and scoring evidence.

## Final selection after the bounded extension

Tuned pricing XGBoost reached development AUC **0.761422**, confirmed at **0.761574** with the second seed. GPU category crosses scored **0.757862**, below the tuned GPU affordability baseline **0.758229**, and did not improve the tested blends. They were excluded. The strongest development blend was 25% pricing LightGBM / 50% shallow CPU CatBoost / 25% tuned pricing XGBoost at **0.763285**.

Before reading audit labels, three designs and all component manifest hashes were frozen in `configs/sprint/frozen_finalists.json`. The single reserved audit comparison selected the simpler **50% CPU CatBoost / 50% CPU XGBoost** blend: **0.764923**, versus 0.764597 for the three-model blend and 0.762829 for XGBoost alone. These are held-out selection scores, not public leaderboard results. The selected pair is being refit on the full labelled population. No feature/weight search will follow the audit.

The pricing features gave the clearest additional measured signal. They expose relationships among existing credit and payment fields; they do not add true monthly-history observations. No evidence so far supports a claim of >=0.80 AUC from this reduced schema.
