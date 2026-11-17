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

Pending the controlled experiments. Report failed ideas alongside successful ones; do not describe an adopted feature as an improvement until the comparison is complete.
