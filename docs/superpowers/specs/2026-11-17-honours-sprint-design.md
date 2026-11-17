# GCI honours sprint design — November 17, 2026

[[00 Dashboard]] · [[docs/superpowers/plans/2026-11-17-honours-sprint|Astra implementation plan]]

Implementation authorized November 17. Ryo selected Astra; implementation and controlled training are in progress.

Latest steering: **aim for >=0.80 public ROC AUC**, while retaining the honours/top-10% objective. This is an aspirational target; experiments must establish achievable performance. Live November 17 verification identified the 0.76253 CSV as F02 affordability LightGBM and the rank 500 public score as 0.77032.

## Goal and constraints

Produce the strongest validated, reproducible submission possible before **November 19, 2026, 8 p.m. Toronto time**. Starting user-reported result: **0.76253 public AUC, rank 3,237**. Target: top 10%, approximately rank 500. The scored CSV was verified as F02 affordability LightGBM; the observed rank 500 score on November 17 is 0.77032. Final scoring uses the entire test set and the last submitted CSV. Honours requires matching reproducible code.

Only supplied competition data is permitted. Cite reference code. No external labels, auxiliary Kaggle data, pretrained external model weights, or manual prediction overrides.

## Compute decision

Recommended immediate route: **Colab GPU for CatBoost; CPU for LightGBM.** CSC CPU resources are an optional additional worker. CSC GPUs can be used after legitimate access/allocation becomes available; do not wait for them.

- Colab: Tesla T4 allocated and native CatBoost GPU fitting verified November 17. GPU settings are explicit in every trial and confirmation; source/backend identity is recorded.
- Tiramisu: published two RTX PRO 6000 Blackwell GPUs, 96 GB each. Current device ACLs exclude rkitano; NVML fails and CUDA initialization returns 100. It needs an access/allocation grant. Current CPU output reports 128 visible logical CPUs, 377 GiB RAM, about 317 GiB available and Python 3.13.5.
- Mannitol: the supplied November 17 12:51 snapshot shows RTX 3050, 6,144 MiB total, 5,643 MiB occupied and 100% utilization. Do not schedule our training over that workload.
- Begin shared CPU work with four threads. If Colab cannot allocate a GPU promptly, continue CPU LightGBM and a reduced CPU CatBoost search.

Alternatives considered: CSC GPU as the primary route is currently blocked/occupied; Colab-only is a suitable simpler fallback if a second environment delays modelling. Distributed training, service infrastructure and LLM training are poor uses of the remaining time.

## Evidence and priorities

B01 development OOF: LightGBM **0.748699**, CatBoost **0.752544**. F02 affordability LightGBM: **0.757137**, improving every fold versus raw LightGBM. These are local scores, not public results. Score summaries barely helped; combining missingness/tenure with affordability lost to affordability alone.

The saved Colab baseline was interrupted partway through CatBoost. Reuse completed local baseline evidence. New GPU runs are distinct variants; do not claim feature gains from a confounded CPU/GPU comparison.

The existing Optuna objective replaces fixed per-model parameters with sampled ones. This drops GPU settings. Returning/applying best parameters can drop them again. Preserving backend settings across search and confirmation is the first required coding fix.

Keep audit seed **20261009**, audit fraction **0.20**, first comparison seed **42** and five confirmation folds. There are 136,961 development rows and 34,241 audit rows. All learned preprocessing stays within training folds. Audit is excluded from tuning/early stopping until finalist designs are frozen.

## Schedule and acceptance

1. November 17: confirm scored-file identity/cutoff; GPU pilot with a 30-minute setup cap; affordability comparisons; initial 20-trial, three-fold Optuna searches with two-hour limits per model.
2. November 18: confirm strongest candidates on five folds and a second seed; freeze at most three finalists; inspect audit once for the final choice.
3. November 19: final refit on all **171,202** labelled rows. Target verified CSV/code ZIP by **noon**, leaving the afternoon for scoring and corrections before **8 p.m.**

Retain fitted folds, OOF/test predictions, code/input/config/package hashes and the exact CSV. GPU CatBoost is nondeterministic; new training need not produce identical bytes. Verify saved-model inference against the frozen CSV. Preserve historical manifests. Do not delay experiments for a broad infrastructure/resumption rewrite.

Deliver exactly **61,500** probabilities in sample ID order, schema **SK_ID_CURR,TARGET**, finite values in [0,1], and a matching code ZIP with citations/configuration. No automated competition upload.

## Sources and limits

- Competition README: `/Users/ryokitano/Downloads/README.ipynb`, re-read November 17 and matched to Drive.
- [CSC machine list](https://wiki.csclub.uwaterloo.ca/Machine_List) and [usage agreement](https://csclub.uwaterloo.ca/resources/machine-usage-agreement/).
- [CatBoost GPU documentation](https://catboost.ai/docs/en/features/training-on-gpu) and [Colab FAQ](https://research.google.com/colaboratory/faq.html).
- [CUDA errors](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__TYPES.html): CUDA_ERROR_NO_DEVICE = 100.

The four optional named skill files were not found in this session's skill roots/plugin cache. Their invocation is not claimed. The current scripts and official library documentation cover the immediate work. More compute increases experiment capacity; it does not guarantee top-10% performance.

## Authorized extension: derived categories and model diversity

Ryo requested deeper feature engineering and model diversity during implementation on November 17. Continue before opening the reserved audit set. The chosen implementation adds three independently screened groups: category combinations with fold-trained frequencies, credit-enquiry window comparisons, and hypothetical amortizing-loan pricing for fixed candidate terms. Each is a deterministic transformation of supplied predictors; no generated monthly repayment history is represented as an observation. The actual past-payment histories and learned duration model in the fifth-place solution require absent source tables.

Add XGBoost as a third model family, using native categories and an explicit CPU/CUDA setting with fallback rejection. Preserve saved-model inference and the common folds. Cap its initial comparison/search at one hour; screen new feature groups on the development rows, then confirm promising candidates on five folds. Prefer broad, coarse blend weights over a high-dimensional stack on this dataset. Keep the current verified candidate and artifacts available throughout. No additional data, labels or pretrained models are introduced.

Alternatives considered: imputing unavailable history columns would introduce assumptions without new observed evidence; a temporal network would model fabricated sequences. Derived groups and a different boosting algorithm are testable with the existing split and fit the remaining schedule. A neural application-only model remains secondary until a stronger deliverable is secure.
