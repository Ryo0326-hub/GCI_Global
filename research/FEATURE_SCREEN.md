# Feature comparison — October 9, 2026

[[00 Dashboard]] · [[03 Experiments]] · [[06 Decisions]]

Affordability is the strongest measured feature group. It improved LightGBM on every fixed development fold and raised pooled OOF AUC by **0.008438**. Keep it for the next model comparison; confirm it before a larger Optuna search.

| Candidate | LightGBM OOF AUC | Change from raw-feature LightGBM | Decision |
| --- | --- | --- | --- |
| [[Experiments/Runs/20261009T041309638772Z_B01_all_features|B01 raw features]] | 0.748699 | — | Comparison baseline |
| [[Experiments/Runs/20261009T045958594430Z_F01_scores|F01 scores]] | 0.748759 | +0.000060 | Defer: no material gain |
| [[Experiments/Runs/20261009T050024803961Z_F02_affordability|F02 affordability]] | **0.757137** | **+0.008438** | Keep for confirmation |
| [[Experiments/Runs/20261009T050051427044Z_F03_missingness_tenure|F03 missingness + tenure]] | 0.749427 | +0.000728 | Small standalone gain |
| [[Experiments/Runs/20261009T050127647521Z_F04_accepted_combination|F04 affordability + missingness + tenure]] | 0.756631 | +0.007932 | Defer: 0.000506 below F02 |

B01 CatBoost scored **0.752544** local OOF AUC and **0.756** public AUC, rank **668**, as confirmed by you. The feature screens above are LightGBM-only comparisons; CatBoost with affordability has not yet been measured. Local scores are not public scores. F02's public score remains unknown, and 0.80–0.85 AUC is not currently demonstrated.

## Controlled comparison

All screens used the original verified input bytes, 136,961 development rows, the same five stratified folds, seed 42, and B01's LightGBM settings (2,000-tree cap, early stopping 150, four threads). The same 34,241 audit rows were held out and never evaluated. The raw predictors and employment-placeholder cleaning remain present.

| Fold | B01 LightGBM AUC | F02 LightGBM AUC |
| --- | --- | --- |
| 1 | 0.746790 | 0.756644 |
| 2 | 0.752795 | 0.760508 |
| 3 | 0.750914 | 0.758470 |
| 4 | 0.746212 | 0.753734 |
| 5 | 0.746745 | 0.756739 |

The 0.0005 threshold used to screen groups is a planning heuristic, not a significance test. Improvements can depend on seed and model family. The combined group was tested rather than assuming that standalone gains add up.

## Artifacts and next experiment

F02 run ID: `20261009T050024803961Z_F02_affordability`. It generated 61,500 format-checked probabilities and saved five models, fold predictions and its frozen configuration under `output/screening/runs/`. CSV SHA-256: `41fb2033764d3b8e36a63c0e1af9ca2d23ff23ef18243b060d4e304890edc47e`. The original B01 submission export is preserved.

Re-running all five F02 folds from the frozen configuration reproduced the original CSV bytes exactly. Its fold-assignment file also matches B01 byte for byte. The separate candidate package passed archive integrity and every included-file hash check. It contains a notebook configured for F02 and the same frozen reproduction configuration.

- [submission_F02.csv in Drive](https://drive.google.com/file/d/1x-DY1CpWPt4lBadoksqwhIXexL6TODh3/view), under `GCI_Global/Competition/output/`.
- [comp_F02.zip in Drive](https://drive.google.com/file/d/1LEvfOpsbYZt32mGrFW_y-gBpB6sbYH8x/view), under `GCI_Global/Competition/`; ZIP SHA-256 `b93afac8684f26f0ee63b071709dfcb41e95e5baac474d272050868dd24ad86a`.
- Local pair: `output/candidates/F02_affordability/`. No Omnicampus upload is automated.

After verifying Colab setup, use the F02 settings in [[02 Runbook]]. Next compare CatBoost with the same affordability group, then confirm the strongest candidate on a second fold seed. Tune only development rows and confirm tuned settings with five folds. Keep audit disabled until the planned finalist freeze.

Re-run the controlled screen locally with an explicit baseline manifest:

```bash
.venv/bin/python scripts/screen_features.py --input /Users/ryokitano/Downloads/Competition/input --baseline output/runs/20261009T041309638772Z_B01_all_features/manifest.json
```

Machine-readable results from the original screen are in `output/feature_screen.json`. New screen executions write their summary into the selected screening output folder. Generated run notes are tracked in Git; predictions and model files remain local.
