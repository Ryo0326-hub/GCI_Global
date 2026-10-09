# Colab environment repair — October 9, 2026

[[00 Dashboard]] · [[02 Runbook]] · [[06 Decisions]]

The reported session crash was caused by the setup's own forced restart. That code has been removed. Setup and notebook controls now use standard-library code, and each numerical step runs in a fresh Python interpreter with progress streamed back to the notebook. Reload the updated notebook and choose Runtime → Run all.

## Confirmed crash cause

The saved Drive notebook, modified at **1:33:36 p.m. Toronto time**, records this successful setup on **Python 3.13.16**:

> NumPy strings, SciPy sparse, sklearn, LightGBM and CatBoost tiny fits passed

It then prints the setup-restart message. The supplied log records `AsyncIOLoopKernelRestarter: restarting kernel (1/5)` at **1:33:25 p.m.** The embedded helper called `os.kill(os.getpid(), signal.SIGKILL)` immediately after successful validation. That ended the kernel and produced Colab's crash popup. Training cells had not executed.

The frozen-module debugger warnings and websocket timeout adjustment are startup messages. They are not evidence of a model exception or exhausted RAM in this incident. The log and saved notebook jointly identify the intentional kill as the cause.

The exact saved notebook and supplied log are preserved outside Git in `qa/colab_crash_before.ipynb` and `qa/colab_crash_log.csv`.

## Earlier NumPy failure

Keep NumPy **2.2.6**. Gemini's downgrade to 1.26.4 conflicts with installed packages requiring NumPy 2 and is unsupported on Python 3.13. NumPy's release notes list Python 3.9–3.12 for 1.26.4. [NumPy release notes](https://numpy.org/devdocs/release/1.26.4-notes.html)

The original traceback entered sklearn, SciPy's NumPy compatibility layer and `numpy.strings`, where assigning a ufunc module attribute failed. Stale in-memory objects after installation remain a likely explanation; this is an inference rather than a conclusively established cause. Fresh environments with the pinned stack, including the actual Colab setup above, passed that import path. [NumPy troubleshooting](https://numpy.org/doc/stable/user/troubleshooting-importerror.html)

## Current execution design

Setup installs the exact pins when needed and tests them in a fresh subprocess. It returns a verified package summary without importing numerical packages into the notebook kernel or stopping that kernel. Input inspection, Optuna, cross-validation, blend diagnostics and ZIP creation each start a fresh interpreter using the embedded pipeline and the verified pins. Results return as JSON; fold progress and exceptions remain visible. Cancellation terminates the active worker rather than leaving training running.

The training pipeline, features, splits, model settings, CSV contract and Drive destinations remain those of the measured experiments. The notebook configuration is a plain dictionary which is converted to the pipeline's Config inside each worker. Earlier fitted runs and their recorded CSV hashes retain their original meaning.

| Package | Pin |
| --- | --- |
| NumPy | 2.2.6 |
| pandas | 2.2.3 |
| SciPy | 1.15.3 |
| scikit-learn | 1.6.1 |
| LightGBM | 4.6.0 |
| CatBoost | 1.2.8 |
| Optuna | 4.4.0 |
| joblib | 1.5.1 |

## Validation and next action

Fresh local Python **3.12.12** and **3.13.11** environments passed eight tests. The new regression checks deliberately place stale modules in the parent process, verify that workers import the on-disk modules, ensure the parent survives setup, and check worker errors and temporary-file cleanup. Notebook validation checks all five worker snippets, source hashes, dependency pins and the absence of numerical imports in the notebook kernel.

A bounded local test executes the generated notebook controls, verifies the real competition input hashes, then uses the existing 3,000-row development-only fixture for one Optuna trial, both model families, blend diagnostics, and matching CSV/code ZIP/report export. Its scores are test results, not competition evidence. This excludes live Drive authorization and Colab UI interaction; a complete hosted training/export run still needs confirmation.

The fixed notebook was saved to the same Drive ID and folder. Its downloaded bytes match the local notebook: SHA-256 `a354e25a19a3ad9ba4f35ed46cf58c3760a1e23fefda271c3575c73316d1c562` (64,422 bytes). [GitHub checks passed on Python 3.12 and 3.13](https://github.com/Ryo0326-hub/GCI_Global/actions/runs/37968625620). The F02 code ZIP was refreshed with the same execution design, while retaining its original matching CSV and frozen configuration.

1. Reload [comp.ipynb in Colab](https://colab.research.google.com/drive/1sAPtD9Kmf3DiXiyFaLinhfOHHTcuGYZR).
2. Choose Runtime → Run all and authorize your Drive mount.
3. Setup should print `Environment ready. Run all continues; model steps use fresh Python processes.` and continue.
4. Save with Cmd+S / Ctrl+S before the code ZIP cell.

Use the current session after the earlier restart. Delete the runtime only if a fresh-process check explicitly fails. Do not downgrade NumPy.
