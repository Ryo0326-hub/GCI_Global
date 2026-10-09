# Colab environment repair — October 9, 2026

[[00 Dashboard]] · [[02 Runbook]] · [[06 Decisions]]

The original pinned NumPy 2.2.6 stack passed fresh Python 3.12 and 3.13 checks. Keep those pins. Gemini's downgrade to NumPy 1.26.4 conflicts with packages requiring NumPy 2 and is unsupported on the Python 3.13 runtime in the traceback. NumPy's release notes list Python 3.9–3.12 for 1.26.4. [NumPy release notes](https://numpy.org/devdocs/release/1.26.4-notes.html)

## Observed failure and evidence

The saved Colab traceback entered `/usr/local/lib/python3.13/dist-packages`, then sklearn, SciPy's NumPy compatibility layer, `numpy.strings`, and `_override___module__`. Assigning `ufunc.__module__ = "numpy.strings"` raised an AttributeError. Only the installation cell had been changed to NumPy 1.26.4; other source cells matched the generated notebook. The failed Drive notebook is preserved locally in `qa/colab_before_fix.ipynb` outside Git.

Stale in-memory NumPy objects after an installation are a likely explanation of the original error. That is an inference: a fresh environment with NumPy 2.2.6 imports the same strings path successfully, so the traceback does not establish blanket incompatibility between NumPy 2 and these pinned SciPy/sklearn versions. NumPy documents old or inconsistent installations as an import-error cause. [NumPy troubleshooting](https://numpy.org/doc/stable/user/troubleshooting-importerror.html)

The original setup relied on Colab's restart prompt. It should have enforced a restart before importing the changed packages.

## Repair

The notebook writes a standard-library setup helper before importing numerical packages. It checks Python and installed versions, installs the exact requirements when needed, and tests imports and tiny model fits in a fresh subprocess. On the first setup or a changed stack, it writes a setup marker and terminates the notebook kernel once to clear cached imports. Colab reconnects; choosing Run all again continues with the tested packages. The marker includes pins, Python version and the old process ID so the fresh session does not restart repeatedly.

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

## Validation and remaining check

Fresh local Python **3.12.12** and **3.13.11** environments passed all six tests, notebook schema/source checks, NumPy string operations, SciPy sparse construction, sklearn AUC, and tiny LightGBM/CatBoost fits. Tests cover restart-once behavior, changed versions, stale imports and rejection of NumPy 1 on Python 3.13. GitHub checks run the same validations on Linux for both Python versions.

The updated notebook was saved to its existing Drive file ID with the same folder and Colab MIME type, and downloaded bytes matched the local file. This verifies the saved artifact; a complete live Colab training/export run is still pending.

Verified notebook SHA-256: `befa4a49836bea88ed890c336b3223cc079f3711f40690d24b973e56ec9a2e1f` (62,560 bytes; Drive file ID `1sAPtD9Kmf3DiXiyFaLinhfOHHTcuGYZR`).

1. Reload [comp.ipynb in Colab](https://colab.research.google.com/drive/1sAPtD9Kmf3DiXiyFaLinhfOHHTcuGYZR).
2. Choose Runtime → Run all. The setup restart is expected.
3. After reconnecting, choose Run all again. The setup prints `Environment ready. Training can continue.` before mounting Drive.
4. If the fresh-process check fails, choose Runtime → Disconnect and delete runtime and repeat these steps.

Do not replace the NumPy pin with 1.26.4. The supplied pip conflicts were caused by that downgrade.
