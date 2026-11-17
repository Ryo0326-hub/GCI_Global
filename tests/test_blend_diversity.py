"""A third model must be considered in the same bounded OOF blend diagnostic."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import gci_pipeline as gci


def test_coarse_blend_supports_three_models_and_two_model_subsets(tmp_path):
    rng = np.random.default_rng(42)
    target = np.arange(200) % 2
    frame = pd.DataFrame({'SK_ID_CURR': np.arange(200), 'TARGET': target,
                          'lightgbm': np.clip(.3 + .2 * target + rng.normal(0,.2,200),.01,.99),
                          'catboost': np.clip(.3 + .2 * target + rng.normal(0,.2,200),.01,.99),
                          'xgboost': .2 + .6 * target})
    for models in [('lightgbm','catboost','xgboost'), ('catboost','xgboost')]:
        frame[['SK_ID_CURR','TARGET',*models]].to_csv(tmp_path/'oof.csv',index=False)
        weights, table = gci.select_blend(tmp_path)
        assert set(weights) == set(models)
        assert abs(sum(weights.values()) - 1) < 1e-10
        assert weights['xgboost'] > 0
        assert len(table) <= 15
        assert table.selection_auc.max() == 1.
