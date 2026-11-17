"""Native categorical inference and explicit device selection for model diversity."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import gci_pipeline as gci


def test_xgboost_saved_inference_handles_unseen_categories(tmp_path):
    inputs = tmp_path / 'input'
    inputs.mkdir()
    train = pd.DataFrame({'SK_ID_CURR': range(120), 'TARGET': [0, 0, 0, 1] * 30,
                          'DAYS_EMPLOYED': range(-120, 0), 'NAME_CONTRACT_TYPE': ['A', 'B'] * 60})
    test = pd.DataFrame({'SK_ID_CURR': [201, 202], 'DAYS_EMPLOYED': [-10, -20],
                         'NAME_CONTRACT_TYPE': ['UNSEEN', None]})
    sample = pd.DataFrame({'SK_ID_CURR': [202, 201], 'TARGET': [.5, .5]})
    for name, frame in [('train', train), ('test', test), ('sample_submission', sample)]:
        frame.to_csv(inputs / f'{name}.csv', index=False)
    cfg = gci.Config(models=('xgboost',), n_splits=3, max_iterations=5, threads=1,
                     early_stopping=3, verify_input_hashes=False, model_params={'xgboost': {'device': 'cpu'}})
    manifest = gci.run_experiment(inputs, tmp_path / 'output', cfg)
    assert all(row['backend'] == 'CPU' for row in manifest['model_metrics']['xgboost']['folds'])
    result = gci.verify_saved_predictions(inputs, manifest['run_dir'])
    assert result['max_absolute_error'] == 0
    assert result['byte_identical']
    assert (Path(manifest['run_dir']) / 'xgboost_fold_0.ubj').is_file()


def test_xgboost_cannot_silently_fall_back_from_requested_cuda(tmp_path, monkeypatch):
    import xgboost
    class FallbackModel:
        def __init__(self, **kwargs): pass
        def fit(self, *args, **kwargs): return self
        def get_booster(self): return self
        def save_config(self): return json.dumps({'learner': {'generic_param': {'device': 'cpu'}}})
    monkeypatch.setattr(xgboost, 'XGBClassifier', FallbackModel)
    cfg = gci.Config(models=('xgboost',), n_splits=2, threads=1,
                     model_params={'xgboost': {'device': 'cuda:0'}})
    x = pd.DataFrame({'DAYS_EMPLOYED': range(40)})
    y = pd.Series(np.arange(40) % 2)
    with pytest.raises(RuntimeError, match='Requested'):
        gci.fit_cv(x, y, None, pd.Series(range(40)), cfg)
