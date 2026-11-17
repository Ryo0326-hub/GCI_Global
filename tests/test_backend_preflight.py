"""All requested accelerators must be checked before any full fold starts."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import gci_pipeline as gci


@pytest.mark.parametrize('name, backend', [
    ('xgboost', {'device': 'cuda:0'}),
    ('catboost', {'task_type': 'GPU', 'devices': '0'}),
    ('lightgbm', {'device': 'gpu'}),
])
def test_unavailable_accelerator_stops_before_any_full_fold(monkeypatch, name, backend):
    import catboost
    import lightgbm
    import xgboost
    probe_sizes = []

    class FallsBackToCPU:
        def __init__(self, **params):
            self.params = params
            self.booster_ = SimpleNamespace(params={'device_type': 'cpu'})

        def fit(self, x, y, **kwargs):
            # The availability check must remain bounded independently of the run cap.
            probe_sizes.append((len(x), self.params.get('n_estimators', self.params.get('iterations'))))
            return self

        def get_booster(self):
            return self

        def save_config(self):
            return json.dumps({'learner': {'generic_param': {'device': 'cpu'}}})

        def get_all_params(self):
            return {'task_type': 'CPU'}

    monkeypatch.setattr(xgboost, 'XGBClassifier', FallsBackToCPU)
    monkeypatch.setattr(catboost, 'CatBoostClassifier', FallsBackToCPU)
    monkeypatch.setattr(lightgbm, 'LGBMClassifier', FallsBackToCPU)

    def unexpected_fold(*args, **kwargs):
        raise AssertionError('full-fold preprocessing started before accelerator check')

    monkeypatch.setattr(gci.FeatureProcessor, 'fit', unexpected_fold)
    # Put a CPU model first: even that work must wait for every accelerator check.
    first = 'lightgbm' if name != 'lightgbm' else 'catboost'
    cfg = gci.Config(models=(first, name), n_splits=2, max_iterations=4000,
                     model_params={name: backend})
    x = pd.DataFrame({'DAYS_EMPLOYED': range(120)})
    y = pd.Series(np.arange(120) % 2)
    with pytest.raises(RuntimeError, match='Requested'):
        gci.fit_cv(x, y, None, pd.Series(range(120)), cfg)
    assert len(probe_sizes) == 1
    assert 0 < probe_sizes[0][0] <= 32 and 0 < probe_sizes[0][1] <= 2
