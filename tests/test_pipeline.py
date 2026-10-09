from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gci_pipeline import Config, FeatureProcessor, make_submission, safe_ratio, validate_submission


def test_submission_aligns_by_id_and_rejects_missing_rows():
    sample = pd.DataFrame({"SK_ID_CURR": [10, 30, 20], "TARGET": [0.5] * 3})
    submission = make_submission(sample, [20, 10, 30], [0.2, 0.1, 0.3])
    assert submission.TARGET.tolist() == [0.1, 0.3, 0.2]
    with pytest.raises(ValueError):
        make_submission(sample, [10, 20], [0.1, 0.2])
    with pytest.raises(ValueError):
        make_submission(sample, [10, 10, 30], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        make_submission(sample, [10, 20, 30], [0.1, np.inf, 0.3])
    with pytest.raises(ValueError):
        validate_submission(submission.iloc[::-1].reset_index(drop=True), sample)


def test_ratio_handles_test_only_zero_and_missing_denominator():
    result = safe_ratio(pd.Series([1., 1., 1., 1.]), pd.Series([2., 0., np.nan, np.inf]))
    assert result.iloc[0] == 0.5
    assert result.iloc[1:].isna().all()


def test_peer_statistics_and_categories_fit_only_training_rows():
    training = pd.DataFrame({"OCCUPATION_TYPE": ["A"] * 20,
                             "NAME_EDUCATION_TYPE": ["B"] * 20,
                             "ORGANIZATION_TYPE": ["C"] * 20,
                             "AMT_INCOME_TOTAL": [100.] * 20})
    validation = training.iloc[:1].copy()
    validation["AMT_INCOME_TOTAL"] = 10_000_000.
    validation["OCCUPATION_TYPE"] = "UNSEEN"
    processor = FeatureProcessor(("peer", "categories"), "lightgbm").fit(training)
    result = processor.transform(validation)
    assert result.OCCUPATION_TYPE_PEER_INCOME.iloc[0] == 100
    assert pd.isna(result.OCCUPATION_TYPE.iloc[0])
    assert result.OCCUPATION_TYPE_FREQUENCY.iloc[0] == 0


def test_final_mode_requires_frozen_selection_and_never_evaluates_audit():
    with pytest.raises(ValueError):
        Config(mode="final")
    with pytest.raises(ValueError):
        Config(mode="final", frozen_choices=True, blend_weights={"lightgbm": 1.0}, evaluate_audit=True)
    with pytest.raises(ValueError):
        Config(blend_weights={"lightgbm": 0.3})
