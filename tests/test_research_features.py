"""Edge cases and fold isolation for the public-solution-inspired candidates."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gci_pipeline import Config, FeatureProcessor, make_features


def applicants(n=40):
    return pd.DataFrame({
        "SK_ID_CURR": np.arange(n), "TARGET": np.arange(n) % 2,
        "DAYS_EMPLOYED": [-1000.] * n, "DAYS_BIRTH": [-40 * 365.25] * n,
        "AMT_CREDIT": [1200.] * n, "AMT_ANNUITY": [100.] * n,
        "AMT_GOODS_PRICE": [1000.] * n, "AMT_INCOME_TOTAL": [600.] * n,
        "CNT_FAM_MEMBERS": [3.] * n, "CNT_CHILDREN": [1.] * n,
        "NAME_EDUCATION_TYPE": ["Secondary"] * n,
        "OCCUPATION_TYPE": ["Office"] * n,
        "NAME_INCOME_TYPE": ["Working"] * n,
        "EXT_SOURCE_1": [.2] * n, "EXT_SOURCE_2": [.4] * n,
        "EXT_SOURCE_3": [.6] * n,
    })


def test_loan_terms_are_rowwise_and_invalid_denominators_stay_missing():
    frame = applicants(4)
    frame.loc[1, "CNT_FAM_MEMBERS"] = 1  # zero adults, not infinite income
    frame.loc[2, "AMT_ANNUITY"] = 0
    frame.loc[3, "AMT_INCOME_TOTAL"] = np.nan
    original = frame.copy(deep=True)
    Config(feature_groups=("loan_terms",))
    result = make_features(frame, ("loan_terms",))
    assert len(result) == len(frame)
    assert "SK_ID_CURR" not in result and "TARGET" not in result
    assert result.INCOME_PER_ADULT.iloc[0] == 300
    assert pd.isna(result.INCOME_PER_ADULT.iloc[1])
    assert pd.isna(result.GOODS_ANNUITY.iloc[2])
    assert result.GOODS_INCOME.iloc[0] == pytest.approx(1000 / 600)
    assert result.FINANCING_INCOME.iloc[0] == pytest.approx(200 / 600)
    assert result.AGE_AT_REPAYMENT_PROXY.iloc[0] > 40
    assert not np.isinf(result.select_dtypes("number").to_numpy()).any()
    pd.testing.assert_frame_equal(frame, original)
    # The presence or labels of other rows cannot change a row-wise feature.
    changed = pd.concat([frame.iloc[:1], applicants(1).assign(TARGET=9, AMT_CREDIT=1e12)])
    pd.testing.assert_series_equal(make_features(changed, ("loan_terms",)).iloc[0], result.iloc[0])


def test_peer_profiles_fit_only_training_and_handle_unseen_groups():
    Config(feature_groups=("peer_profile",))
    training = applicants()
    x = make_features(training, ("peer_profile",))
    processor = FeatureProcessor(("peer_profile",), "lightgbm").fit(x)
    validation = applicants(2).drop(columns="TARGET")
    validation["AMT_INCOME_TOTAL"] = [900., 10_000_000.]
    validation.loc[1, "OCCUPATION_TYPE"] = "UNSEEN"
    z = processor.transform(make_features(validation, ("peer_profile",)))
    assert z.PROFILE_0_AMT_INCOME_TOTAL_MEDIAN.tolist() == [600., 600.]
    assert z.PROFILE_0_AMT_INCOME_TOTAL_RELATIVE.iloc[0] == 1.5
    assert z.PROFILE_0_COUNT.tolist() == [40., 0.]
    assert z.PROFILE_1_CREDIT_ANNUITY_MEDIAN.tolist() == [12., 12.]
    assert z.PROFILE_0_EXT_PROFILE_MEAN_MEDIAN.iloc[0] == pytest.approx(.4)
    # Transforming a new batch cannot mutate training statistics.
    z2 = processor.transform(make_features(validation.iloc[:1], ("peer_profile",)))
    pd.testing.assert_series_equal(z2.iloc[0], z.iloc[0])
    assert not np.isinf(z.select_dtypes("number").to_numpy()).any()
