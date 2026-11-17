"""Derived scenarios are explicit, rowwise, and never fabricate observed histories."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from gci_pipeline import Config, FeatureProcessor, make_features


def applicants(n=40):
    return pd.DataFrame({
        'SK_ID_CURR': range(n), 'TARGET': np.arange(n) % 2,
        'DAYS_EMPLOYED': [-1000.] * n, 'DAYS_BIRTH': [-40 * 365.25] * n,
        'AMT_CREDIT': [1000.] * n, 'AMT_ANNUITY': [100.] * n,
        'NAME_EDUCATION_TYPE': ['Secondary'] * n, 'OCCUPATION_TYPE': ['Office'] * n,
        'NAME_INCOME_TYPE': ['Working'] * n, 'ORGANIZATION_TYPE': ['Business'] * n,
        'NAME_CONTRACT_TYPE': ['Cash loans'] * n, 'NAME_HOUSING_TYPE': ['House'] * n,
        'NAME_FAMILY_STATUS': ['Married'] * n, 'REGION_RATING_CLIENT': [2] * n,
        'AMT_REQ_CREDIT_BUREAU_HOUR': [0.] * n, 'AMT_REQ_CREDIT_BUREAU_MON': [2.] * n,
        'AMT_REQ_CREDIT_BUREAU_QRT': [3.] * n, 'AMT_REQ_CREDIT_BUREAU_YEAR': [12.] * n,
    })


def test_category_crosses_keep_fold_local_counts_and_unknown_categories():
    Config(feature_groups=('category_crosses',))
    x = make_features(applicants(), ('category_crosses',))
    assert 'CROSS_HOUSING_FAMILY' in x and 'CROSS_OCCUPATION_ORGANIZATION' in x
    processor = FeatureProcessor(('category_crosses',), 'lightgbm').fit(x)
    validation = applicants(2)
    validation.loc[1, 'NAME_HOUSING_TYPE'] = 'UNSEEN'
    z = processor.transform(make_features(validation, ('category_crosses',)))
    assert z.CROSS_HOUSING_FAMILY_FREQUENCY.tolist() == [1., 0.]
    assert pd.isna(z.CROSS_HOUSING_FAMILY.iloc[1])
    one = processor.transform(make_features(validation.iloc[:1], ('category_crosses',)))
    pd.testing.assert_series_equal(one.iloc[0], z.iloc[0])
    assert 'TARGET' not in z and 'SK_ID_CURR' not in z


def test_enquiry_patterns_keep_missingness_and_handle_zero_activity():
    Config(feature_groups=('inquiry_patterns',))
    frame = applicants(3)
    frame.loc[1, ['AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_YEAR']] = 0.
    frame.loc[2, 'AMT_REQ_CREDIT_BUREAU_MON'] = np.nan
    original = frame.copy(deep=True)
    result = make_features(frame, ('inquiry_patterns',))
    assert result.ENQUIRY_MONTH_TO_ANNUAL_RATE.iloc[0] == pytest.approx(1.)
    assert result.ENQUIRY_MONTH_TO_ANNUAL_RATE.iloc[1] == 0.
    assert pd.isna(result.ENQUIRY_MONTH_TO_ANNUAL_RATE.iloc[2])
    assert not np.isinf(result.select_dtypes('number').to_numpy()).any()
    pd.testing.assert_frame_equal(frame, original)


def test_pricing_hypotheses_recover_assumed_rate_without_claiming_true_terms():
    Config(feature_groups=('pricing_proxies',))
    frame = applicants(3)
    frame.loc[0, 'AMT_ANNUITY'] = 1000 * .01 / (1 - 1.01 ** -24)
    frame.loc[1, 'AMT_CREDIT'] = 0.
    frame.loc[2, 'AMT_ANNUITY'] = np.nan
    result = make_features(frame, ('pricing_proxies',))
    assert result.RATE_HYPOTHESIS_24M.iloc[0] == pytest.approx(.12, abs=1e-7)
    assert pd.isna(result.RATE_HYPOTHESIS_12M.iloc[0])
    assert result.RATE_HYPOTHESIS_24M.iloc[1:].isna().all()
    assert not np.isinf(result.select_dtypes('number').to_numpy()).any()
    changed = frame.iloc[:1].assign(TARGET=999)
    pd.testing.assert_series_equal(make_features(changed, ('pricing_proxies',)).iloc[0], result.iloc[0])
