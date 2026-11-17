"""Keep each ensemble member's validated features and training limits intact."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import gci_pipeline as gci


def test_model_options_do_not_override_shared_fold_or_audit_identity():
    with pytest.raises(ValueError,match='model_options'):
        gci.Config(model_options={'catboost':{'seed':7}})
    with pytest.raises(ValueError,match='feature'):
        gci.Config(model_options={'catboost':{'feature_groups':['invented_history']}})


def test_members_keep_separate_feature_groups_and_saved_inference(tmp_path):
    import joblib
    inputs=tmp_path/'input';inputs.mkdir()
    n=120
    frame=pd.DataFrame({'SK_ID_CURR':np.arange(n),'DAYS_EMPLOYED':-np.arange(n)-1,
       'AMT_INCOME_TOTAL':1000.+np.arange(n)*7,'AMT_CREDIT':10000.+np.arange(n)*23,
       'AMT_ANNUITY':100.+np.arange(n)*3,'AMT_GOODS_PRICE':9000.+np.arange(n)*17,
       'CNT_FAM_MEMBERS':2.+np.arange(n)%3})
    train=frame.assign(TARGET=np.arange(n)%4==0).astype({'TARGET':int})
    test=frame.iloc[:3].assign(SK_ID_CURR=[200,201,202])
    sample=pd.DataFrame({'SK_ID_CURR':[202,201,200],'TARGET':[.5]*3})
    for name,value in [('train',train),('test',test),('sample_submission',sample)]:value.to_csv(inputs/f'{name}.csv',index=False)
    cfg=gci.Config(models=('lightgbm','catboost'),n_splits=3,max_iterations=10,early_stopping=3,threads=1,verify_input_hashes=False,
        model_options={'lightgbm':{'feature_groups':['affordability'],'max_iterations':4},
                       'catboost':{'feature_groups':[],'max_iterations':3}},blend_weights={'lightgbm':.5,'catboost':.5})
    m=gci.run_experiment(inputs,tmp_path/'output',cfg)
    run=Path(m['run_dir'])
    lgb=joblib.load(run/'lightgbm_fold_0.joblib')
    cb=joblib.load(run/'catboost_fold_0.joblib')
    assert 'CREDIT_INCOME' in lgb['model'].feature_name_
    assert 'CREDIT_INCOME' not in cb['model'].feature_names_
    assert cb['model'].tree_count_<=3
    assert lgb['model'].n_estimators==4
    assert m['model_metrics']['lightgbm']['feature_count']==m['model_metrics']['catboost']['feature_count']+8
    assert gci.verify_saved_predictions(inputs,run)['byte_identical']
