"""An audit comparison consumes frozen, hashed runs and cannot be rerun silently."""
from pathlib import Path
import json, sys
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import gci_pipeline as gci


def test_frozen_audit_rejects_unfrozen_or_changed_runs_and_scores_only_reserved_rows(tmp_path):
    from audit_frozen import evaluate_frozen
    inputs=tmp_path/'input'; inputs.mkdir()
    train=pd.DataFrame({'SK_ID_CURR':range(120),'TARGET':[0,0,0,1]*30,'DAYS_EMPLOYED':range(-120,0)})
    test=pd.DataFrame({'SK_ID_CURR':[200,201],'DAYS_EMPLOYED':[-10,-20]})
    sample=pd.DataFrame({'SK_ID_CURR':[201,200],'TARGET':[.5,.5]})
    for name,frame in [('train',train),('test',test),('sample_submission',sample)]: frame.to_csv(inputs/f'{name}.csv',index=False)
    cfg=gci.Config(models=('catboost',),n_splits=3,max_iterations=5,early_stopping=3,threads=1,verify_input_hashes=False)
    m=gci.run_experiment(inputs,tmp_path/'train',cfg)
    run=Path(m['run_dir'])
    frozen={'frozen_choices':False,'components':{'cb':{'run_dir':str(run),'model':'catboost','manifest_sha256':gci.sha256(run/'manifest.json')}},
            'candidates':{'single':{'cb':1.}}}
    path=tmp_path/'frozen.json'; path.write_text(json.dumps(frozen))
    with pytest.raises(ValueError,match='frozen'): evaluate_frozen(inputs,path,tmp_path/'audit')
    frozen['frozen_choices']=True
    frozen['components']['cb']['manifest_sha256']='wrong'
    path.write_text(json.dumps(frozen))
    with pytest.raises(ValueError,match='manifest'): evaluate_frozen(inputs,path,tmp_path/'audit')
    assert not (tmp_path/'audit/audit_started.json').exists()
    frozen['components']['cb']['manifest_sha256']=gci.sha256(run/'manifest.json')
    path.write_text(json.dumps(frozen))
    report=evaluate_frozen(inputs,path,tmp_path/'audit')
    assert report['selected_candidate']=='single'
    assert report['audit_rows']==24
    prediction=pd.read_csv(tmp_path/'audit/audit_predictions.csv')
    _,reserved=gci.development_split(train,cfg)
    assert prediction.SK_ID_CURR.tolist()==train.SK_ID_CURR.iloc[reserved].tolist()
    assert report['frozen_sha256']==gci.sha256(path)
    with pytest.raises(FileExistsError): evaluate_frozen(inputs,path,tmp_path/'audit')
