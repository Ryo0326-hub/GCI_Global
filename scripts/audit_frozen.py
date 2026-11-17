"""Evaluate at most three pre-frozen candidates once, using immutable saved folds.

Each component runs in a fresh process with its own executed source snapshot.
No training, early stopping, feature selection or weight search occurs here.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import subprocess
import sys


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): value.update(block)
    return value.hexdigest()


def validate_frozen(path):
    frozen=json.loads(Path(path).read_text())
    if frozen.get('frozen_choices') is not True:
        raise ValueError('Audit requires explicitly frozen choices')
    components=frozen.get('components',{})
    candidates=frozen.get('candidates',{})
    if not components or not 1<=len(candidates)<=3:
        raise ValueError('Freeze components and one to three candidates before audit')
    for name,component in components.items():
        if not name.replace('_','').isalnum(): raise ValueError('Invalid component name')
        run=Path(component['run_dir'])
        if digest(run/'manifest.json')!=component['manifest_sha256']:
            raise ValueError(f'Frozen manifest changed: {name}')
        m=json.loads((run/'manifest.json').read_text())
        if m['config']['mode']!='development' or m['config']['evaluate_audit']:
            raise ValueError('Components must be development fits with uninspected audit')
        if component['model'] not in m['config']['models']:
            raise ValueError('Component model was not fitted')
        if digest(run/'gci_pipeline.py')!=m['pipeline_sha256']:
            raise ValueError('Executed source snapshot changed')
        for filename,expected in m['artifact_hashes'].items():
            if digest(run/filename)!=expected: raise ValueError(f'Component artifact changed: {filename}')
    for weights in candidates.values():
        if not weights or set(weights)-set(components): raise ValueError('Unknown component in candidate')
        if any(not math.isfinite(v) or v<0 for v in weights.values()) or not math.isclose(sum(weights.values()),1.):
            raise ValueError('Frozen weights must be finite, nonnegative and sum to one')
    return frozen


def worker(input_dir,run_dir,model,destination):
    # Run only trusted, hash-checked project snapshots, never downloaded reference code.
    if sys.platform=='darwin':
        import ctypes
        library=Path('/opt/anaconda3/lib/libomp.dylib')
        if library.is_file(): ctypes.CDLL(str(library))
    sys.path.insert(0,str(Path(run_dir).resolve()))
    import gci_pipeline as gci
    import joblib
    import numpy as np
    import pandas as pd
    from importlib.metadata import version
    run=Path(run_dir)
    m=json.loads((run/'manifest.json').read_text())
    cfg=gci.Config(**m['config'])
    if any(version(name)!=value for name,value in m['package_versions'].items()):
        raise ValueError('Use the recorded package versions for audit inference')
    train,_,_,hashes=gci.load_inputs(input_dir,cfg.verify_input_hashes)
    if hashes!=m['input_hashes']: raise ValueError('Input bytes differ from the frozen run')
    dev,audit=gci.development_split(train,cfg)
    oof=pd.read_csv(run/'oof.csv')
    if oof.SK_ID_CURR.duplicated().any() or set(oof.SK_ID_CURR)!=set(train.SK_ID_CURR.iloc[dev]):
        raise ValueError('Saved OOF rows do not match the reserved development split')
    assignment=pd.read_csv(run/'fold_assignments.csv').set_index('SK_ID_CURR')
    if not assignment.loc[train.SK_ID_CURR.iloc[audit],'partition'].eq('audit').all():
        raise ValueError('A reserved audit row was included in fitting')
    groups=getattr(cfg,'model_options',{}).get(model,{}).get('feature_groups',cfg.feature_groups)
    x=gci.make_features(train.iloc[audit],groups)
    prediction=np.zeros(len(audit))
    for fold in range(cfg.n_splits):
        saved=joblib.load(run/f'{model}_fold_{fold}.joblib')
        prediction+=saved['model'].predict_proba(saved['processor'].transform(x))[:,1]/cfg.n_splits
    if not np.isfinite(prediction).all() or ((prediction<0)|(prediction>1)).any():
        raise ValueError('Invalid audit probabilities')
    pd.DataFrame({'SK_ID_CURR':train.SK_ID_CURR.iloc[audit].to_numpy(),'prediction':prediction}).to_csv(destination,index=False)


def evaluate_frozen(input_dir,frozen_path,output_dir):
    frozen_path=Path(frozen_path).resolve()
    frozen=validate_frozen(frozen_path)
    output=Path(output_dir).resolve();output.mkdir(parents=True,exist_ok=True)
    freeze_hash=digest(frozen_path)
    with (output/'audit_started.json').open('x') as stream:
        json.dump({'frozen_sha256':freeze_hash,'started_utc':datetime.now(timezone.utc).isoformat()},stream,indent=2)
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    frames=[]
    for name,component in frozen['components'].items():
        path=output/f'{name}_predictions.csv'
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker','--input',str(Path(input_dir).resolve()),
                        '--run',str(Path(component['run_dir']).resolve()),'--model',component['model'],'--destination',str(path)],check=True)
        frame=pd.read_csv(path).set_index('SK_ID_CURR')
        if frames and not frame.index.equals(frames[0].index):
            raise ValueError('Frozen components use different audit IDs or order')
        frames.append(frame.rename(columns={'prediction':name}))
    predictions=pd.concat(frames,axis=1)
    labels=pd.read_csv(Path(input_dir)/'train.csv',usecols=['SK_ID_CURR','TARGET']).set_index('SK_ID_CURR').loc[predictions.index,'TARGET']
    scores={}
    for name,weights in frozen['candidates'].items():
        candidate=sum(weight*predictions[component] for component,weight in weights.items())
        scores[name]=float(roc_auc_score(labels,candidate))
    if digest(frozen_path)!=freeze_hash: raise ValueError('Frozen choices changed during audit')
    predictions.reset_index().to_csv(output/'audit_predictions.csv',index=False)
    selected=max(scores,key=scores.get)
    report={'frozen_sha256':freeze_hash,'audit_rows':len(predictions),'candidate_auc':scores,
            'selected_candidate':selected,'selected_weights':frozen['candidates'][selected],
            'limitation':'One held-out comparison of preselected candidates; no performance guarantee on the competition test set.'}
    (output/'audit_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True)
    parser.add_argument('--frozen');parser.add_argument('--output')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--run');parser.add_argument('--model');parser.add_argument('--destination')
    args=parser.parse_args()
    if args.worker: worker(args.input,args.run,args.model,args.destination)
    else: print(json.dumps(evaluate_frozen(args.input,args.frozen,args.output),indent=2))

if __name__=='__main__': main()
