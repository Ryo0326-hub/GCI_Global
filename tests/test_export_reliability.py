"""An interrupted or corrupt promotion must preserve the previous candidate."""
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import gci_pipeline as gci


@pytest.mark.parametrize('failure', ['cancel', 'disk', 'corrupt'])
def test_export_failure_preserves_current_candidate(tmp_path, monkeypatch, failure):
    inputs = tmp_path / 'input'
    inputs.mkdir()
    train = pd.DataFrame({'SK_ID_CURR': range(120), 'TARGET': [0, 0, 0, 1] * 30,
                          'DAYS_EMPLOYED': range(-120, 0)})
    test = pd.DataFrame({'SK_ID_CURR': [200, 201], 'DAYS_EMPLOYED': [-2, -1]})
    sample = pd.DataFrame({'SK_ID_CURR': [201, 200], 'TARGET': [.5, .5]})
    for name, frame in [('train', train), ('test', test), ('sample_submission', sample)]:
        frame.to_csv(inputs / f'{name}.csv', index=False)
    output = tmp_path / 'output'
    cfg = gci.Config(models=('catboost',), n_splits=2, max_iterations=3,
                     early_stopping=2, threads=1, verify_input_hashes=False)
    gci.run_experiment(inputs, output, cfg)
    old_csv = (output / 'submission.csv').read_bytes()
    old_manifest = (output / 'latest_run.json').read_bytes()
    original_copy = gci.shutil.copy2

    def broken_export(source, destination, **kwargs):
        if Path(source).name == 'submission.csv' and Path(destination).parent == output:
            Path(destination).write_bytes(Path(source).read_bytes()[:16])
            if failure == 'cancel':
                raise KeyboardInterrupt('cancelled export')
            if failure == 'disk':
                raise OSError('interrupted Drive write')
            return str(destination)
        return original_copy(source, destination, **kwargs)

    monkeypatch.setattr(gci.shutil, 'copy2', broken_export)
    expected = {'cancel': KeyboardInterrupt, 'disk': OSError, 'corrupt': ValueError}[failure]
    with pytest.raises(expected):
        gci.run_experiment(inputs, output, cfg)
    assert (output / 'submission.csv').read_bytes() == old_csv
    assert (output / 'latest_run.json').read_bytes() == old_manifest
    assert not list(output.glob('.submission-*'))


@pytest.mark.parametrize('error', [OSError, KeyboardInterrupt])
def test_interrupted_manifest_write_preserves_previous_json(tmp_path, monkeypatch, error):
    target = tmp_path / 'latest_run.json'
    target.write_text('{"run_id": "previous"}\n')
    previous = target.read_bytes()
    original_write = Path.write_text

    def interrupted_write(path, text, *args, **kwargs):
        original_write(path, text[:3], *args, **kwargs)
        raise error('interrupted metadata write')

    monkeypatch.setattr(Path, 'write_text', interrupted_write)
    with pytest.raises(error):
        gci.save_json(target, {'run_id': 'next'})
    assert target.read_bytes() == previous
    assert set(tmp_path.iterdir()) == {target}
