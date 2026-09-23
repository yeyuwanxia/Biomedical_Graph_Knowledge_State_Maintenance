"""Small regression tests for the new launcher and statistical summaries; no real data."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dry_run_needs_no_data_and_writes_nothing():
    before = {str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file()}
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/reproduce.py'), '--protocol', 'paper', '--devices', 'cuda:0', '--dry-run'], check=True, capture_output=True, text=True)
    plan = json.loads(result.stdout)
    assert plan['stages'] == ['prepare', 'formal', 'report', 'analysis']
    assert plan['training_started'] is False
    assert {str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file()} == before


def test_resume_refuses_changed_protocol_or_inputs(tmp_path, monkeypatch):
    module = load('reproduce')
    monkeypatch.setattr(module, 'OUT', tmp_path)
    monkeypatch.setattr(module, 'verify_frozen', lambda: {'runtime.py': 'source-hash'})
    monkeypatch.setattr(module, 'input_hashes', lambda: {'graph': 'input-one'})
    module.check_context('paper', create=True)
    module.check_context('paper')
    with pytest.raises(RuntimeError, match='protocol'):
        module.check_context('redevelop')
    monkeypatch.setattr(module, 'input_hashes', lambda: {'graph': 'input-two'})
    with pytest.raises(RuntimeError, match='input_hashes'):
        module.check_context('paper')


def test_split_average_gives_equal_weight_to_bases():
    module = load('summarize_paper')
    data = pd.DataFrame([dict(split=1, base_seed=1, request_seed=q, value=0.) for q in [1, 2, 3]] + [dict(split=1, base_seed=2, request_seed=4, value=10.)])
    split = module.hierarchical_split_means(data, ['value'])
    assert split.iloc[0]['value'] == 5.  # Averaging all four requests would incorrectly give 2.5.


def test_summary_rejects_pseudoreplicated_rows():
    module = load('summarize_paper')
    rows = pd.DataFrame([dict(method='FT', split=1, quality=.5), dict(method='FT', split=1, quality=.7)])
    with pytest.raises(ValueError, match='one aggregated row per split'):
        module.summarize_splits(rows, ['method'])


def test_resume_compares_serialized_environment_and_rejects_change(tmp_path, monkeypatch):
    module = load('reproduce')
    monkeypatch.setattr(module, 'OUT', tmp_path)
    monkeypatch.setattr(module, 'capture_environment', lambda torch, devices: {
        'adam_effective_defaults': {'betas': (.9, .999)}, 'devices': devices})
    module.check_environment(None, ['cuda:0'], create=True)
    module.check_environment(None, ['cuda:0'])
    with pytest.raises(RuntimeError, match='GPU schedule changed'):
        module.check_environment(None, ['cuda:0', 'cuda:1'])
