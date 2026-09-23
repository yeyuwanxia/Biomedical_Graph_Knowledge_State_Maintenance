"""Small synthetic checks for pairing and the reference diagnostic hierarchy."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('paper_statistics', Path(__file__).resolve().parents[1] / 'scripts/summarize_paper.py')
paper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paper)


def test_pairs_align_split_ids_before_subtraction():
    left = pd.Series([11., 22., 33., 44., 55.], index=[1, 2, 3, 4, 5])
    right = pd.Series([50., 30., 10., 40., 20.], index=[5, 3, 1, 4, 2])
    values = paper.paired_split_difference(left, right)
    np.testing.assert_allclose(values, [1, 2, 3, 4, 5])
    summary = paper.confidence(values)
    assert summary['mean'] == 3 and summary['n_splits'] == 5
    assert summary['positive'] == 5 and summary['negative'] == 0
    assert summary['ci95_lower'] == pytest.approx(1.03675684)


@pytest.mark.parametrize('index,values', [([1, 2, 3, 4], [0]*4), ([1, 2, 3, 4, 4], [0]*5), ([1, 2, 3, 4, 5], [0, 0, np.nan, 0, 0])])
def test_pairs_reject_incomplete_duplicate_or_missing_splits(index, values):
    with pytest.raises(ValueError):
        paper.paired_split_difference(pd.Series(range(5), index=range(1, 6)), pd.Series(values, index=index))


def test_withdrawal_interaction_rebuilt_and_checked():
    rows, official = [], []
    for task in ['node', 'relation']:
        for split in range(1, 6):
            for method, mode, budget, gain in [('Original', 'selected', 0, 0), ('Input-only', 'selected', 0, .01), ('Continue-FT', 'fixed', 150, .02), ('Fixed-FT', 'fixed', 150, .04)]:
                rows.append(dict(task=task, split=split, method=method, mode=mode, budget=budget, quality=.1*split+gain))
            official.append(dict(task=task, split=split, no_withdraw_update_gain=.02, withdrawn_update_gain=.03, interaction=.01))
    metrics, exported = pd.DataFrame(rows), pd.DataFrame(official)
    split, summary = paper.withdrawal_interactions(metrics, exported)
    np.testing.assert_allclose(split.interaction, .01)
    assert len(summary) == 8
    exported.loc[0, 'interaction'] = .2
    with pytest.raises(ValueError, match='disagrees'):
        paper.withdrawal_interactions(metrics, exported)


def test_retrain_repeats_average_before_paired_comparison():
    rows = []
    for task in ['node', 'relation']:
        for split in range(1, 6):
            common = dict(task=task, split=split, base_seed=100+split, request_seed=1000+split)
            for i in range(3):
                value = split + 2*i
                rows.append(common | dict(comparison='retrain-retrain', method=f'pair{i}', js=value, tv=value/10, disagreement=value/100))
            for method, repeats in [('Tuned-FT', 1), ('Tuned-Full-old', 3)]:
                for repeat in range(repeats):
                    for retrain_index in range(3):
                        value = split + repeat + retrain_index
                        rows.append(common | dict(comparison='method-retrain', method=method, repeat=repeat, retrain_index=retrain_index, js=value, tv=value/10, disagreement=value/100))
    split, intervals, pairs, pair_intervals = paper.retrain_reference_summaries(pd.DataFrame(rows))
    np.testing.assert_allclose(pairs[pairs.metric == 'js'].difference, 1)
    rr = intervals[(intervals.comparison == 'retrain-retrain') & (intervals.metric == 'js')]
    np.testing.assert_allclose(rr['mean'], 5)
    assert len(split) == 30 and len(pair_intervals) == 6
