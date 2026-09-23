#!/usr/bin/env python3
"""Derive paper-facing statistics from a newly completed frozen report, never from bundled results."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reproduction'


def confidence(values):
    import numpy as np
    from scipy.stats import t
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    n = len(a)
    mean = float(a.mean()) if n else float('nan')
    sd = float(a.std(ddof=1)) if n > 1 else float('nan')
    half = float(t.ppf(.975, n - 1) * sd / np.sqrt(n)) if n > 1 else float('nan')
    return dict(n_splits=n, mean=mean, sd=sd, ci95_lower=mean-half, ci95_upper=mean+half,
                positive=int((a > 0).sum()), negative=int((a < 0).sum()), ties=int((a == 0).sum()))


def summarize_splits(frame, keys):
    """Input must already have one row per comparison and split."""
    import pandas as pd
    if frame.duplicated(keys + ['split']).any():
        raise ValueError('Expected one aggregated row per split, not individual requests or reference repeats.')
    numeric = [c for c in frame.select_dtypes(include='number') if c not in keys + ['split', 'base_seed', 'request_seed', 'repeat']]
    records = []
    for identity, group in frame.groupby(keys, dropna=False, sort=True):
        if not isinstance(identity, tuple):
            identity = (identity,)
        for metric in numeric:
            records.append(dict(zip(keys, identity)) | {'metric': metric} | confidence(group[metric]))
    return pd.DataFrame(records)


def transition_pairs(frame, protocol):
    """Pair within split before estimating uncertainty; positive protection means fewer original correct predictions lost."""
    import pandas as pd
    mode, ft, full = ('selected', 'Tuned-FT', 'Tuned-Full-old') if protocol == 'tuned' else ('fixed', 'Fixed-FT', 'Fixed-Full-old-a1')
    f = frame[(frame.task == 'node') & (frame.budget == 150) & (frame['mode'] == mode) & (frame.class_id == -1)]
    a, b = [f[f.method == method].set_index('split').sort_index() for method in [ft, full]]
    if a.index.has_duplicates or b.index.has_duplicates or set(a.index) != set(b.index) or len(a) != 5:
        raise ValueError(f'{protocol}: expected the same five unique formal splits for both methods')
    b = b.reindex(a.index)
    pairs = pd.DataFrame({'split': a.index,
                          'correction_difference': (b.originally_wrong_corrected - a.originally_wrong_corrected).to_numpy(),
                          'protection_benefit': (a.originally_correct_lost - b.originally_correct_lost).to_numpy()})
    pairs['net_correct_difference'] = pairs.correction_difference + pairs.protection_benefit
    pairs['protocol'] = protocol
    summaries = [dict(protocol=protocol, metric=c) | confidence(pairs[c])
                 for c in ['correction_difference', 'protection_benefit', 'net_correct_difference']]
    return pairs, summaries


def hierarchical_split_means(frame, metrics):
    request = frame.groupby(['split', 'base_seed', 'request_seed'])[metrics].mean().reset_index()
    base = request.groupby(['split', 'base_seed'])[metrics].mean().reset_index()
    return base.groupby('split')[metrics].mean().reset_index()


def relation_withdrawal():
    import numpy as np
    import pandas as pd
    records = []
    for directory in sorted((OUT / 'formal/requests').glob('relation_s*_b*/q*')):
        request = json.loads((directory / 'request.json').read_text())
        plans = json.loads((directory / 'planned_runs.json').read_text())
        matches = [p for p in plans if p['label'] == 'Fixed-FT']
        if len(matches) != 1:
            raise ValueError(f'Missing unique Fixed-FT plan: {directory}')
        spec = matches[0]['spec']
        identity = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:12]
        key = directory.parent.name
        predfile = OUT / 'formal/runs' / key / directory.name / (spec['family'] + '_' + identity) / '150_fixed.npz'
        with np.load(directory / 'context.npz') as z:
            initial, original = z['input_withdrawn'], z['original_withdrawn']
        with np.load(predfile) as z:
            updated = z['withdrawn_probabilities']
        if original.shape != initial.shape or initial.shape != updated.shape or not initial.size:
            raise ValueError('Withdrawn relation arrays do not align')
        delta = updated - initial
        if not np.isfinite(delta).all():
            raise ValueError('Nonfinite withdrawn relation probabilities')
        records.append(dict(split=int(key.split('_s')[1].split('_b')[0]), base_seed=request['base_seed'],
                            request_seed=request['seed'], original_score=float(original.mean()),
                            input_score=float(initial.mean()), updated_score=float(updated.mean()),
                            signed_post_input_change=float(delta.mean()),
                            absolute_post_input_change=float(np.abs(delta).mean()),
                            signed_input_change=float((initial-original).mean()),
                            signed_total_change=float((updated-original).mean())))
    if not records:
        raise FileNotFoundError('No formal relation request artifacts; finish training and report first.')
    frame = pd.DataFrame(records)
    metrics = ['original_score', 'input_score', 'updated_score', 'signed_post_input_change',
               'absolute_post_input_change', 'signed_input_change', 'signed_total_change']
    split = hierarchical_split_means(frame, metrics)
    if len(split) != 5:
        raise ValueError('Expected five complete relation splits')
    summary = pd.DataFrame([dict(metric=c) | confidence(split[c]) for c in metrics])
    return frame, split, summary



def paired_split_difference(left, right, label='contrast'):
    """Subtract matched split estimates, rejecting incomplete or duplicated pairs."""
    import numpy as np
    if left.index.has_duplicates or right.index.has_duplicates or set(left.index) != set(right.index) or len(left) != 5:
        raise ValueError(f'{label}: expected the same five unique formal splits')
    left = left.sort_index()
    right = right.reindex(left.index)
    if not np.isfinite(left.to_numpy(dtype=float)).all() or not np.isfinite(right.to_numpy(dtype=float)).all():
        raise ValueError(f'{label}: missing or nonfinite paired estimate')
    return left - right


def metric_series(frame, task, method, metric, budget=150, mode='selected'):
    rows = frame[(frame.task == task) & (frame.method == method) &
                 (frame.budget == budget) & (frame['mode'] == mode)]
    return rows.set_index('split')[metric]


def paper_paired_contrasts(frame):
    """Prespecified comparisons appearing in the manuscript's paired tables."""
    import pandas as pd
    plans = []
    for task in ['node', 'relation']:
        for method, baseline in [('Tuned-Full-old', 'Tuned-FT'), ('Tuned-FT', 'Input-only')]:
            for metric in ['quality', 'retrain_js']:
                plans.append((task, method, baseline, metric, 'selected', 0 if baseline == 'Input-only' else 150))
    for metric in ['quality', 'retrain_js']:
        plans.append(('node', 'Tuned-Full-old', 'Tuned-Random-all', metric, 'selected', 150))
    plans.append(('node', 'Tuned-Full-old', 'Tuned-Structure', 'quality', 'selected', 150))
    for a in ['0.1', '1', '10']:
        plans.append(('node', f'Fixed-Full-input-a{a}', f'Fixed-Full-old-a{a}', 'quality', 'fixed', 150))
    plans.append(('node', 'Fixed-Full-old-a1', 'Fixed-Structure', 'quality', 'fixed', 150))
    for metric in ['quality', 'retrain_js']:
        plans.append(('node', 'Coverage-Class-balanced-train', 'Coverage-Random-train', metric, 'fixed', 150))
    rows, summaries = [], []
    for task, method, baseline, metric, mode, baseline_budget in plans:
        a = metric_series(frame, task, method, metric, mode=mode)
        baseline_mode = 'selected' if baseline_budget == 0 else mode
        b = metric_series(frame, task, baseline, metric, budget=baseline_budget, mode=baseline_mode)
        values = paired_split_difference(a, b, method + ' minus ' + baseline)
        identity = dict(task=task, method=method, baseline=baseline, metric=metric,
                        mode=mode, budget=150, baseline_mode=baseline_mode, baseline_budget=baseline_budget)
        rows.extend(identity | dict(split=split, difference=float(value)) for split, value in values.items())
        summaries.append(identity | confidence(values))
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def withdrawal_interactions(metrics, official):
    """Rebuild all four-cell contrasts and verify the frozen interaction export."""
    import numpy as np
    import pandas as pd
    rows = []
    for task in ['node', 'relation']:
        original = metric_series(metrics, task, 'Original', 'quality', budget=0)
        input_only = metric_series(metrics, task, 'Input-only', 'quality', budget=0)
        continuing = metric_series(metrics, task, 'Continue-FT', 'quality', mode='fixed')
        fixed = metric_series(metrics, task, 'Fixed-FT', 'quality', mode='fixed')
        immediate = paired_split_difference(input_only, original, task + ' input response')
        normal = paired_split_difference(continuing, original, task + ' continued learning')
        withdrawn = paired_split_difference(fixed, input_only, task + ' withdrawn learning')
        group = pd.DataFrame({'split': immediate.index, 'task': task,
                              'input_only_minus_original': immediate.to_numpy(),
                              'no_withdraw_update_gain': normal.to_numpy(),
                              'withdrawn_update_gain': withdrawn.to_numpy(),
                              'interaction': (withdrawn - normal).to_numpy()})
        existing = official[official.task == task].set_index('split')
        for metric in ['no_withdraw_update_gain', 'withdrawn_update_gain', 'interaction']:
            error = paired_split_difference(group.set_index('split')[metric], existing[metric], task + ' ' + metric)
            if not np.allclose(error, 0, atol=1e-12, rtol=0):
                raise ValueError('Frozen withdrawal interaction disagrees with hierarchical metrics: ' + metric)
        rows.append(group)
    result = pd.concat(rows, ignore_index=True)
    return result, summarize_splits(result, ['task'])


def retrain_reference_summaries(frame):
    """Keep the five-request reference diagnostic separate from full formal results."""
    import pandas as pd
    metrics = ['js', 'tv', 'disagreement']
    identifiers = ['task', 'split', 'base_seed', 'request_seed']
    rr = frame[frame.comparison == 'retrain-retrain']
    mr = frame[frame.comparison == 'method-retrain']
    if rr.empty or mr.empty:
        raise ValueError('Both retrain-retrain and method-retrain diagnostics are required')
    # This diagnostic is defined on exactly the first base/request in every split.
    for task in ['node', 'relation']:
        requests = rr[rr.task == task][identifiers].drop_duplicates()
        if len(requests) != 5 or requests.split.nunique() != 5:
            raise ValueError(f'{task}: expected one retraining diagnostic request in each of five splits')
        peer = mr[mr.task == task][identifiers].drop_duplicates()
        if set(map(tuple, peer.to_numpy())) != set(map(tuple, requests.to_numpy())):
            raise ValueError('Method and retrain diagnostics do not use the same requests')
    # Average three retrain pairs within each request; then preserve base/split nesting.
    rr_request = rr.groupby(identifiers)[metrics].mean().reset_index()
    rr_base = rr_request.groupby(['task', 'split', 'base_seed'])[metrics].mean().reset_index()
    rr_split = rr_base.groupby(['task', 'split'])[metrics].mean().reset_index()
    rr_split['comparison'] = 'retrain-retrain'; rr_split['method'] = 'three_retrain_pairs'
    # Average reference-selection repeats for a given retrain first, then the three retrains.
    mr_reference = mr.groupby(identifiers + ['method', 'retrain_index'])[metrics].mean().reset_index()
    mr_request = mr_reference.groupby(identifiers + ['method'])[metrics].mean().reset_index()
    mr_base = mr_request.groupby(['task', 'split', 'base_seed', 'method'])[metrics].mean().reset_index()
    mr_split = mr_base.groupby(['task', 'split', 'method'])[metrics].mean().reset_index()
    mr_split['comparison'] = 'method-retrain'
    split = pd.concat([rr_split, mr_split], ignore_index=True)
    if not (split.groupby(['task', 'comparison', 'method']).size() == 5).all():
        raise ValueError('Incomplete five-split retrain diagnostic')
    summaries = summarize_splits(split, ['task', 'comparison', 'method'])
    paired_rows, paired_summaries = [], []
    for task in ['node', 'relation']:
        current = mr_split[mr_split.task == task]
        baseline = current[current.method == 'Tuned-FT'].set_index('split')
        for method, group in current.groupby('method'):
            if method == 'Tuned-FT':
                continue
            for metric in metrics:
                values = paired_split_difference(group.set_index('split')[metric], baseline[metric], task + ' ' + method)
                identity = dict(task=task, method=method, baseline='Tuned-FT', metric=metric,
                                comparison='distance_to_three_retrains')
                paired_rows.extend(identity | dict(split=sid, difference=float(v)) for sid, v in values.items())
                paired_summaries.append(identity | confidence(values))
    return split, summaries, pd.DataFrame(paired_rows), pd.DataFrame(paired_summaries)

def main():
    import pandas as pd
    tables = OUT / 'tables'
    audit = OUT / 'audit.json'
    if not audit.exists() or not json.loads(audit.read_text()).get('complete'):
        raise RuntimeError('A complete frozen report (audit.json) is required first.')
    destination = OUT / 'paper_tables'
    destination.mkdir(exist_ok=True)
    groups = {
        'metrics': ['task', 'method', 'budget', 'mode'],
        'classes': ['task', 'method', 'budget', 'mode', 'class_id', 'class_name'],
        'transitions': ['task', 'method', 'budget', 'mode', 'class_id', 'class_name'],
        'reference_composition': ['task', 'family', 'level', 'class_id', 'class_name'],
    }
    for name, keys in groups.items():
        frame = pd.read_csv(tables / f'{name}_per_split.csv')
        summarize_splits(frame, keys).to_csv(destination / f'{name}_split_intervals.csv', index=False)
    metrics = pd.read_csv(tables / 'metrics_per_split.csv')
    paired, intervals = paper_paired_contrasts(metrics)
    paired.to_csv(destination / 'paired_contrasts_per_split.csv', index=False)
    intervals.to_csv(destination / 'paired_contrasts_intervals.csv', index=False)
    interactions, intervals = withdrawal_interactions(metrics, pd.read_csv(tables / 'withdrawal_interaction.csv'))
    interactions.to_csv(destination / 'withdrawal_interaction_per_split.csv', index=False)
    intervals.to_csv(destination / 'withdrawal_interaction_intervals.csv', index=False)
    variability, intervals, paired, pair_intervals = retrain_reference_summaries(pd.read_csv(tables / 'retrain_variability.csv'))
    variability.to_csv(destination / 'retrain_reference_per_split.csv', index=False)
    intervals.to_csv(destination / 'retrain_reference_intervals.csv', index=False)
    paired.to_csv(destination / 'retrain_reference_pairs_per_split.csv', index=False)
    pair_intervals.to_csv(destination / 'retrain_reference_pairs_intervals.csv', index=False)
    transitions = pd.read_csv(tables / 'transitions_per_split.csv')
    pairs, summaries = [], []
    for protocol in ['fixed', 'tuned']:
        paired, summary = transition_pairs(transitions, protocol)
        pairs.append(paired); summaries.extend(summary)
    pd.concat(pairs, ignore_index=True).to_csv(destination / 'protection_correction_per_split.csv', index=False)
    pd.DataFrame(summaries).to_csv(destination / 'protection_correction_intervals.csv', index=False)
    raw, split, summary = relation_withdrawal()
    raw.to_csv(destination / 'relation_withdrawal_per_request.csv', index=False)
    split.to_csv(destination / 'relation_withdrawal_per_split.csv', index=False)
    summary.to_csv(destination / 'relation_withdrawal_intervals.csv', index=False)
    development_available = False
    costfile = tables / 'development_search_costs.csv'
    try:
        costs = pd.read_csv(costfile)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        costs = pd.DataFrame()
    if len(costs):
        prep = costs.groupby(['task', 'family', 'split']).first_preparation_s.mean().groupby(['task', 'family']).mean().rename('preparation_seconds_per_base')
        totals = costs.groupby(['task', 'family'])[['training_and_validation_s', 'wall_before_receipt_s']].sum()
        counts = costs.groupby(['task', 'family']).size().rename('candidate_runs')
        pd.concat([prep, totals, counts], axis=1).reset_index().to_csv(destination / 'development_costs.csv', index=False)
        development_available = True
    else:
        stale = destination / 'development_costs.csv'
        if stale.exists():
            raise RuntimeError('Unexpected development cost output without development records; use a consistent run directory.')
    context = json.loads((OUT / 'reproduction_context.json').read_text())
    note = {'protocol': context['protocol'], 'development_cost_available': development_available,
            'statistical_unit': 'Five split means, after requests and bases are averaged; t intervals, sample SD.',
            'quality_scale': 'F1/AP are fractions; multiply by 100 only for paper percentage-point displays.',
            'protection_interval_display': 'Keep protection interval at two decimal places in manuscript.',
            'methods': 'Fixed and tuned contrasts remain separate. Final FT/Full prediction disagreement is a different statistic.',
            'retrain_variability_source': 'reproduction/tables/retrain_variability.csv',
            'no_historical_results_loaded': True}
    (destination / 'README.json').write_text(json.dumps(note, ensure_ascii=False, indent=2)+'\n')
    print('New-run paper statistics written to', destination)


if __name__ == '__main__':
    main()
