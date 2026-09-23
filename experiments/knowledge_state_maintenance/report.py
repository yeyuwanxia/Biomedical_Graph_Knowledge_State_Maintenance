"""Independent probability recomputation and split-level paired reporting."""
from __future__ import annotations

import itertools
import pandas as pd
from scipy.stats import t, ttest_1samp
from .runtime import *


def aggregate(frame, keys, filename):
    metrics = [c for c in frame.select_dtypes(include='number').columns if c not in keys + ['split', 'base_seed', 'request_seed', 'repeat']]
    frame.to_csv(OUT / 'tables' / (filename + '_per_run.csv'), index=False)
    request = frame.groupby(keys + ['split', 'base_seed', 'request_seed'], dropna=False)[metrics].mean().reset_index()
    base = request.groupby(keys + ['split', 'base_seed'], dropna=False)[metrics].mean().reset_index()
    split = base.groupby(keys + ['split'], dropna=False)[metrics].mean().reset_index()
    base.to_csv(OUT / 'tables' / (filename + '_per_base.csv'), index=False)
    split.to_csv(OUT / 'tables' / (filename + '_per_split.csv'), index=False)
    summary = split.groupby(keys, dropna=False)[metrics].agg(['mean', 'std', 'count'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary = summary.reset_index(); summary.to_csv(OUT / 'tables' / (filename + '_summary.csv'), index=False)
    return split, summary


def interval(values):
    a = np.asarray(values, dtype=float); n = len(a); mean = float(a.mean()); sd = float(a.std(ddof=1)) if n > 1 else float('nan')
    half = float(t.ppf(.975, n - 1) * sd / np.sqrt(n)) if n > 1 else float('nan')
    return dict(n_splits=n, mean=mean, sd=sd, lower=mean - half, upper=mean + half,
                positive=int((a > 0).sum()), negative=int((a < 0).sum()), ties=int((a == 0).sum()))


def report():
    from .run import spec, spec_id, sources
    assert read_json(OUT / 'formal_release.json')['source_hashes'] == sources()
    (OUT / 'tables').mkdir(exist_ok=True)
    rows, classes, transitions, cover, variability = [], [], [], [], []
    expected, completed, errors, training_audits = set(), set(), [], []
    max_error = 0.; checked = 0
    names = {v: k for k, v in read_json(ROOT / 'data/raw/Hetionet/processed_full_nosource/metadata.json')['class_mapping'].items()}
    for reqpath in sorted((OUT / 'formal/requests').glob('*/*')):
        if not verified(reqpath): continue
        req = read_json(reqpath / 'request.json'); key = reqpath.parent.name
        task = key.split('_s')[0]; split = int(key.split('_s')[1].split('_b')[0]); seed = req['base_seed']
        runroot = OUT / 'formal/runs' / key / reqpath.name
        planned = read_json(reqpath / 'planned_runs.json')
        labels = {}
        for plan in planned: labels.setdefault(spec_id(plan['spec']), []).append(plan['label'])
        with np.load(reqpath / 'context.npz') as z: context = {k: z[k] for k in z.files}
        retrain_path = runroot / spec_id(spec('Matched-Retrain')) / '300_selected.npz'
        with np.load(retrain_path) as z: retrain = {k: z[k] for k in z.files}
        y = context['labels']; original_pred = context['original_probabilities'].argmax(1) if task == 'node' else None
        d = load_data(task, split, seed)
        component = read_json(OUT / 'formal/bases' / key / 'components.json')
        for sid, aliases in labels.items():
            path = runroot / sid; expected.add(str(path.relative_to(OUT)))
            if not verified(path): errors.append(f'Missing {path}'); continue
            completed.add(str(path.relative_to(OUT)))
            result = read_json(path / 'result.json'); assert result['audit']['passed']
            s = result['rows'][0]['spec']
            training_audits.append(dict(task=task, split=split, base_seed=seed, request_seed=req['seed'], spec_id=sid, **result['audit']))
            refs = read_json(path / 'references.json')
            for rec in result['rows']:
                with np.load(path / rec['probabilities']) as z: z = {k: z[k] for k in z.files}
                assert np.array_equal(z['items'], context['items']) and np.array_equal(z['labels'], y)
                p = z['probabilities']; q = quality(p, y, task, d['classes'])
                for metric, value in q.items():
                    error = abs(rec['metrics'][metric] - value); max_error = max(error, max_error); checked += 1
                    assert error < 1e-12
                q.update({'retrain_' + k: float(v.mean()) for k, v in distances(p, retrain['probabilities']).items()})
                if task == 'relation': q['withdrawn_retrain_mae'] = float(np.abs(z['withdrawn_probabilities'] - retrain['withdrawn_probabilities']).mean())
                for label in aliases:
                    meta = dict(task=task, split=split, base_seed=seed, request_seed=req['seed'], repeat=s['repeat'],
                                method=label, budget=rec['budget'], mode=rec['mode'])
                    rows.append(meta | rec['metrics'] | q | {k: rec[k] for k in ['cache_ready_update_s', 'first_preparation_s', 'first_use_s', 'selected_epoch', 'final_inference_s', 'model_setup_s', 'service_end_to_end_s']})
                    if task != 'node': continue
                    cm = class_metrics(y, p.argmax(1), 11)
                    for c in range(11):
                        mask = y == c
                        classes.append(meta | dict(class_id=c, class_name=names[c], precision=cm['precision'][c], recall=cm['recall'][c],
                            f1=cm['f1'][c], support=cm['support'][c], predicted=cm['predicted'][c],
                            retrain_js=float(distances(p[mask], retrain['probabilities'][mask])['js'].mean()),
                            original_embedding_l2=float(z['evaluation_original_l2'][mask].mean()) if 'evaluation_original_l2' in z else float('nan'),
                            input_embedding_l2=float(z['evaluation_input_l2'][mask].mean()) if 'evaluation_input_l2' in z else float('nan')))
                    if rec['budget'] != 150 and s['family'] not in ['Matched-Retrain', 'Input-only']: continue
                    ftplan = next(v for v in planned if v['label'] == ('Tuned-FT' if label.startswith('Tuned') else 'Fixed-FT'))
                    with np.load(runroot / spec_id(ftplan['spec']) / ('150_' + rec['mode'] + '.npz')) as fz: fp = fz['probabilities'].argmax(1)
                    pred = p.argmax(1)
                    for c in [-1] + list(range(11)):
                        mask = np.ones(len(y), dtype=bool) if c == -1 else y == c
                        lost = mask & (fp == y) & (pred != y); saved = mask & (fp != y) & (pred == y)
                        transitions.append(meta | dict(class_id=c, class_name='ALL' if c == -1 else names[c],
                            support=int(mask.sum()), ft_correct_reference_wrong=int(lost.sum()), ft_wrong_reference_correct=int(saved.sum()),
                            net_correct_change=int(saved.sum() - lost.sum()), lost_originally_wrong=int((lost & (original_pred != y)).sum()),
                            originally_wrong_corrected=int((mask & (original_pred != y) & (pred == y)).sum()),
                            originally_wrong_remaining=int((mask & (original_pred != y) & (pred != y)).sum()),
                            originally_correct_retained=int((mask & (original_pred == y) & (pred == y)).sum()),
                            originally_correct_lost=int((mask & (original_pred == y) & (pred != y)).sum())))
            if task == 'node' and refs['primary']:
                for level in ['primary', 'secondary']:
                    ids = np.array(refs[level]); label_values = d['node_labels'][ids].numpy()
                    degrees = np.array([component['degree'][str(v)] for v in ids])
                    for c in range(11):
                        cover.append(dict(task=task, split=split, base_seed=seed, request_seed=req['seed'], repeat=s['repeat'], family=s['family'],
                            level=level, class_id=c, class_name=names[c], count=int((label_values == c).sum()), fraction=float((label_values == c).mean()),
                            mean_degree=float(degrees.mean()), degree_p25=float(np.quantile(degrees, .25)),
                            degree_median=float(np.median(degrees)), degree_p75=float(np.quantile(degrees, .75)),
                            class_mean_degree=float(degrees[label_values == c].mean()) if (label_values == c).any() else float('nan')))
        # Repeated retraining is an uncertainty diagnostic, not extra method trials.
        extra_paths = [runroot / spec_id(spec('Matched-Retrain', repeat=r)) / '300_selected.npz' for r in range(3)]
        if all(p.exists() for p in extra_paths):
            rp = [np.load(p)['probabilities'] for p in extra_paths]
            for a, b in itertools.combinations(range(3), 2):
                variability.append(dict(task=task, split=split, base_seed=seed, request_seed=req['seed'], comparison='retrain-retrain', method=f'R{a}-R{b}',
                    **{k: float(v.mean()) for k, v in distances(rp[a], rp[b]).items()}))
            for plan in planned:
                if not plan['label'].startswith('Tuned') and plan['label'] != 'Input-only': continue
                fname = '0_selected.npz' if plan['label'] == 'Input-only' else '150_selected.npz'
                with np.load(runroot / spec_id(plan['spec']) / fname) as z: pp = z['probabilities']
                for r, reference in enumerate(rp):
                    variability.append(dict(task=task, split=split, base_seed=seed, request_seed=req['seed'], comparison='method-retrain', method=plan['label'],
                        repeat=plan['spec']['repeat'], retrain_index=r, **{k: float(v.mean()) for k, v in distances(pp, reference).items()}))
        # Four-cell interaction uses fixed-150 FT on both input branches.
        original_quality = quality(context['original_probabilities'], y, task, d['classes'])['quality']
        for label in ['Original']:
            rows.append(dict(task=task, split=split, base_seed=seed, request_seed=req['seed'], repeat=0, method=label,
                             budget=0, mode='selected', quality=original_quality))
    if errors: raise RuntimeError('\n'.join(errors))
    frame = pd.DataFrame(rows)
    splitframe, summary = aggregate(frame, ['task', 'method', 'budget', 'mode'], 'metrics')
    classsplit, _ = aggregate(pd.DataFrame(classes), ['task', 'method', 'budget', 'mode', 'class_id', 'class_name'], 'classes')
    aggregate(pd.DataFrame(transitions), ['task', 'method', 'budget', 'mode', 'class_id', 'class_name'], 'transitions')
    aggregate(pd.DataFrame(cover).drop_duplicates(), ['task', 'family', 'level', 'class_id', 'class_name'], 'reference_composition')
    pd.DataFrame(variability).to_csv(OUT / 'tables/retrain_variability.csv', index=False)
    pd.DataFrame(training_audits).to_csv(OUT / 'tables/training_audit.csv', index=False)
    search_costs = []
    for path in sorted((OUT / 'development/runs').glob('*/*/*/result.json')):
        result = read_json(path); rec = result['rows'][0]
        search_costs.append(dict(task=rec['task'], family=rec['spec']['family'], split=rec['split'],
            **result['timing']))
    pd.DataFrame(search_costs).to_csv(OUT / 'tables/development_search_costs.csv', index=False)
    pairs = []; decisions = []; interactions = []
    release = read_json(OUT / 'formal_release.json')
    for task, config in release['tasks'].items():
        for split_id, base_seeds in config['base_seeds'].items():
            for base_seed in base_seeds:
                key = f'{task}_s{split_id}_b{base_seed}'
                assert read_json(OUT / 'formal/workers' / (key + '.json'))['complete']
                assert len(list((OUT / 'formal/requests' / key).glob('*/planned_runs.json'))) == 2
    for task in ['node', 'relation']:
        current = splitframe[(splitframe.task == task) & (splitframe['mode'] == 'selected')]
        current = current[(current.budget == 150) | current.method.isin(['Input-only', 'Matched-Retrain', 'Original'])]
        def pick(method, metric):
            return current[current.method == method].set_index('split')[metric].sort_index()
        for method in ['Tuned-Full-old', 'Tuned-Full-input', 'Tuned-Random-all', 'Tuned-Structure']:
            qci = interval(pick(method, 'quality') - pick('Tuned-FT', 'quality'))
            jci = interval(pick(method, 'retrain_js') - pick('Tuned-FT', 'retrain_js'))
            qmean = current[current.method == method].mean(numeric_only=True)
            ftmean = current[current.method == 'Tuned-FT'].mean(numeric_only=True)
            rtmean = current[current.method == 'Matched-Retrain'].mean(numeric_only=True)
            reduction = 1 - qmean.retrain_js / ftmean.retrain_js
            gate = qci['lower'] > -release['tasks'][task]['planning_delta'] and reduction >= .1 and jci['upper'] < 0 and qmean.cache_ready_update_s <= 1.25 * ftmean.cache_ready_update_s and qmean.cache_ready_update_s < rtmean.cache_ready_update_s
            decisions.append(dict(task=task, method=method, joint_engineering_criterion=bool(gate), quality_delta_ci=qci, js_delta_ci=jci,
                                  js_relative_reduction=float(reduction), first_use_faster_than_retrain=bool(qmean.first_use_s < rtmean.cache_ready_update_s)))
            for baseline in ['Tuned-FT', 'Input-only', 'Tuned-Random-all']:
                for metric in ['quality', 'retrain_js', 'cache_ready_update_s']:
                    pairs.append(dict(task=task, method=method, baseline=baseline, metric=metric, **interval(pick(method, metric) - pick(baseline, metric))))
        for mode in ['fixed', 'selected']:
            for family in ['Fixed-Full-old-a0.1', 'Fixed-Full-old-a1', 'Fixed-Full-old-a10', 'Fixed-Full-input-a0.1', 'Fixed-Full-input-a1', 'Fixed-Full-input-a10']:
                a = splitframe[(splitframe.task == task) & (splitframe.method == family) & (splitframe.budget == 150) & (splitframe['mode'] == mode)].set_index('split')
                b = splitframe[(splitframe.task == task) & (splitframe.method == 'Fixed-FT') & (splitframe.budget == 150) & (splitframe['mode'] == mode)].set_index('split')
                for metric in ['quality', 'retrain_js']:
                    pairs.append(dict(task=task, method=family, baseline='Fixed-FT', mode=mode, metric=metric, **interval(a[metric] - b[metric])))
        for split in sorted(current.split.unique()):
            def v(method, budget, mode):
                z = frame[(frame.task == task) & (frame.split == split) & (frame.method == method) & (frame.budget == budget) & (frame['mode'] == mode)]
                return z.groupby('base_seed').quality.mean().mean()
            normal = v('Continue-FT', 150, 'fixed') - v('Original', 0, 'selected')
            withdrawn = v('Fixed-FT', 150, 'fixed') - v('Input-only', 0, 'selected')
            interactions.append(dict(task=task, split=int(split), no_withdraw_update_gain=normal, withdrawn_update_gain=withdrawn, interaction=withdrawn - normal))
    # Holm-adjusted class tests, based on five split means, not thousands of nodes.
    class_tests = []
    for method in ['Tuned-Full-old', 'Tuned-Full-input']:
        a = classsplit[(classsplit.method == method) & (classsplit.budget == 150) & (classsplit['mode'] == 'selected')]
        b = classsplit[(classsplit.method == 'Tuned-FT') & (classsplit.budget == 150) & (classsplit['mode'] == 'selected')]
        current = []
        for c in range(11):
            diff = a[a.class_id == c].set_index('split').f1 - b[b.class_id == c].set_index('split').f1
            pvalue = float(ttest_1samp(diff, 0).pvalue) if diff.std() > 0 else (1. if diff.mean() == 0 else 0.)
            current.append(dict(method=method, class_id=c, class_name=names[c], p=pvalue, **interval(diff)))
        order = np.argsort([v['p'] for v in current]); running = 0.
        for rank, index in enumerate(order):
            running = max(running, min(1., (11 - rank) * current[index]['p'])); current[index]['holm_p'] = running
        class_tests.extend(current)
    pd.DataFrame(pairs).to_csv(OUT / 'tables/paired_intervals.csv', index=False)
    pd.DataFrame(interactions).to_csv(OUT / 'tables/withdrawal_interaction.csv', index=False)
    pd.DataFrame(class_tests).to_csv(OUT / 'tables/class_tests.csv', index=False)
    write_json(OUT / 'decisions.json', decisions)
    write_json(OUT / 'audit.json', dict(complete=True, expected_physical_runs=len(expected), complete_physical_runs=len(completed),
        max_probability_recomputation_error=max_error, probability_metrics_checked=checked, all_receipts_verified=True,
        formal_source_hashes_verified=True, no_20260905_results_used=True,
        statistical_unit='Reference repetitions -> requests -> bases -> 5 split means; one biomedical graph',
        explicit_caution='Confidence intervals and Holm tests are conditional, limited-split descriptions; not formal unlearning guarantees'))
    lines = ['# 新正式维护实验结果', '', '9月5日结果已排除。开发选参及样本量在正式测试前冻结。以下正负结果全部保留。', '',
             '|任务|方法|质量(Macro-F1/AP)|重训JS|缓存就绪秒|首次含准备秒|', '|---|---|---:|---:|---:|---:|']
    selected = summary[(summary['mode'] == 'selected') & ((summary.method.str.startswith('Tuned')) | summary.method.isin(['Input-only', 'Matched-Retrain']))]
    selected = selected[(selected.budget == 150) | selected.method.isin(['Input-only', 'Matched-Retrain'])]
    for r in selected.itertuples():
        lines.append(f'|{r.task}|{r.method}|{r.quality_mean:.6f}|{r.retrain_js_mean:.6f}|{r.cache_ready_update_s_mean:.3f}|{r.first_use_s_mean:.3f}|')
    lines += ['', '机制主结果为固定150步；方法主结果为开发选定配置的验证最佳模型。其它预算点完整保留在tables。', '',
              'withdrawal_interaction.csv报告无撤回继续训练与撤回后FT的增益差；不能自动解释为参数记忆消除。',
              'transitions_*报告原错误纠正与原正确预测损失；classes_*报告全部11类。覆盖实验仅使用保留训练标签，度分布差异另列。',
              'retrain_variability.csv比较每指定请求三个重训实例，样本有限，不证明模型分布等价。', '', '## 预设工程判定']
    for d in decisions: lines.append(f"- {d['task']} / {d['method']}：联合判定={d['joint_engineering_criterion']}；首次含准备快于重训={d['first_use_faster_than_retrain']}。")
    lines += ['', '阈值为执行前确定的研究工程标准，不是临床阈值。资源封顶、开发方差估计不足、新划分仍属于同一图等限制见formal_release.json及PROTOCOL.md。',
              '原始运行及概率、参考集合、源代码快照、损失和验证轨迹均保存；本报告不把负结果改写为成功的方法优势。']
    (OUT / 'ANALYSIS.md').write_text('\n'.join(lines) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (OUT / 'figures').mkdir(exist_ok=True)
    for task in ['node', 'relation']:
        values = summary[(summary.task == task) & (summary['mode'] == 'selected') & summary.method.str.startswith('Tuned')]
        for xcolumn, xlabel, suffix in [('cache_ready_update_s_mean', 'Cache-ready update seconds', 'quality_cost'),
                                      ('retrain_js_mean', 'JS to matched retrain', 'quality_js')]:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for method, group in values.groupby('method'):
                group = group.sort_values('budget')
                ax.plot(group[xcolumn], group.quality_mean, marker='o', label=method)
                for point in group.itertuples(): ax.annotate(str(point.budget), (getattr(point, xcolumn), point.quality_mean), fontsize=7)
            ax.set(xlabel=xlabel, ylabel='Macro-F1' if task == 'node' else 'Average precision', title=task + ': validation-selected checkpoints at fixed budgets')
            ax.legend(fontsize=7); ax.grid(alpha=.2); fig.tight_layout()
            fig.savefig(OUT / 'figures' / f'{task}_{suffix}.pdf'); plt.close(fig)
    print('REPORT_COMPLETE', len(completed), 'physical runs', checked, 'metrics', flush=True)
