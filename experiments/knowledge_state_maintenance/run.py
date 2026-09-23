"""Execute preregistered development, freeze selection, then new formal trials."""
from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import subprocess
import tarfile
import traceback
from datetime import datetime, timezone
from .runtime import *


def utc(): return datetime.now(timezone.utc).isoformat()


def sources():
    paths = list((ROOT / 'src').rglob('*.py')) + list((ROOT / 'experiments/knowledge_state_maintenance').glob('*.py'))
    paths += [ROOT / 'tests/test_knowledge_state_maintenance.py', OUT / 'PROTOCOL.md']
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def freeze():
    path = OUT / 'development_freeze.json'
    if path.exists():
        assert read_json(path)['source_hashes'] == sources(), 'Source changed after development freeze'
        return
    cfg = dict(created_at=utc(), source_hashes=sources(), tasks=['node', 'relation'],
               development_splits=[181101, 181102], formal_splits=[181201, 181202, 181203, 181204, 181205],
               reference_seeds=REF_SEEDS, development_grid={f: grid(f) for f in FAMILIES},
               sample_size_rule={'delta_node': .005, 'delta_relation': .01, 'min_bases': 10, 'max_bases': 20,
                   'formula': 'ceil(((1.96+.84)*s/delta)^2), round up to multiple of 5 then clip to [10,20]',
                   'caution': 'Two pilot observations; illustrative planning, not demonstrated power'},
               excluded_results=[e['path'] for e in read_json(ROOT / 'results/EXCLUDED_RESULTS.json')['entries']],
               randomization='Within-base request schedule fixed by request seed; 1 worker per visible GPU',
               environment=dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                   visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), threads=4))
    write_json(path, cfg)
    with tarfile.open(OUT / 'source_snapshot.tar.gz', 'w:gz') as tar:
        for p in cfg['source_hashes']: tar.add(ROOT / p, arcname=p)


def check_freeze():
    assert read_json(OUT / 'development_freeze.json')['source_hashes'] == sources(), 'Frozen source modified'


def identity(task, split, seed): return f'{task}_s{split}_b{seed}'


def prepare_base(task, split, seed, phase, device):
    key = identity(task, split, seed); path = OUT / phase / 'bases' / key
    if verified(path): return path
    path.mkdir(parents=True, exist_ok=True)
    d = load_data(task, split, seed); dd = on_device(d, device)
    model = new_model(d, seed, device); initial = tensor_hash(state(model))
    fitted = fit(model, dd, device, seed + 1000000, lr=.01, base=True)
    torch.save(state(model), path / 'base.pt')
    eval_split = 'val' if phase == 'development' else 'test'
    p, _, h = evaluate(model, dd, eval_split, [0])
    torch.save(h.detach().cpu(), path / 'original_embeddings.pt')
    np.savez_compressed(path / 'original_predictions.npz', probabilities=p,
                        labels=d[eval_split + '_y'].numpy(), items=d[eval_split + '_items'].numpy())
    components, times = compute_components(model, dd, device)
    write_json(path / 'components.json', components)
    sync(device); start = time.perf_counter()
    with torch.no_grad(): _, snap = model.encode(dd['x'], dd['edges'])
    snap.cpu(); sync(device); times['snapshot_s'] = time.perf_counter() - start
    write_json(path / 'fit.json', fitted | {'checkpoints': {}})
    write_json(path / 'metadata.json', dict(task=task, split=split, base_seed=seed, phase=phase,
        evaluation=eval_split, initial_hash=initial, checkpoint_hash=tensor_hash(state(model)),
        input_hash=tensor_hash({k: d[k] for k in ['x', 'edges', 'train_items', 'train_y']}),
        training_s=fitted['elapsed_s'], preparation_times=times, evaluation_quality=quality(p, d[eval_split + '_y'].numpy(), task, d['classes']),
        base_training_seed=seed + 1000000, created_at=utc()))
    receipt(path); print('BASE', key, fitted['selected_epoch'], flush=True)
    return path


def spec(family, lr=.003, multiplier=1., repeat=0, branch='withdrawn'):
    return dict(family=family, lr=float(lr), multiplier=0. if family in ['FT', 'Input-only', 'Matched-Retrain'] else float(multiplier),
                repeat=repeat, branch=branch)


def spec_id(s): return s['family'] + '_' + digest(s)[:12]


def prepare_request(d, req, basepath, phase, device):
    path = OUT / phase / 'requests' / identity(d['task'], d['split'], req['base_seed']) / f"q{req['seed']}"
    if verified(path): return path
    path.mkdir(parents=True, exist_ok=True)
    a = withdraw(d, req); eval_split = 'val' if phase == 'development' else 'test'
    model = new_model(d, req['base_seed'], device)
    model.load_state_dict(torch.load(basepath / 'base.pt', weights_only=True, map_location=device)); model.eval()
    dd, aa = on_device(d, device), on_device(a, device)
    sync(device); start = time.perf_counter()
    pi, _, hi = evaluate(model, aa, eval_split, req['probe'])
    hi_cpu = hi.detach().cpu(); sync(device); input_forward_s = time.perf_counter() - start
    torch.save(hi_cpu, path / 'input_embeddings.pt')
    with np.load(basepath / 'original_predictions.npz') as z: p0 = z['probabilities']
    kwargs = dict(original_probabilities=p0, input_probabilities=pi, labels=a[eval_split + '_y'].numpy(),
                  items=a[eval_split + '_items'].numpy(), probe=np.array(req['probe']),
                  train_items=a['train_items'].numpy(), train_labels=a['train_y'].numpy())
    if d['task'] == 'relation':
        targets = torch.as_tensor(req['targets'], dtype=torch.long, device=device)
        with torch.no_grad():
            zo, _ = model.encode(dd['x'], dd['edges']); zi, _ = model.encode(aa['x'], aa['edges'])
            kwargs['original_withdrawn'] = probability(model.decode(zo, targets), True)
            kwargs['input_withdrawn'] = probability(model.decode(zi, targets), True)
            kwargs['withdrawn_pairs'] = np.array(req['targets'])
    np.savez_compressed(path / 'context.npz', **kwargs)
    write_json(path / 'request.json', req)
    write_json(path / 'audit.json', dict(input_forward_s=input_forward_s,
        input_hash=tensor_hash({k: a[k] for k in ['x', 'edges', 'train_items', 'train_y']}),
        original_input_hash=tensor_hash({k: d[k] for k in ['x', 'edges', 'train_items', 'train_y']}),
        feature_hash=tensor_hash({'x': a['x']}), evaluation_hash=tensor_hash({k: a[k] for k in [eval_split + '_items', eval_split + '_y']}),
        negative_hash=digest(d.get('train_negative', [])), no_test_evaluation=phase == 'development'))
    receipt(path); return path


def execute(d, req, basepath, reqpath, s, phase, device):
    key = identity(d['task'], d['split'], req['base_seed'])
    path = OUT / phase / 'runs' / key / f"q{req['seed']}" / spec_id(s)
    if verified(path): return path
    path.mkdir(parents=True, exist_ok=True); wall_start = time.perf_counter()
    sync(device); start = time.perf_counter()
    a = d if s['branch'] == 'original' else withdraw(d, req)
    aa = on_device(a, device); sync(device); input_s = time.perf_counter() - start
    eval_split = 'val' if phase == 'development' else 'test'
    start = time.perf_counter()
    init_seed = req['base_seed'] + (s['repeat'] * 20000000 if s['family'] == 'Matched-Retrain' else 0)
    model = new_model(d, init_seed, device); initial_hash = tensor_hash(state(model))
    if s['family'] != 'Matched-Retrain': model.load_state_dict(torch.load(basepath / 'base.pt', map_location=device, weights_only=True))
    model_setup_s = time.perf_counter() - start
    refs = None; setup_s = 0.; prep_s = 0.; snapshot = None
    info = read_json(basepath / 'metadata.json'); components = read_json(basepath / 'components.json')
    if s['family'] not in ['FT', 'Input-only', 'Matched-Retrain']:
        sync(device); start = time.perf_counter()
        refs = select_refs(s['family'], components, d, req, s['repeat'])
        spath = reqpath / 'input_embeddings.pt' if s['family'] == 'Full-input' else basepath / 'original_embeddings.pt'
        snapshot = torch.load(spath, weights_only=True, map_location='cpu')
        sync(device); setup_s = time.perf_counter() - start
        t = info['preparation_times']
        if s['family'].startswith('Full'): prep_s = sum(t.values())
        elif s['family'] == 'Structure': prep_s = sum(t[k] for k in ['graph_s', 'centrality_s', 'ppr_s', 'structure_assembly_s', 'snapshot_s'])
        else: prep_s = t['snapshot_s']
        if s['family'] == 'Full-input':
            # Input-conditioned teacher must be computed anew per request.
            setup_s += read_json(reqpath / 'audit.json')['input_forward_s']
            prep_s -= t['snapshot_s']
    if s['family'] == 'Input-only':
        fitted = dict(selected_epoch=0, epochs_ran=0, history=[], checkpoints={}, elapsed_s=0., checkpoint_diagnostic_s=0., rng_start=None, rng_end=None)
        choices = [(0, 'selected', state(model), 0, 0.)]
    elif s['family'] == 'Matched-Retrain':
        fitted = fit(model, aa, device, init_seed + 1000000, lr=.01, base=True)
        choices = [(300, 'selected', state(model), fitted['selected_epoch'], fitted['elapsed_s'])]
    else:
        fitted = fit(model, aa, device, req['update_seed'], lr=s['lr'], multiplier=s['multiplier'], refs=refs, snapshot=snapshot)
        choices = []
        for budget, checkpoint in fitted['checkpoints'].items():
            for mode in ['fixed', 'selected']:
                if phase == 'development' and (budget != 150 or mode != 'selected'): continue
                choices.append((budget, mode, checkpoint[mode], budget if mode == 'fixed' else checkpoint['selected_epoch'], checkpoint['optimizer_elapsed_s']))
    original_h = torch.load(basepath / 'original_embeddings.pt', weights_only=True, map_location='cpu')
    input_h = torch.load(reqpath / 'input_embeddings.pt', weights_only=True, map_location='cpu')
    with np.load(reqpath / 'context.npz') as z: context = {k: z[k] for k in z.files}
    rows = []; eval_start = time.perf_counter()
    for budget, mode, weights, epoch, elapsed in choices:
        sync(device); inference_start = time.perf_counter()
        model.load_state_dict(weights)
        p, probe_h, h = evaluate(model, aa, eval_split, req['probe'])
        sync(device); final_inference_s = time.perf_counter() - inference_start
        y = context['labels']; qm = quality(p, y, d['task'], d['classes'])
        kwargs = dict(probabilities=p, labels=y, items=context['items'], probe_embeddings=probe_h.numpy())
        extras = {}
        if s['family'] != 'Matched-Retrain':
            retained = np.ones(len(a['x']), dtype=bool)
            if d['task'] == 'node': retained[req['targets']] = False
            hc = h.detach().cpu()
            extras.update(retained_l2=float((hc[retained] - original_h[retained]).norm(dim=1).mean()),
                          parameter_update_embedding_l2=float((hc[retained] - input_h[retained]).norm(dim=1).mean()),
                          probe_l2=float((probe_h - original_h[req['probe']]).norm(dim=1).mean()))
            if d['task'] == 'node':
                ids = context['items']
                kwargs['evaluation_original_l2'] = (hc[ids] - original_h[ids]).norm(dim=1).numpy()
                kwargs['evaluation_input_l2'] = (hc[ids] - input_h[ids]).norm(dim=1).numpy()
        if d['task'] == 'relation':
            with torch.no_grad():
                z, _ = model.encode(aa['x'], aa['edges'])
                w = probability(model.decode(z, torch.tensor(req['targets'], device=device)), True)
            kwargs['withdrawn_probabilities'] = w
            extras.update(withdrawn_score=float(w.mean()), withdrawn_change=float((w - context['original_withdrawn']).mean()),
                          withdrawn_post_input_change=float((w - context['input_withdrawn']).mean()))
        for reference, pp in [('original', context['original_probabilities']), ('input', context['input_probabilities'])]:
            extras.update({reference + '_' + k: float(v.mean()) for k, v in distances(p, pp).items()})
        fname = f'{budget}_{mode}.npz'; np.savez_compressed(path / fname, **kwargs)
        rows.append(dict(task=d['task'], split=d['split'], base_seed=req['base_seed'], request_seed=req['seed'],
                         spec=s, budget=budget, mode=mode, selected_epoch=epoch, evaluation=eval_split,
                         probabilities=fname, metrics=qm | extras,
                         cache_ready_update_s=input_s + setup_s + elapsed,
                         first_preparation_s=prep_s, first_use_s=input_s + setup_s + elapsed + prep_s,
                         final_inference_s=final_inference_s, model_setup_s=model_setup_s,
                         service_end_to_end_s=input_s + model_setup_s + setup_s + elapsed + prep_s + final_inference_s))
    sync(device); evaluation_s = time.perf_counter() - eval_start
    best_model = fitted['checkpoints'].get(150, {}).get('selected', state(model))
    torch.save(best_model, path / 'selected.pt')
    write_json(path / 'fit.json', {k: v for k, v in fitted.items() if k != 'checkpoints'})
    if fitted['checkpoints']:
        torch.save({k: {'fixed': v['fixed'], 'selected': v['selected'], 'selected_epoch': v['selected_epoch']} for k, v in fitted['checkpoints'].items()}, path / 'checkpoints.pt')
    write_json(path / 'references.json', refs or dict(primary=[], secondary=[]))
    train_hash = tensor_hash({k: a[k] for k in ['x', 'edges', 'train_items', 'train_y']})
    request_audit = read_json(reqpath / 'audit.json')
    assert train_hash == request_audit['original_input_hash' if s['branch'] == 'original' else 'input_hash']
    if s['family'] == 'Matched-Retrain' and s['repeat'] == 0: assert initial_hash == info['initial_hash']
    if refs:
        assert len(refs['primary']) == max(1, round(len(a['x']) * .01))
        assert len(refs['primary'] + refs['secondary']) == max(1, round(len(a['x']) * .05))
    write_json(path / 'result.json', dict(rows=rows, audit=dict(passed=True, input_hash=train_hash,
        initial_hash=initial_hash, base_checkpoint_hash=info['checkpoint_hash'], update_seed=req['update_seed'],
        update_rng_start=fitted['rng_start'], update_rng_end=fitted['rng_end'], device=device, test_used_for_training=False,
        reference_target='Original model on deleted input' if s['family'] == 'Full-input' else 'Original model on original input',
        evaluation_hash=request_audit['evaluation_hash']),
        timing=dict(input_and_transfer_s=input_s, model_setup_s=model_setup_s, reference_setup_s=setup_s,
                    training_and_validation_s=fitted['elapsed_s'], first_preparation_s=prep_s,
                    evaluation_and_probability_serialization_s=evaluation_s, wall_before_receipt_s=time.perf_counter() - wall_start)))
    receipt(path); print('DONE', key, req['index'], spec_id(s), round(time.perf_counter() - wall_start, 2), flush=True)
    return path


def formal_specs(task, selection, extra_retrains):
    labeled = []
    def add(label, s): labeled.append(dict(label=label, spec=s))
    add('Input-only', spec('Input-only'))
    add('Matched-Retrain', spec('Matched-Retrain'))
    add('Continue-FT', spec('FT', branch='original'))
    add('Fixed-FT', spec('FT'))
    for family in ['Full-old', 'Full-input']:
        for m in [.1, 1., 10.]: add(f'Fixed-{family}-a{m:g}', spec(family, multiplier=m))
    for family in ['Random-all', 'Structure']:
        for repeat in range(3 if family == 'Random-all' else 1): add('Fixed-' + family, spec(family, repeat=repeat))
    if task == 'node':
        for family in ['Random-train', 'Class-balanced-train']:
            for repeat in range(3): add('Coverage-' + family, spec(family, repeat=repeat))
    for family, cfg in selection.items():
        for repeat in range(3 if family == 'Random-all' else 1): add('Tuned-' + family, spec(family, **cfg, repeat=repeat))
    if extra_retrains:
        for repeat in [1, 2]: add('Retrain-variability', spec('Matched-Retrain', repeat=repeat))
    return labeled


def worker(task, split, base_seed, phase, device):
    torch.set_num_threads(4); check_freeze()
    if phase == 'formal':
        release = read_json(OUT / 'formal_release.json'); assert release['source_hashes'] == sources()
    basepath = prepare_base(task, split, base_seed, phase, device)
    d = load_data(task, split, base_seed)
    for i in range(1 if phase == 'development' else 2):
        req = make_request(d, base_seed, i); reqpath = prepare_request(d, req, basepath, phase, device)
        if phase == 'development':
            labeled = [dict(label=f'Development-{family}-{j}', spec=spec(family, **cfg)) for family in FAMILIES for j, cfg in enumerate(grid(family))]
        else:
            cfg = release['tasks'][task]
            labeled = formal_specs(task, cfg['selected_configs'], i == 0 and base_seed == cfg['base_seeds'][str(split)][0])
        write_json(reqpath / 'planned_runs.json', labeled)
        # planned_runs is prospective metadata added AFTER request context receipt.
        unique = {spec_id(r['spec']): r['spec'] for r in labeled}
        keys = sorted(unique); np.random.default_rng(req['seed'] + 11000000).shuffle(keys)
        for k in keys: execute(d, req, basepath, reqpath, unique[k], phase, device)
    write_json(OUT / phase / 'workers' / (identity(task, split, base_seed) + '.json'), dict(complete=True, completed_at=utc()))


def batches(jobs, phase):
    logs = OUT / phase / 'logs'; logs.mkdir(parents=True, exist_ok=True)
    def lane(lane_id):
        for task, split, seed in jobs[lane_id::3]:
            key = identity(task, split, seed); logfile = logs / (key + '.log')
            args = [sys.executable, '-u', '-m', 'experiments.knowledge_state_maintenance.run', 'worker', '--task', task,
                    '--split', str(split), '--base-seed', str(seed), '--phase', phase, '--device', f'cuda:{lane_id}']
            print('LAUNCH', phase, key, 'gpu', lane_id, flush=True)
            with logfile.open('a') as log:
                result = subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=7200)
            if result.returncode:
                write_json(OUT / 'failures' / (phase + '_' + key + '.json'), dict(phase=phase, job=key, returncode=result.returncode, log=str(logfile.relative_to(ROOT)), timestamp=utc()))
                raise RuntimeError(f'{key} failed; see {logfile}')
            print('WORKER_COMPLETE', phase, key, flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(lane, i) for i in range(3)]
        for future in futures: future.result()


def select_and_release():
    path = OUT / 'formal_release.json'
    if path.exists():
        assert read_json(path)['source_hashes'] == sources(); return read_json(path)
    records = []
    for p in sorted((OUT / 'development/runs').glob('*/*/*/result.json')):
        assert verified(p.parent)
        records.extend(read_json(p)['rows'])
    tasks = {}
    for task in ['node', 'relation']:
        selected = {}; details = {}
        for family in FAMILIES:
            values = []
            for cfg in grid(family):
                matches = [r for r in records if r['task'] == task and r['spec'] == spec(family, **cfg)]
                assert len(matches) == 2, (task, family, cfg, len(matches))
                values.append(float(np.mean([r['metrics']['quality'] for r in matches])))
            index = int(np.argmax(values)); selected[family] = grid(family)[index]
            details[family] = dict(validation_means=values, selected_index=index, candidates=grid(family))
        paired = []
        for split in [181101, 181102]:
            def get(s):
                match = [r for r in records if r['task'] == task and r['split'] == split and r['spec'] == s]
                assert len(match) == 1; return match[0]['metrics']['quality']
            paired.append(get(spec('Full-old')) - get(spec('FT')))
        sd = float(np.std(paired, ddof=1)); delta = .005 if task == 'node' else .01
        raw = math.ceil(((1.96 + .84) * sd / delta) ** 2)
        per_split = min(4, max(2, math.ceil(raw / 5)))
        bases = {str(split): [190000 + (0 if task == 'node' else 10000) + j * 10 + k for k in range(per_split)]
                 for j, split in enumerate([181201, 181202, 181203, 181204, 181205])}
        tasks[task] = dict(selected_configs=selected, selection_details=details, pilot_differences=paired,
            pilot_sd=sd, planning_delta=delta, raw_planned_n=raw, actual_bases=per_split * 5,
            planning_capped=raw > 20, power_claim=False, base_seeds=bases)
    release = dict(created_at=utc(), source_hashes=sources(), development_freeze_sha256=sha(OUT / 'development_freeze.json'),
                   tasks=tasks, selection_uses='development validation only', formal_test_seen=False)
    write_json(path, release); return release


def run_all():
    torch.set_num_threads(4)
    if not torch.cuda.is_available(): raise RuntimeError('GPU execution requires sandbox-external authorization')
    if torch.cuda.device_count() < 3: raise RuntimeError('Expected three visible GPUs; no silent device change')
    gpu_preflight()
    freeze()
    prepare_source()
    for task in ['node', 'relation']:
        for split in [181101, 181102]: prepare_split(task, split)
    development_jobs = [(task, split, 185000 + t * 1000 + j) for t, task in enumerate(['node', 'relation']) for j, split in enumerate([181101, 181102])]
    write_json(OUT / 'execution_status.json', dict(stage='development', started_at=utc(), jobs=development_jobs))
    batches(development_jobs, 'development')
    release = select_and_release()
    jobs = []
    for task, cfg in release['tasks'].items():
        for split, seeds in cfg['base_seeds'].items():
            prepare_split(task, int(split))
            for seed in seeds: jobs.append((task, int(split), seed))
    write_json(OUT / 'execution_status.json', dict(stage='formal', released_at=utc(), jobs=jobs))
    batches(jobs, 'formal')
    from .report import report
    report()
    write_json(OUT / 'execution_status.json', dict(stage='complete', completed_at=utc(), formal_jobs=jobs))


def gpu_preflight():
    """Real full graph zero-weight parity, before any development outcomes."""
    path = OUT / 'gpu_preflight.json'
    if path.exists():
        saved = read_json(path)
        assert saved['passed'] and saved['source_hashes'] == sources()
        return
    torch.set_num_threads(4); device = 'cuda:0'
    raw = torch.load(ROOT / 'data/raw/Hetionet/processed_full_nosource/data.pt', map_location='cpu', weights_only=False)
    data = apply_stratified_split(raw, seed=180099)
    d = dict(x=data.x, classes=11, task='node')
    x, edges = data.x.to(device), data.edge_index.to(device)
    ids, labels = torch.where(data.train_mask)[0].to(device), data.y[data.train_mask].to(device)
    results = []
    for with_zero_reference in [False, True]:
        model = new_model(d, 180098, device); model.eval()
        with torch.no_grad(): _, h = model.encode(x, edges)
        criterion = AnchorStabilizationLoss(0., 0.)
        criterion.set_snapshots(h.detach(), list(range(50)), list(range(50, 200)))
        with rng(180097, device):
            before = rng_hash(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=.003, weight_decay=.0005)
            losses = []
            for _ in range(3):
                model.train(); optimizer.zero_grad(set_to_none=True)
                z, h = model.encode(x, edges); loss = F.cross_entropy(z[ids], labels)
                if with_zero_reference: loss = loss + criterion(h)
                loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
            after = rng_hash(device)
        results.append(dict(state=state(model), before=before, after=after, losses=losses))
    largest = 0.
    for k in results[0]['state']:
        a, b = results[0]['state'][k], results[1]['state'][k]
        assert torch.allclose(a, b, atol=2e-6, rtol=1e-5), k
        largest = max(largest, float((a - b).abs().max()))
    assert results[0]['before'] == results[1]['before'] and results[0]['after'] == results[1]['after']
    write_json(path, dict(passed=True, source_hashes=sources(), full_graph_nodes=len(x), device=device,
        tested_steps=3, maximum_parameter_difference=largest, atol=2e-6, rtol=1e-5, shared_rng_verified=True,
        training_losses=[r['losses'] for r in results], created_at=utc()))
    del x, edges, model; torch.cuda.empty_cache()
    print('GPU_PREFLIGHT_PASSED', largest, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['run', 'worker', 'report'])
    parser.add_argument('--task', choices=['node', 'relation']); parser.add_argument('--split', type=int)
    parser.add_argument('--base-seed', type=int); parser.add_argument('--phase', choices=['development', 'formal'])
    parser.add_argument('--device', default='cuda:0'); args = parser.parse_args()
    try:
        if args.command == 'worker': worker(args.task, args.split, args.base_seed, args.phase, args.device)
        elif args.command == 'report':
            from .report import report
            report()
        else: run_all()
    except Exception as error:
        write_json(OUT / 'failures' / (f'controller_{int(time.time())}.json'), dict(error=str(error), traceback=traceback.format_exc(), time=utc()))
        raise
