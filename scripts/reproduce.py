#!/usr/bin/env python3
"""Portable orchestration around the unchanged 2026-09-11 frozen experiment code."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reproduction'
STAGES = ('prepare', 'development', 'formal', 'report', 'analysis', 'all')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def devices_arg(value):
    devices = [v.strip() for v in value.split(',')]
    if not devices or any(not re.fullmatch(r'cuda:\d+', d) for d in devices):
        raise argparse.ArgumentTypeError('Use logical CUDA devices, e.g. cuda:0 or cuda:0,cuda:1,cuda:2')
    if len(set(devices)) != len(devices) or 'cuda:0' not in devices:
        raise argparse.ArgumentTypeError('Devices must be unique and include cuda:0 for the frozen GPU preflight')
    return devices


def verify_frozen():
    expected = read(ROOT / 'EXPERIMENT_SOURCE_MANIFEST.json')['source_hashes']
    actual_paths = list((ROOT / 'src').rglob('*.py')) + list((ROOT / 'experiments/knowledge_state_maintenance').glob('*.py'))
    actual_paths += [ROOT / 'tests/test_knowledge_state_maintenance.py', OUT / 'PROTOCOL.md']
    actual = {str(p.relative_to(ROOT)): sha(p) for p in actual_paths}
    if actual != expected:
        raise RuntimeError('Frozen sources changed. Use a fresh extraction or document a new protocol version.')
    return actual


def input_hashes():
    paths = ['data/raw/Hetionet/hetionet-v1.0.json.bz2',
             'data/raw/Hetionet/processed_full_nosource/data.pt',
             'data/raw/Hetionet/processed_full_nosource/metadata.json']
    missing = [p for p in paths if not (ROOT / p).is_file()]
    if missing:
        raise RuntimeError('Prepare external data first: python scripts/prepare_data.py --raw-file ../datasets/hetionet-v1.0.json.bz2. Missing: ' + ', '.join(missing))
    actual = {p: sha(ROOT / p) for p in paths}
    expected = read(ROOT / 'configs/paper_protocol.json')['raw_graph_sha256']
    if actual[paths[0]] != expected:
        raise RuntimeError('Raw graph SHA256 differs from the paper input; this package targets that exact input.')
    return actual


def check_context(protocol, create=False):
    identity = dict(protocol=protocol, frozen_source_hashes=verify_frozen(), input_hashes=input_hashes())
    path = OUT / 'reproduction_context.json'
    if path.exists():
        saved = read(path)
        for key, value in identity.items():
            if saved[key] != value:
                raise RuntimeError(f'Existing run differs in {key}; use a separate extraction for a new run.')
    elif create:
        stale = [n for n in ['development_freeze.json', 'formal_release.json', 'gpu_preflight.json'] if (OUT / n).exists()]
        if stale:
            raise RuntimeError('Unmanaged prior run detected: ' + ', '.join(stale) + '. Use a fresh extraction.')
        write(path, identity | dict(created_at=now(), note='New reproduction; no historical experimental artifacts shipped.'))
    else:
        raise RuntimeError('Run the prepare stage (or all) first in this extraction.')
    return identity


def capture_environment(torch, devices):
    """Only called when the user runs a new experiment; this is not a historical lock."""
    import importlib.metadata
    packages = {}
    for name in ['torch', 'torch-geometric', 'numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'networkx', 'pytest']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu = []
    for device in devices:
        prop = torch.cuda.get_device_properties(device)
        gpu.append(dict(device=device, name=prop.name, total_memory_bytes=prop.total_memory))
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([param], lr=.003, weight_decay=.0005)
    info = dict(collected_at=now(), scope='This new reproduction only', python=sys.version,
                platform=platform.platform(), cpu_count=os.cpu_count(), cuda_runtime=torch.version.cuda,
                visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), devices=gpu,
                packages=packages, adam_effective_defaults=optimizer.defaults)
    for name in ['cpuinfo', 'meminfo']:
        p = Path('/proc') / name
        if p.exists():
            info[name] = p.read_text()
    for key, command in [('pip_freeze', [sys.executable, '-m', 'pip', 'freeze']),
                         ('nvidia_smi', ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'])]:
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            info[key] = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        except (OSError, subprocess.TimeoutExpired) as error:
            info[key] = dict(error=str(error))
    return info



def check_environment(torch, devices, create=False):
    path = OUT / 'reproduction_environment.json'
    current = json.loads(json.dumps(capture_environment(torch, devices)))
    if path.exists():
        saved = read(path)
        keys = ['python', 'platform', 'cpu_count', 'cuda_runtime', 'visible_devices',
                'devices', 'packages', 'adam_effective_defaults', 'pip_freeze', 'nvidia_smi']
        changed = [key for key in keys if saved.get(key) != current.get(key)]
        if changed:
            raise RuntimeError('Environment or GPU schedule changed: ' + ', '.join(changed) +
                               '. Use a fresh extraction instead of mixing timings and results.')
    elif create:
        write(path, current)
    else:
        raise RuntimeError('The original new-run environment record is missing; run prepare in a clean extraction.')


def load_backend(devices, need_gpu):
    sys.path.insert(0, str(ROOT))
    from experiments.knowledge_state_maintenance import run
    run.torch.set_num_threads(4)
    if need_gpu:
        if not run.torch.cuda.is_available():
            raise RuntimeError('CUDA is required for the full-graph scoring and preflight. Unit tests can run on CPU.')
        count = run.torch.cuda.device_count()
        if any(int(d.split(':')[1]) >= count for d in devices):
            raise RuntimeError(f'Requested {devices}; only {count} logical CUDA devices are visible.')
    return run


def prepare(run, args):
    check_context(args.protocol, create=True)
    exclusion = ROOT / 'results/EXCLUDED_RESULTS.json'
    if not exclusion.exists():
        write(exclusion, dict(entries=[], note='Empty prior-result exclusion policy for a fresh code-only reproduction.'))
    elif read(exclusion).get('entries') != []:
        raise RuntimeError('This launcher requires a fresh exclusion policy; do not import historical results into this extraction.')
    check_environment(run.torch, args.devices, create=True)
    run.gpu_preflight()
    run.freeze()
    run.prepare_source()
    if args.protocol == 'redevelop':
        for task in ['node', 'relation']:
            for split in [181101, 181102]:
                run.prepare_split(task, split)
    else:
        release = paper_release(run)
        for task, config in release['tasks'].items():
            for split in config['base_seeds']:
                run.prepare_split(task, int(split))


def paper_release(run):
    config = read(ROOT / 'configs/paper_protocol.json')
    path = OUT / 'formal_release.json'
    if path.exists():
        release = read(path)
        if release['source_hashes'] != run.sources() or release['tasks'] != config['tasks']:
            raise RuntimeError('Existing formal release does not match the published-parameter protocol.')
        return release
    release = dict(created_at=now(), source_hashes=run.sources(),
                   development_freeze_sha256=sha(OUT / 'development_freeze.json'), tasks=config['tasks'],
                   selection_uses='Published hyperparameters and replication seeds; no development search in this reproduction',
                   formal_test_seen=False, reproduction_protocol='paper')
    run.write_json(path, release)
    return release


def schedule(run, jobs, phase, devices, timeout):
    logs = OUT / phase / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    def lane(index):
        for task, split, seed in jobs[index::len(devices)]:
            key = run.identity(task, split, seed)
            command = [sys.executable, '-u', '-m', 'experiments.knowledge_state_maintenance.run', 'worker',
                       '--task', task, '--split', str(split), '--base-seed', str(seed), '--phase', phase,
                       '--device', devices[index]]
            logfile = logs / (key + '.log')
            print(f'LAUNCH {phase} {key} {devices[index]} -> {logfile.relative_to(ROOT)}', flush=True)
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
            with logfile.open('a') as log:
                try:
                    result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                            timeout=timeout, env=env)
                except subprocess.TimeoutExpired:
                    write(OUT / 'failures' / (phase + '_' + key + '.json'), dict(phase=phase, job=key,
                          reason='worker timeout', timeout_seconds=timeout, log=str(logfile.relative_to(ROOT)), created_at=now()))
                    raise
            if result.returncode:
                write(OUT / 'failures' / (phase + '_' + key + '.json'), dict(phase=phase, job=key,
                      returncode=result.returncode, log=str(logfile.relative_to(ROOT)), created_at=now()))
                raise RuntimeError(f'Worker failed: {key}; inspect {logfile}')
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [pool.submit(lane, i) for i in range(len(devices))]
        for future in futures:
            future.result()


def development(run, args):
    if args.protocol != 'redevelop':
        raise RuntimeError('The paper protocol skips development; use --protocol redevelop in a separate extraction.')
    jobs = [(task, split, 185000 + t * 1000 + j)
            for t, task in enumerate(['node', 'relation']) for j, split in enumerate([181101, 181102])]
    schedule(run, jobs, 'development', args.devices, args.timeout)
    release = run.select_and_release()
    published = read(ROOT / 'configs/paper_protocol.json')['tasks']
    comparison = {task: {key: config[key] == published[task][key] for key in ['selected_configs', 'base_seeds']}
                  for task, config in release['tasks'].items()}
    write(OUT / 'published_configuration_comparison.json', comparison)
    print('Fresh selection vs. paper settings:', json.dumps(comparison), flush=True)


def formal(run, args):
    if not (OUT / 'formal_release.json').exists():
        raise RuntimeError('Formal release is missing; complete development, or prepare with --protocol paper.')
    release = paper_release(run) if args.protocol == 'paper' else read(OUT / 'formal_release.json')
    jobs = []
    for task, config in release['tasks'].items():
        for split, seeds in config['base_seeds'].items():
            run.prepare_split(task, int(split))
            jobs.extend((task, int(split), seed) for seed in seeds)
    schedule(run, jobs, 'formal', args.devices, args.timeout)


def generate_report(args):
    from experiments.knowledge_state_maintenance.report import report
    report()
    if args.protocol == 'paper':
        p = OUT / 'ANALYSIS.md'
        p.write_text('> 本次为 published-parameter 复现：沿用论文超参数/种子，未在本次重新做开发选参或样本量规划。开发搜索成本不可由本次计算。下文原始报告模板中的历史措辞不代表本次执行开发搜索。\n\n' + p.read_text())


def analysis():
    commands = read(ROOT / 'configs/analysis_steps.json')['commands']
    for command in commands:
        print('ANALYSIS', ' '.join(command), flush=True)
        subprocess.run([sys.executable, *command], cwd=ROOT, check=True,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=STAGES, default='all')
    parser.add_argument('--protocol', choices=['redevelop', 'paper'], default='redevelop')
    parser.add_argument('--devices', type=devices_arg, default=devices_arg('cuda:0,cuda:1,cuda:2'))
    parser.add_argument('--timeout', type=int, default=7200, help='Per-worker timeout seconds; historical default: 7200')
    parser.add_argument('--dry-run', action='store_true', help='Print stages only; no data, torch import, output creation or training')
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    if args.protocol == 'paper' and args.stage == 'development':
        parser.error('paper skips development; use redevelop in another extraction')
    stages = (['prepare'] + (['development'] if args.protocol == 'redevelop' else []) + ['formal', 'report', 'analysis']) if args.stage == 'all' else [args.stage]
    if args.dry_run:
        print(json.dumps(dict(protocol=args.protocol, stages=stages, logical_cuda_devices=args.devices,
                              output_directory=str(OUT.relative_to(ROOT)), data_included=False, training_started=False), indent=2))
        return
    verify_frozen()
    need_gpu = any(s in stages for s in ['prepare', 'development', 'formal'])
    run = load_backend(args.devices, need_gpu)
    for stage in stages:
        print('STAGE', stage, flush=True)
        if stage != 'prepare':
            check_context(args.protocol)
            run.check_freeze()
            if stage in ['development', 'formal']:
                check_environment(run.torch, args.devices)
        if stage == 'prepare':
            prepare(run, args)
        elif stage == 'development':
            development(run, args)
        elif stage == 'formal':
            formal(run, args)
        elif stage == 'report':
            generate_report(args)
        elif stage == 'analysis':
            analysis()
    print('Requested stages completed.', flush=True)


if __name__ == '__main__':
    main()
