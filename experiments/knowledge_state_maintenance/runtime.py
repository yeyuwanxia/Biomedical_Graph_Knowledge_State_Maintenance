from __future__ import annotations

import bz2
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import average_precision_score, roc_auc_score
from data import apply_stratified_split
from hasi.hub_identification import HubScorer, HubScoreConfig
from hasi.anchor_stabilization import AnchorStabilizationLoss

OUT = ROOT / 'reproduction'
POINTS = [25, 50, 100, 150]
FAMILIES = ['FT', 'Full-old', 'Full-input', 'Random-all', 'Structure']
REF_SEEDS = [17001, 17002, 17003]


def safe(x):
    if isinstance(x, dict): return {str(k): safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [safe(v) for v in x]
    if isinstance(x, (np.ndarray, torch.Tensor)): return safe(x.tolist())
    if isinstance(x, np.generic): return safe(x.item())
    if isinstance(x, float) and not math.isfinite(x): return None
    if isinstance(x, Path): return str(x)
    return x


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(safe(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read_json(path): return json.loads(Path(path).read_text())


def sha(path):
    with open(path, 'rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def digest(x): return hashlib.sha256(json.dumps(safe(x), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def tensor_hash(tensors):
    h = hashlib.sha256()
    for k, v in sorted(tensors.items()):
        v = v.detach().cpu().contiguous()
        h.update(k.encode()); h.update(str((str(v.dtype), tuple(v.shape))).encode()); h.update(v.numpy().tobytes())
    return h.hexdigest()


def state(model): return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def receipt(path):
    write_json(path / 'complete.json', {'hashes': {str(p.relative_to(path)): sha(p) for p in sorted(path.rglob('*')) if p.is_file() and p.name != 'complete.json'}})


def verified(path):
    if not (path / 'complete.json').exists(): return False
    assert all(sha(path / p) == h for p, h in read_json(path / 'complete.json')['hashes'].items()), str(path)
    return True


def sync(device):
    if str(device).startswith('cuda'): torch.cuda.synchronize(device)


@contextmanager
def rng(seed, device):
    py, npstate = random.getstate(), np.random.get_state()
    devices = [torch.device(device).index or 0] if str(device).startswith('cuda') else []
    with torch.random.fork_rng(devices=devices):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        try: yield
        finally: random.setstate(py); np.random.set_state(npstate)


def rng_hash(device):
    tensors = {'cpu': torch.get_rng_state()}
    if str(device).startswith('cuda'): tensors['cuda'] = torch.cuda.get_rng_state(device)
    return tensor_hash(tensors)


class Model(nn.Module):
    def __init__(self, features, classes, relation=False):
        super().__init__(); self.relation = relation
        self.convs = nn.ModuleList([GCNConv(features, 64), GCNConv(64, 32 if relation else classes)])
        if relation: self.decoder = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def encode(self, x, edges):
        h = F.relu(self.convs[0](x, edges))
        z = self.convs[1](F.dropout(h, p=.5, training=self.training), edges)
        return z, h

    def decode(self, z, items):
        if not self.relation: return z[items]
        return self.decoder(torch.cat([z[items[:, 0]], z[items[:, 1]]], 1)).flatten()


def new_model(d, seed, device):
    torch.manual_seed(seed)
    return Model(d['x'].shape[1], d['classes'], d['task'] == 'relation').to(device)


def probability(logits, relation):
    return (torch.sigmoid(logits.double()) if relation else torch.softmax(logits.double(), -1)).detach().cpu().numpy()


def class_metrics(y, pred, classes):
    cm = np.bincount(np.asarray(y, dtype=int) * classes + np.asarray(pred, dtype=int), minlength=classes * classes).reshape(classes, classes)
    tp = cm.diagonal().astype(float); support = cm.sum(1); predicted = cm.sum(0)
    precision = np.divide(tp, predicted, out=np.zeros(classes), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(classes), where=support != 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(classes), where=(support + predicted) != 0)
    return dict(precision=precision, recall=recall, f1=f1, support=support, predicted=predicted, confusion=cm)


def quality(p, y, task, classes):
    if task == 'node':
        c = class_metrics(y, p.argmax(1), classes)
        return dict(quality=float(c['f1'].mean()), accuracy=float((p.argmax(1) == y).mean()),
                    nll=float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()))
    return dict(quality=float(average_precision_score(y, p)), auroc=float(roc_auc_score(y, p)),
                brier=float(((p - y) ** 2).mean()),
                nll=float(-(y * np.log(np.clip(p, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1))).mean()))


def distances(p, q):
    if p.ndim == 1: p, q = np.c_[1 - p, p], np.c_[1 - q, q]
    p, q = [np.clip(a, 1e-12, 1) for a in (p, q)]
    p, q = p / p.sum(1, keepdims=True), q / q.sum(1, keepdims=True); m = (p + q) / 2
    return dict(js=.5 * ((p * np.log(p / m)).sum(1) + (q * np.log(q / m)).sum(1)),
                tv=.5 * np.abs(p - q).sum(1), disagreement=(p.argmax(1) != q.argmax(1)).astype(float))


def build_edges(background, pairs, n):
    a = np.concatenate([background, pairs], axis=0)
    a = a[a[:, 0] != a[:, 1]]
    code = np.unique(np.r_[a[:, 0] * n + a[:, 1], a[:, 1] * n + a[:, 0]])
    return torch.from_numpy(np.stack([code // n, code % n])).long()


def prepare_source():
    path = OUT / 'data/relation_source'
    if verified(path): return
    path.mkdir(parents=True, exist_ok=True)
    rawpath = ROOT / 'data/raw/Hetionet/hetionet-v1.0.json.bz2'
    raw = json.loads(bz2.decompress(rawpath.read_bytes()))
    nodes = sorted(raw['nodes'], key=lambda v: (v['kind'], str(v['identifier'])))
    index = {(v['kind'], str(v['identifier'])): i for i, v in enumerate(nodes)}
    pos, bg = [], []
    for e in raw['edges']:
        a, b = e['source_id'], e['target_id']
        u, v = index[a[0], str(a[1])], index[b[0], str(b[1])]
        if {a[0], b[0]} == {'Compound', 'Disease'}:
            if e['kind'] == 'treats': pos.append((u, v) if a[0] == 'Compound' else (v, u))
        else: bg.append((u, v))
    pos = np.array(sorted(set(pos)), dtype=np.int64); assert len(pos) == 755
    kinds = sorted({v['kind'] for v in nodes}); types = np.array([kinds.index(v['kind']) for v in nodes])
    identity = np.random.default_rng(180001).normal(0, 1 / np.sqrt(32), (len(nodes), 32)).astype(np.float32)
    x = torch.from_numpy(np.c_[np.eye(len(kinds), dtype=np.float32)[types], identity])
    compounds = np.flatnonzero(types == kinds.index('Compound')); diseases = np.flatnonzero(types == kinds.index('Disease'))
    known = set(map(tuple, pos)); unknown = np.array([(u, v) for u in compounds for v in diseases if (u, v) not in known], dtype=np.int64)
    torch.save(dict(x=x, background=np.array(bg, dtype=np.int64), positives=pos, unknown=unknown, node_types=types), path / 'source.pt')
    write_json(path / 'metadata.json', dict(raw_sha256=sha(rawpath), node_order='kind, string identifier', kinds=kinds,
        feature_seed=180001, positive_count=len(pos), candidate_count=len(compounds) * len(diseases),
        feature_rule='Type one-hot plus Gaussian node identity, independent of all relation labels and topology',
        background_rule='Exclude all direct Compound-Disease records before adding legal training treats'))
    receipt(path)


def prepare_split(task, split):
    path = OUT / 'data' / task / str(split)
    if verified(path): return
    path.mkdir(parents=True, exist_ok=True)
    if task == 'node':
        rawpath = ROOT / 'data/raw/Hetionet/processed_full_nosource/data.pt'
        raw = torch.load(rawpath, map_location='cpu', weights_only=False)
        data = apply_stratified_split(raw, seed=split)
        d = dict(task=task, split=split, x=data.x, edges=data.edge_index, node_labels=data.y.long(), classes=11)
        for name in ['train', 'val', 'test']:
            ids = torch.where(getattr(data, name + '_mask'))[0]
            d[name + '_items'], d[name + '_y'] = ids, data.y[ids].long()
        origin = dict(raw_sha256=sha(rawpath), feature_protocol='Frozen original topology descriptors; no source-level deletion guarantee')
    else:
        source = torch.load(OUT / 'data/relation_source/source.pt', weights_only=False, map_location='cpu')
        p = source['positives']; order = np.random.default_rng(split).permutation(len(p))
        train, val, test = p[order[:604]], p[order[604:679]], p[order[679:]]
        negorder = np.random.default_rng(split + 100000).permutation(len(source['unknown']))
        vn, tn = source['unknown'][negorder[:750]], source['unknown'][negorder[750:1510]]
        d = dict(task=task, split=split, x=source['x'], edges=build_edges(source['background'], train, len(source['x'])),
                 background=source['background'], train_positive=train, positives=p,
                 legal_negative=source['unknown'][negorder[1510:]], classes=2)
        for name, positive, negative in [('val', val, vn), ('test', test, tn)]:
            d[name + '_positive'] = positive
            d[name + '_items'] = torch.from_numpy(np.concatenate([positive, negative])).long()
            d[name + '_y'] = torch.from_numpy(np.r_[np.ones(len(positive)), np.zeros(len(negative))]).float()
        origin = dict(source_sha256=sha(OUT / 'data/relation_source/source.pt'), counts=[604, 75, 76], unknown_ratio=10)
    torch.save(d, path / 'data.pt'); write_json(path / 'metadata.json', origin | dict(split=split, task=task))
    receipt(path)


def load_data(task, split, base_seed):
    path = OUT / 'data' / task / str(split)
    d = torch.load(path / 'data.pt', map_location='cpu', weights_only=False)
    if task == 'relation':
        selected = np.random.default_rng(base_seed + 200000).choice(len(d['legal_negative']), 604, replace=False)
        d['train_negative'] = d['legal_negative'][selected]
        d['train_items'] = torch.from_numpy(np.concatenate([d['train_positive'], d['train_negative']])).long()
        d['train_y'] = torch.from_numpy(np.r_[np.ones(604), np.zeros(604)]).float()
    return d


def make_request(d, base_seed, index):
    seed = 3000000 + base_seed * 10 + index
    n = len(d['train_items']) if d['task'] == 'node' else len(d['train_positive'])
    positions = np.sort(np.random.default_rng(seed).choice(n, max(1, round(.05 * n)), replace=False))
    targets = d['train_items'][positions].numpy() if d['task'] == 'node' else d['train_positive'][positions]
    survivors = np.setdiff1d(np.arange(len(d['x'])), targets) if d['task'] == 'node' else np.arange(len(d['x']))
    return dict(seed=seed, index=index, base_seed=base_seed, positions=positions.tolist(), targets=targets.tolist(),
                update_seed=seed + 7000000, probe=np.random.default_rng(seed + 9000000).choice(survivors, 512, replace=False).tolist(),
                hash=digest(targets))


def withdraw(d, req):
    a = dict(d); a['x'] = d['x'].clone()
    if d['task'] == 'node':
        targets = torch.tensor(req['targets'], dtype=torch.long)
        removed = torch.zeros(len(d['x']), dtype=torch.bool); removed[targets] = True
        a['x'][targets] = 0
        a['edges'] = d['edges'][:, ~(removed[d['edges'][0]] | removed[d['edges'][1]])]
        keep = ~removed[d['train_items']]
        a['train_items'], a['train_y'] = d['train_items'][keep], d['train_y'][keep]
        assert not torch.isin(a['edges'], targets).any() and not a['x'][targets].any()
        assert not torch.isin(a['train_items'], targets).any()
    else:
        keep = np.ones(len(d['train_positive']), dtype=bool); keep[req['positions']] = False
        a['train_positive'] = d['train_positive'][keep]
        a['edges'] = build_edges(d['background'], a['train_positive'], len(d['x']))
        a['train_items'] = torch.from_numpy(np.concatenate([a['train_positive'], d['train_negative']])).long()
        a['train_y'] = torch.from_numpy(np.r_[np.ones(keep.sum()), np.zeros(len(d['train_negative']))]).float()
        audit_relation(a, req)
    return a


def audit_relation(d, req):
    n = len(d['x']); code = (d['edges'][0] * n + d['edges'][1]).numpy()
    forbidden = np.concatenate([d['val_positive'], d['test_positive'], np.asarray(req['targets']).reshape(-1, 2)])
    forbidden_codes = np.r_[forbidden[:, 0] * n + forbidden[:, 1], forbidden[:, 1] * n + forbidden[:, 0]]
    assert not np.isin(forbidden_codes, code).any(), 'Held-out or withdrawn direct/reverse pair in graph'
    assert not set(map(tuple, d['train_negative'])) & set(map(tuple, d['positives'])), 'Known positive used as negative'
    assert not set(map(tuple, d['train_positive'])) & set(map(tuple, forbidden)), 'Held-out/withdrawn supervision'
    for which in ['val', 'test']:
        assert not set(map(tuple, d['train_negative'])) & set(map(tuple, d[which + '_items'].numpy())), 'Evaluation pair supervised'


def on_device(d, device):
    return {k: (v.to(device) if k in ['x', 'edges', 'train_items', 'train_y', 'val_items', 'val_y', 'test_items', 'test_y'] else v) for k, v in d.items()}


def loss_fn(logits, y, task):
    return F.cross_entropy(logits, y) if task == 'node' else F.binary_cross_entropy_with_logits(logits, y)


def fit(model, d, device, seed, lr=.003, multiplier=0., refs=None, snapshot=None, base=False):
    sync(device); start = time.perf_counter()
    criterion = None
    if refs is not None:
        criterion = AnchorStabilizationLoss(.25 * multiplier, .02 * multiplier)
        criterion.set_snapshots(snapshot, refs['primary'], refs['secondary']); criterion.to(device)
    best_score = -float('inf'); best = None; selected = 0; stale = 0; history = []; checkpoints = {}
    overhead = 0.
    with rng(seed, device):
        before_rng = rng_hash(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=.0005)
        for epoch in range(1, (300 if base else 150) + 1):
            model.train(); optimizer.zero_grad(set_to_none=True)
            z, h = model.encode(d['x'], d['edges']); logits = model.decode(z, d['train_items'])
            supervised = loss_fn(logits, d['train_y'], d['task'])
            extra = criterion(h) if criterion is not None else supervised.new_zeros(())
            objective = supervised + extra
            if not torch.isfinite(objective): raise ValueError('Nonfinite training objective')
            objective.backward(); optimizer.step()
            model.eval()
            with torch.no_grad():
                z, _ = model.encode(d['x'], d['edges'])
                p = probability(model.decode(z, d['val_items']), d['task'] == 'relation')
                score = quality(p, d['val_y'].cpu().numpy(), d['task'], d['classes'])['quality']
            history.append(dict(epoch=epoch, training_loss=float(supervised.detach()), reference_loss=float(extra.detach()), validation_quality=score))
            if score > best_score:
                best_score = score; best = state(model); selected = epoch; stale = 0
            else: stale += 1
            if not base and epoch in POINTS:
                sync(device); diagnostic_start = time.perf_counter()
                checkpoints[epoch] = dict(fixed=state(model), selected=copy.deepcopy(best), selected_epoch=selected,
                    validation_quality=best_score, optimizer_elapsed_s=diagnostic_start - start - overhead)
                overhead += time.perf_counter() - diagnostic_start
            if base and stale >= 50: break
        after_rng = rng_hash(device)
    sync(device)
    if best is None: raise ValueError('No checkpoint')
    model.load_state_dict(best); model.eval()
    return dict(selected_epoch=selected, epochs_ran=epoch, selected_validation_quality=best_score,
                history=history, checkpoints=checkpoints, elapsed_s=time.perf_counter() - start,
                checkpoint_diagnostic_s=overhead, rng_start=before_rng, rng_end=after_rng)


def compute_components(model, d, device):
    import networkx as nx
    times = {}; sync(device); start = time.perf_counter()
    graph = nx.Graph(); graph.add_nodes_from(range(len(d['x'])))
    graph.add_edges_from(d['edges'].t().cpu().tolist())
    times['graph_s'] = time.perf_counter() - start
    scorer = HubScorer(HubScoreConfig(compute_backend='torch', compute_device=device))
    sync(device); start = time.perf_counter()
    centrality = scorer.compute_centrality_scores(graph, list(graph.nodes)); pool = scorer._retain_candidates(centrality)
    sync(device); times['centrality_s'] = time.perf_counter() - start; start = time.perf_counter()
    ppr = scorer.compute_erf_influence(graph, pool)
    sync(device); times['ppr_s'] = time.perf_counter() - start; start = time.perf_counter()
    model.eval(); z, h = model.encode(d['x'], d['edges'])
    loss = loss_fn(model.decode(z, d['train_items']), d['train_y'], d['task'])
    g = torch.autograd.grad(loss, h)[0].detach().norm(dim=1).cpu().numpy()
    gradient = scorer._normalize_dict(dict(enumerate(g)), pool)
    sync(device); times['gradient_s'] = time.perf_counter() - start; start = time.perf_counter()
    structure = {v: .3 * centrality[v] + .3 * ppr[v] for v in pool}
    ranks = {'Structure': sorted(pool, key=lambda v: (-structure[v], v))}
    times['structure_assembly_s'] = time.perf_counter() - start; start = time.perf_counter()
    full = {v: structure[v] + .4 * gradient[v] for v in pool}
    ranks['Full'] = sorted(pool, key=lambda v: (-full[v], v))
    times['full_assembly_s'] = time.perf_counter() - start
    return dict(pool=sorted(pool), full=full, structure=structure, centrality=centrality,
                ranks=ranks, degree=dict(graph.degree()), gradient_label_scope='Only caller supplied training loss', diagnostics=scorer.last_diagnostics), times


def balanced_order(ids, training_labels, seed):
    """Only labels paired with eligible TRAIN ids enter this function."""
    gen = np.random.default_rng(seed)
    buckets = {int(c): list(gen.permutation(np.asarray(ids)[training_labels == c])) for c in np.unique(training_labels)}
    order = []
    while any(buckets.values()):
        for c in sorted(buckets):
            if buckets[c]: order.append(int(buckets[c].pop()))
    return order


def select_refs(family, components, d, req, repeat):
    if family == 'FT': return None
    n = len(d['x']); removed = set(req['targets']) if d['task'] == 'node' else set()
    seed = REF_SEEDS[repeat]
    if family.startswith('Full'): order = components['ranks']['Full']
    elif family == 'Structure': order = components['ranks']['Structure']
    elif family == 'Random-all': order = np.random.default_rng(seed).permutation(n).tolist()
    elif family in ['Random-train', 'Class-balanced-train']:
        ids, labels = d['train_items'].cpu().numpy(), d['train_y'].cpu().numpy()
        keep = ~np.isin(ids, list(removed)); ids, labels = ids[keep], labels[keep]
        order = np.random.default_rng(seed).permutation(ids).tolist() if family == 'Random-train' else balanced_order(ids, labels, seed)
    else: raise ValueError(family)
    order = [int(v) for v in order if int(v) not in removed]
    primary, total = max(1, round(n * .01)), max(1, round(n * .05))
    if len(order) < total: raise ValueError('Insufficient eligible reference nodes; no silent quota change')
    refs = dict(primary=order[:primary], secondary=order[primary:total])
    assert not set(refs['primary']) & set(refs['secondary'])
    return refs


def evaluate(model, d, split, probe):
    model.eval()
    with torch.no_grad():
        z, h = model.encode(d['x'], d['edges'])
        p = probability(model.decode(z, d[split + '_items']), d['task'] == 'relation')
    return p, h[probe].detach().cpu(), h


def grid(family):
    if family == 'FT': return [dict(lr=lr, multiplier=0.) for lr in [.0001, .0002, .0003, .0005, .001, .002, .003, .005, .01]]
    return [dict(lr=lr, multiplier=m) for lr in [.001, .003, .01] for m in [.1, 1., 10.]]
