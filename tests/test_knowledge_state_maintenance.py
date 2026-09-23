import copy
import numpy as np
import pytest
import torch
from sklearn.metrics import f1_score
from experiments.knowledge_state_maintenance.runtime import (
    Model, class_metrics, fit, new_model, on_device, withdraw, audit_relation,
    balanced_order, select_refs, build_edges, tensor_hash, grid,
)


def toy():
    torch.manual_seed(100)
    n = 15
    return dict(task='node', classes=3, x=torch.randn(n, 5),
        edges=torch.tensor([[i for i in range(n - 1)] + [i + 1 for i in range(n - 1)],
                            [i + 1 for i in range(n - 1)] + [i for i in range(n - 1)]]),
        train_items=torch.arange(9), train_y=torch.arange(9) % 3,
        val_items=torch.arange(9, 15), val_y=torch.arange(9, 15) % 3,
        test_items=torch.tensor([999999]), test_y=torch.tensor([999999]))


def test_zero_reference_reproduces_ft_and_test_is_not_read():
    torch.set_num_threads(1)
    d = toy(); a, b = new_model(d, 12, 'cpu'), new_model(d, 12, 'cpu')
    a.eval()
    with torch.no_grad(): _, h = a.encode(d['x'], d['edges'])
    fa = fit(a, d, 'cpu', 42)
    fb = fit(b, d, 'cpu', 42, multiplier=0., refs={'primary': [1, 2], 'secondary': [4]}, snapshot=h)
    assert fa['rng_start'] == fb['rng_start'] and fa['rng_end'] == fb['rng_end']
    assert fa['selected_epoch'] == fb['selected_epoch']
    for point in [25, 50, 100, 150]:
        for mode in ['fixed', 'selected']:
            assert tensor_hash(fa['checkpoints'][point][mode]) == tensor_hash(fb['checkpoints'][point][mode])


def test_validation_score_reproduces_saved_state():
    d = toy(); model = new_model(d, 14, 'cpu')
    fitted = fit(model, d, 'cpu', 123)
    model.eval()
    with torch.no_grad(): z, _ = model.encode(d['x'], d['edges'])
    pred = model.decode(z, d['val_items']).argmax(1).numpy()
    assert abs(f1_score(d['val_y'], pred, labels=range(3), average='macro', zero_division=0) - fitted['selected_validation_quality']) < 1e-12


def test_node_deletion_keeps_evaluation_and_does_not_mutate_original():
    d = toy(); old = d['x'].clone(); a = withdraw(d, {'targets': [1, 4]})
    assert torch.equal(d['x'], old)
    assert torch.equal(a['val_items'], d['val_items'])
    assert not a['x'][[1, 4]].any()
    assert not torch.isin(a['edges'], torch.tensor([1, 4])).any()
    assert a['train_items'].tolist() == [0, 2, 3, 5, 6, 7, 8]


def test_relation_audit_detects_copies_and_false_negative_withdrawal():
    d = dict(x=torch.zeros(6, 3), train_positive=np.array([[1, 4]]),
        train_negative=np.array([[2, 3]]), positives=np.array([[0, 3], [1, 4], [1, 3], [0, 4]]),
        val_positive=np.array([[1, 3]]), test_positive=np.array([[0, 4]]),
        val_items=torch.tensor([[1, 3], [2, 4]]), test_items=torch.tensor([[0, 4], [2, 5]]))
    req = {'targets': [[0, 3]]}; d['edges'] = build_edges(np.empty((0, 2), dtype=int), d['train_positive'], 6)
    audit_relation(d, req)
    wrong = dict(d, edges=torch.cat([d['edges'], torch.tensor([[3], [1]])], 1))
    with pytest.raises(AssertionError): audit_relation(wrong, req)
    wrong = dict(d, train_negative=np.array([[0, 3]]))
    with pytest.raises(AssertionError): audit_relation(wrong, req)


def test_reference_targets_share_sets_and_training_only_balance():
    n = 200; ids = torch.arange(100); labels = torch.tensor([0] * 90 + [1] * 10)
    d = dict(task='node', x=torch.zeros(n, 3), train_items=ids, train_y=labels, node_labels=torch.zeros(n))
    components = {'ranks': {'Full': list(range(20)), 'Structure': list(reversed(range(20)))}}
    req = {'targets': [1, 2]}
    a = select_refs('Full-old', components, d, req, 0)
    assert a == select_refs('Full-input', components, d, req, 0)
    assert len(a['primary']) == 2 and len(a['secondary']) == 8
    b = select_refs('Class-balanced-train', components, d, req, 0)
    chosen = b['primary'] + b['secondary']
    assert sum(int(labels[v]) == 1 for v in chosen) == 5
    assert set(chosen) <= set(ids.tolist()) - {1, 2}
    d['node_labels'][100:] = 999
    assert b == select_refs('Class-balanced-train', components, d, req, 0)


def test_all_search_families_have_nine_distinct_candidates():
    from experiments.knowledge_state_maintenance.runtime import FAMILIES
    for family in FAMILIES:
        assert len(grid(family)) == 9
        assert len({tuple(sorted(c.items())) for c in grid(family)}) == 9


def test_macro_f1_includes_zero_prediction_classes():
    y = np.array([0, 0, 1, 1, 2, 2]); p = np.array([0, 0, 0, 0, 0, 0])
    assert class_metrics(y, p, 3)['f1'].mean() == f1_score(y, p, average='macro', zero_division=0)


def test_aggregation_does_not_treat_reference_repeats_as_bases(tmp_path, monkeypatch):
    import pandas as pd
    import experiments.knowledge_state_maintenance.report as report
    monkeypatch.setattr(report, 'OUT', tmp_path); (tmp_path / 'tables').mkdir()
    rows = [dict(method='A', split=s, base_seed=b, request_seed=q, repeat=r, quality=s)
        for s in range(5) for b in range(2) for q in range(2) for r in range(3)]
    splits, summary = report.aggregate(pd.DataFrame(rows), ['method'], 'test')
    assert len(splits) == 5 and summary.iloc[0].quality_count == 5
    assert summary.iloc[0].quality_mean == 2
