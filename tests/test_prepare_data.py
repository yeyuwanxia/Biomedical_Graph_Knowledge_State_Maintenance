"""Small synthetic checks; no downloaded data or experiment outputs are used."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("prepare_data", ROOT / "scripts" / "prepare_data.py")
prepare_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_data)


class LocalInputTests(unittest.TestCase):
    def test_wrong_input_is_rejected_before_destination_is_created(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.bz2"
            source.write_bytes(b"wrong input")
            target = root / "data" / "target.bz2"
            with self.assertRaises(ValueError):
                prepare_data.stage_raw_file(source, target, "0" * 64)
            self.assertFalse(target.exists())

    def test_existing_different_input_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, target = root / "source.bz2", root / "target.bz2"
            source.write_bytes(b"expected input")
            target.write_bytes(b"existing input")
            with self.assertRaises(ValueError):
                prepare_data.stage_raw_file(source, target, prepare_data.sha256(source))
            self.assertEqual(target.read_bytes(), b"existing input")

    def test_valid_input_is_staged_and_can_be_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, target = root / "source.bz2", root / "data" / "target.bz2"
            source.write_bytes(b"expected input")
            digest = prepare_data.sha256(source)
            self.assertEqual(prepare_data.stage_raw_file(source, target, digest), digest)
            self.assertEqual(prepare_data.stage_raw_file(target, target, digest), digest)
            self.assertEqual(target.read_bytes(), source.read_bytes())


class FrozenGraphProtocolTests(unittest.TestCase):
    def test_duplicates_self_loops_isolates_and_raw_degree_features(self):
        import torch
        from data.hetionet import FEATURE_SCHEMA, load_hetionet_full_nosource

        raw = {
            "metanode_kinds": ["A", "B"],
            "nodes": [{"kind": "A", "identifier": 1}, {"kind": "A", "identifier": 2},
                      {"kind": "B", "identifier": 1}, {"kind": "B", "identifier": 2}],
            "edges": [
                {"source_id": ["A", 1], "target_id": ["B", 1], "kind": "first"},
                {"source_id": ["A", 1], "target_id": ["B", 1], "kind": "duplicate"},
                {"source_id": ["B", 1], "target_id": ["A", 2], "kind": "next"},
                {"source_id": ["A", 1], "target_id": ["A", 1], "kind": "self"},
            ],
        }
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder) / "Hetionet"
            base.mkdir()
            (base / "hetionet-v1.0.json").write_text(json.dumps(raw), encoding="utf-8")
            data, metadata = load_hetionet_full_nosource(folder, download=False)
            self.assertEqual(data.num_nodes, 4)
            self.assertEqual(tuple(data.x.shape), (4, 20))
            self.assertEqual(metadata["feature_schema"], FEATURE_SCHEMA)
            self.assertEqual(metadata["num_isolated_nodes_before_drop"], 1)
            self.assertEqual(metadata["removed_isolated_nodes"], 0)
            self.assertEqual(data.y.tolist(), [0, 0, 1, 1])
            self.assertEqual(data.edge_index.tolist(), [[0, 1, 2, 2], [2, 2, 0, 1]])
            self.assertEqual(metadata["num_edges_directed"], 3)
            torch.testing.assert_close(data.x[:, 3], torch.log1p(torch.tensor([0., 1., 2., 0.])))
            torch.testing.assert_close(data.x[:, 4], torch.log1p(torch.tensor([2., 0., 1., 0.])))
            self.assertTrue(bool(torch.isfinite(data.x).all()))
            cached, _ = load_hetionet_full_nosource(folder, download=False)
            torch.testing.assert_close(cached.x, data.x, rtol=0, atol=0)
            self.assertTrue(torch.equal(cached.edge_index, data.edge_index))


if __name__ == "__main__":
    unittest.main()
