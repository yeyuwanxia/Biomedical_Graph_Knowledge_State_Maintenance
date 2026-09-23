#!/usr/bin/env python3
"""Verify code identities without importing PyTorch, reading datasets, or running experiments."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strict-code-only', action='store_true', help='Also reject files created after extraction; run before data preparation/tests.')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'PACKAGE_MANIFEST.json').read_text())
    failures = []
    for relative, entry in manifest['files'].items():
        path = ROOT / relative
        if not path.is_file() or sha(path) != entry['sha256']:
            failures.append(relative)
    frozen = json.loads((ROOT / 'EXPERIMENT_SOURCE_MANIFEST.json').read_text())
    for relative, expected in frozen['source_hashes'].items():
        path = ROOT / relative
        if not path.is_file() or sha(path) != expected:
            failures.append('frozen:' + relative)
    for relative in manifest['files']:
        if relative.endswith('.py'):
            ast.parse((ROOT / relative).read_text(), filename=relative)
    if args.strict_code_only:
        permitted = set(manifest['files']) | {'PACKAGE_MANIFEST.json', 'SHA256SUMS'}
        actual = {str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file()}
        failures.extend('unexpected:' + p for p in sorted(actual - permitted))
    if failures:
        raise SystemExit('Package check failed:\n' + '\n'.join(failures))
    print(f"Verified {len(manifest['files'])} delivered files; {len(frozen['source_hashes'])} experiment source files match the delivered manifest.")
    if args.strict_code_only:
        print('The delivered file set contains code, configuration and documentation only.')


if __name__ == '__main__':
    main()
