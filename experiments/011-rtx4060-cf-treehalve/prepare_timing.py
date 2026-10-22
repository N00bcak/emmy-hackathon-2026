#!/usr/bin/env python3
"""Prepare frozen paired-timing PTX/cubins for selected published cases."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = ROOT / 'emmy-hackathon-2026/problems'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, default=ROOT / 'compile.py')
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--case', action='append', default=None)
    args = parser.parse_args()
    cases = args.case or ['m01_matmul_512', 'm02_matmul_rect', 'f02_row_softmax',
                          'a01_attention', 'a02_attention_128']
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    rows = {}
    for case in cases:
        source = PROBLEMS / case
        folder = out / 'prepared' / case
        folder.mkdir(parents=True)
        ir = next(source.glob('*.ir.json'))
        shutil.copyfile(ir, folder / ir.name)
        shutil.copyfile(source / 'meta.json', folder / 'meta.json')
        expected = source / 'expected.npz'
        if expected.exists():
            shutil.copyfile(expected, folder / 'expected.npz')
        meta = json.loads((folder / 'meta.json').read_text())
        kernels = []
        for kernel in meta['kernels']:
            entry = kernel['entry']
            arms = {}
            for label, compiler in [('candidate', args.candidate), ('baseline', args.baseline)]:
                ptx = folder / f'{label}-{entry}.ptx'
                cubin = folder / f'{label}-{entry}.cubin'
                result = subprocess.run([sys.executable, str(compiler), str(folder / ir.name)],
                                        capture_output=True, text=True, check=True, timeout=30)
                ptx.write_text(result.stdout, encoding='ascii')
                subprocess.run(['ptxas', '-arch=sm_89', str(ptx), '-o', str(cubin)],
                               check=True, timeout=30)
                arms[label] = {'ptx': str(ptx), 'ptx_sha256': sha(ptx),
                               'cubin': str(cubin), 'cubin_sha256': sha(cubin)}
            kernels.append({'entry': entry, 'ir': str(folder / ir.name),
                            'ir_sha256': sha(folder / ir.name), **arms})
        rows[case] = {'case_dir': str(folder), 'kernels': kernels}
    manifest = {'candidate_source': str(args.candidate.resolve()),
                'candidate_sha256': sha(args.candidate),
                'baseline_source': str(args.baseline.resolve()),
                'baseline_sha256': sha(args.baseline),
                'arch': 'sm_89', 'protocol': 'cuda-graph-paired-v2', 'cases': rows}
    (out / 'prepared/manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(out / 'prepared/manifest.json')


if __name__ == '__main__':
    main()
