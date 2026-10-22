#!/usr/bin/env python3
"""Record classifications, PTX/cubin hashes, and SASS for Plan cases."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emmy.compiler.graph import Graph
import tensor_core


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arch', action='append', default=[])
    parser.add_argument('--ptxas', default='/opt/cuda/bin/ptxas')
    parser.add_argument('--cuobjdump', default='/opt/cuda/bin/cuobjdump')
    args = parser.parse_args()
    arches = args.arch or ['sm_89', 'sm_120']
    args.out.mkdir(parents=True, exist_ok=False)
    folders = [ROOT / 'emmy-hackathon-2026/problems/m01_matmul_512']
    folders += [p for p in sorted((ROOT / 'experiments/tensor-core-v1/corpus-v2').iterdir())
                if p.is_dir()]
    rows = []
    for folder in folders:
        meta = json.loads((folder / 'meta.json').read_text())
        kernel = meta['kernels'][0]
        ir = next(folder.glob('*.ir.json'), None) or folder / 'case.ir.json'
        graph = Graph.from_dict(json.loads(ir.read_text()))
        decision = tensor_core.classify(graph, graph.nodes[kernel['node']], kernel)
        row = {'case': folder.name, 'classification': decision.verdict,
               'reason': decision.reason, 'ir_sha256': sha(ir.read_bytes())}
        if decision.verdict != tensor_core.PLAN:
            rows.append(row)
            continue
        compiled = subprocess.run([sys.executable, str(ROOT / 'compile.py'), str(ir)],
                                  capture_output=True, text=True, check=True)
        ptx = compiled.stdout
        row['ptx_sha256'] = sha(ptx.encode())
        row['mma_count'] = ptx.count('mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32')
        row['plans'] = {'m': decision.plan.m, 'n': decision.plan.n,
                        'k': decision.plan.k, 'active_warps': decision.plan.active_warps,
                        'block_m': decision.plan.block_m, 'block_n': decision.plan.block_n,
                        'staging': decision.plan.staging}
        row['arches'] = {}
        for arch in arches:
            target = args.out / f'{folder.name}.{arch}.ptx'
            cubin = args.out / f'{folder.name}.{arch}.cubin'
            target.write_text(ptx)
            assembled = subprocess.run([args.ptxas, f'-arch={arch}', str(target), '-o', str(cubin)],
                                       capture_output=True, text=True)
            if assembled.returncode:
                raise SystemExit(f'{folder.name} {arch}: {assembled.stderr}')
            sass = subprocess.run([args.cuobjdump, '-sass', str(cubin)],
                                  capture_output=True, text=True, check=True).stdout
            lines = [line.strip() for line in sass.splitlines()
                     if 'HMMA' in line or 'MMA' in line]
            row['arches'][arch] = {'cubin_sha256': sha(cubin.read_bytes()),
                                   'mma_lines': lines}
            if not lines or not all('HMMA' in line and 'TF32' in line for line in lines):
                raise SystemExit(f'{folder.name} {arch}: TF32 MMA missing from SASS')
        rows.append(row)
    source_hashes = {name: sha((ROOT / name).read_bytes())
                     for name in ('compile.py', 'tensor_core.py', 'tf32_emitter.py')}
    corpus = ROOT / 'experiments/tensor-core-v1/corpus-v2'
    report = {'architectures': arches, 'compiler_sha256': source_hashes['compile.py'],
              'source_sha256': source_hashes,
              'corpus_manifest_sha256': sha((corpus / 'manifest.json').read_bytes()),
              'baseline_sha256': sha((ROOT / 'experiments/tensor-core-v1/baseline/baseline.json').read_bytes()),
              'baseline_v2_sha256': sha((ROOT / 'experiments/tensor-core-v1/baseline/baseline-v2.json').read_bytes()),
              'ptxas_version': subprocess.run([args.ptxas, '--version'], capture_output=True,
                                              text=True, check=True).stdout.splitlines()[0],
              'cases': rows, 'benchmark': False}
    (args.out / 'evidence.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'compiler_sha256': report['compiler_sha256'],
                      'plans': sum(r['classification'] == tensor_core.PLAN for r in rows),
                      'classifications': {r['case']: r['classification'] for r in rows}}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
