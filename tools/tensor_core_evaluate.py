#!/usr/bin/env python3
"""Execute Plan cases through the unchanged local runner and f64 references."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emmy.compiler.graph import Graph
import tensor_core
from tf32_emitter import simulate_matmul, tf32_rna


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', type=Path, default=ROOT/'compile.py')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arch', default='sm_89')
    parser.add_argument('--port', type=int, default=48705)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    compiler = args.compiler.resolve()
    server = subprocess.Popen([sys.executable, str(ROOT/'tools/local_runner.py'),
                               '--cache', str(args.out/'cache'), '--port', str(args.port),
                               '--arch', args.arch],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    records = {'arch': args.arch, 'compiler_sha256': hashlib.sha256(compiler.read_bytes()).hexdigest(),
               'cases': [], 'benchmark': False}
    def post(path, body):
        req = urllib.request.Request(f'http://127.0.0.1:{args.port}'+path,
                                     data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.load(response)
    try:
        for _ in range(100):
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/health', timeout=1) as response:
                    records['health'] = json.load(response)
                break
            except OSError:
                time.sleep(.1)
        roots = [ROOT/'emmy-hackathon-2026/problems/m01_matmul_512']
        roots += [p for p in sorted((ROOT/'experiments/tensor-core-v1/corpus-v2').iterdir()) if p.is_dir()]
        for folder in roots:
            meta = json.loads((folder/'meta.json').read_text())
            kernel = meta['kernels'][0]
            ir = (next(folder.glob('*.ir.json')) if list(folder.glob('*.ir.json'))
                  else folder/'case.ir.json')
            graph = Graph.from_dict(json.loads(ir.read_text()))
            decision = tensor_core.classify(graph, graph.nodes[kernel['node']], kernel)
            row = {'case': folder.name, 'classification': decision.verdict,
                   'reason': decision.reason, 'ir_sha256': hashlib.sha256(ir.read_bytes()).hexdigest()}
            if decision.verdict != 'Plan':
                records['cases'].append(row)
                continue
            compiled = subprocess.run([sys.executable, str(compiler), str(ir)],
                                      capture_output=True, text=True, check=True)
            ptx = compiled.stdout
            (args.out/f'{folder.name}.ptx').write_text(ptx)
            row['ptx_sha256'] = hashlib.sha256(ptx.encode()).hexdigest()
            row['mma_count'] = ptx.count('mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32')
            c = post('/compile', {'ptx': ptx, 'arch': args.arch})
            row['compile'] = c
            if not c.get('ok'):
                row['pass'] = False
                records['cases'].append(row)
                continue
            # Metadata args are declaration-ordered; the ABI is arg_order.  The
            # runner binds parameters positionally, so submit in that order.
            declared = {arg['name']: arg for arg in kernel['args']}
            run_args = []
            supplied = {}
            if (folder/'inputs.npz').exists():
                with np.load(folder/'inputs.npz', allow_pickle=False) as inputs:
                    supplied = {name: inputs[name] for name in inputs.files}
            for name in kernel['arg_order']:
                arg = declared[name]
                spec = {'name': arg['name'], 'dtype': arg['dtype'], 'shape': arg['shape']}
                if arg.get('output'):
                    spec['output'] = True
                elif name in supplied:
                    values = np.asarray(supplied[name], dtype=np.float32)
                    if values.size != int(np.prod(arg['shape'], dtype=np.int64)):
                        raise ValueError(f'corpus input {name!r} does not match metadata shape')
                    spec['data'] = values.reshape(-1).tolist()
                elif arg.get('seed') is not None:
                    spec['seed'] = arg['seed']
                elif arg.get('data') is not None:
                    spec['data'] = arg['data']
                elif arg.get('scalar'):
                    spec['scalar'] = True
                    spec['value'] = arg['value']
                else:
                    raise ValueError(f'input {name!r} has no data, seed, or scalar value')
                run_args.append(spec)
            a_name = decision.plan.a_buffer
            b_name = decision.plan.b_buffer
            if a_name in supplied and b_name in supplied:
                a = supplied[a_name].astype(np.float64)
                b = supplied[b_name].astype(np.float64)
            else:
                def generated(name):
                    arg = declared[name]
                    if arg.get('data') is not None:
                        return np.asarray(arg['data'], dtype=np.float32).reshape(arg['shape'])
                    return np.random.default_rng(arg['seed']).standard_normal(arg['shape']).astype(np.float32)
                a = generated(a_name).astype(np.float64)
                b = generated(b_name).astype(np.float64)
            if decision.plan.a_transpose: a = a.T
            if decision.plan.b_transpose: b = b.T
            reference = (a @ b).astype(np.float32)
            # The integrated run uses explicit data, not a seed, so it is
            # reproducible without depending on the runner's RNG implementation.
            payload = {**kernel, 'args': run_args, 'cubin_id': c['cubin_id'],
                       'warmup': 0, 'iters': 1, 'max_return': 2**22}
            run = post('/run', payload)
            row['run_ok'] = bool(run.get('ok'))
            if run.get('ok'):
                output = run['outputs'][decision.plan.output]
                got = np.asarray(output['values'], dtype=np.float32).reshape(decision.plan.m, decision.plan.n)
                diff = np.abs(got.astype(np.float64) - reference.astype(np.float64))
                tol = meta['atol'] + meta['rtol'] * np.abs(reference.astype(np.float64))
                row['max_abs_error'] = float(diff.max())
                row['failing_elements'] = int((diff > tol).sum())
                row['all_cells'] = bool(diff.size == decision.plan.m * decision.plan.n)
                row['pass'] = bool(row['all_cells'] and not row['failing_elements'])
                simulated = simulate_matmul(a, b)
                simulation_diff = np.abs(got.astype(np.float64) - simulated.astype(np.float64))
                row['cpu_simulation_max_abs_error'] = float(simulation_diff.max())
                # A K=512 accumulation naturally differs from this independent
                # chunk model by a few e-4 in absolute value, so normalize by
                # the operand-product scale.  One-pass TF32 is ~1e-4 relative
                # to the reference; the three-term path is bounded at 4e-6.
                scale = max(1.0, float(np.max(np.abs(reference))))
                row['cpu_simulation_max_rel_error'] = float(simulation_diff.max() / scale)
                one_pass = np.asarray(
                    np.vectorize(tf32_rna, otypes=[np.float32])(a.astype(np.float32)).astype(np.float64) @
                    np.vectorize(tf32_rna, otypes=[np.float32])(b.astype(np.float32)).astype(np.float64),
                    dtype=np.float32)
                one_pass_diff = np.abs(got.astype(np.float64) - one_pass.astype(np.float64))
                row['one_pass_tf32_max_rel_error'] = float(one_pass_diff.max() / scale)
                row['three_term_beats_one_pass'] = bool(
                    row['cpu_simulation_max_rel_error'] < row['one_pass_tf32_max_rel_error'])
                row['cpu_simulation_bound'] = 4e-6
                row['cpu_simulation_pass'] = bool(row['cpu_simulation_max_rel_error'] <= 4e-6)
            else:
                row['run'] = run
                row['pass'] = False
            records['cases'].append(row)
            (args.out/'result.json').write_text(json.dumps(records, indent=2)+'\n')
        plan_rows = [row for row in records['cases'] if row['classification'] == 'Plan']
        records['all_plan_passed'] = all(row.get('pass') and row.get('cpu_simulation_pass')
                                         for row in plan_rows)
    finally:
        server.terminate()
        try: server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill(); server.wait()
    (args.out/'result.json').write_text(json.dumps(records, indent=2)+'\n')
    print(json.dumps({'arch': args.arch, 'all_plan_passed': records.get('all_plan_passed'),
                      'cases': [{k: r.get(k) for k in ('case','classification','pass',
                               'max_abs_error','failing_elements','mma_count',
                               'cpu_simulation_max_abs_error','cpu_simulation_max_rel_error',
                               'one_pass_tf32_max_rel_error','three_term_beats_one_pass',
                               'cpu_simulation_pass')}
                                for r in records['cases']]}, indent=2))
    return int(not records.get('all_plan_passed'))


if __name__ == '__main__':
    raise SystemExit(main())
