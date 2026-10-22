#!/usr/bin/env python3
"""Run the synthetic one-atom MMA through CUDA and check all 128 cells."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tf32_emitter import emit_and_oracle, lane_map


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arch', default='sm_89')
    parser.add_argument('--port', type=int, default=48704)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(0xA51C)
    a = rng.uniform(-2, 2, (16, 8)).astype(np.float32)
    b = rng.uniform(-2, 2, (8, 8)).astype(np.float32)
    a += np.arange(16, dtype=np.float32)[:, None] * np.float32(.031)
    b += np.arange(8, dtype=np.float32)[None, :] * np.float32(.047)
    emitted = emit_and_oracle(a, b)
    ptx_path = args.out / 'synthetic.ptx'
    ptx_path.write_text(emitted.ptx)
    records = {'arch': args.arch, 'ptx_sha256': hashlib.sha256(emitted.ptx.encode()).hexdigest(),
               'mma': 'mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32'}
    ptxas = '/opt/cuda/bin/ptxas'
    cubin = args.out / f'synthetic-{args.arch}.cubin'
    built = subprocess.run([ptxas, f'-arch={args.arch}', str(ptx_path), '-o', str(cubin)],
                           capture_output=True, text=True)
    if built.returncode:
        raise SystemExit(built.stderr)
    sass = subprocess.run(['/opt/cuda/bin/cuobjdump', '-sass', str(cubin)],
                          capture_output=True, text=True, check=True).stdout
    mma_lines = [line.strip() for line in sass.splitlines() if 'HMMA' in line]
    records['sass_hmma_count'] = len(mma_lines)
    records['sass_hmma'] = mma_lines
    if len(mma_lines) != 3 or not all('HMMA' in line and 'TF32' in line for line in mma_lines):
        raise SystemExit('expected three TF32 HMMA instructions in SASS')
    server = subprocess.Popen([sys.executable, str(ROOT/'tools/local_runner.py'),
                               '--cache', str(args.out/'cache'), '--port', str(args.port),
                               '--arch', args.arch],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        import time, urllib.request
        url = f'http://127.0.0.1:{args.port}'
        for _ in range(100):
            try:
                with urllib.request.urlopen(url+'/health', timeout=1) as response:
                    records['health'] = json.load(response)
                break
            except OSError:
                time.sleep(.1)
        else:
            raise RuntimeError('local runner did not start')
        def post(path, body):
            req = urllib.request.Request(url+path, data=json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=40) as response:
                return json.load(response)
        compiled = post('/compile', {'ptx': emitted.ptx, 'arch': args.arch})
        records['compile'] = compiled
        if not compiled.get('ok'):
            raise RuntimeError(compiled)
        args_payload = [
            {'name': 'A', 'dtype': 'f32', 'shape': [16, 8], 'data': a.reshape(-1).tolist()},
            {'name': 'B', 'dtype': 'f32', 'shape': [8, 8], 'data': b.reshape(-1).tolist()},
            {'name': 'C', 'dtype': 'f32', 'shape': [16, 8], 'output': True},
        ]
        run = post('/run', {'cubin_id': compiled['cubin_id'], 'entry': 'synthetic_mma',
                            'grid': [1, 1, 1], 'block': [32, 1, 1], 'smem_bytes': 0,
                            'zero_outputs': [], 'args': args_payload, 'max_return': 4096})
        records['run'] = run
        if not run.get('ok'):
            raise RuntimeError(run)
        got = np.asarray(run['outputs']['C']['values'], dtype=np.float32).reshape(16, 8)
        expected = emitted.oracle.astype(np.float32)
        diff = np.abs(got.astype(np.float64) - expected.astype(np.float64))
        records['max_abs_error'] = float(diff.max())
        records['mismatched_cells'] = int((diff != 0).sum())
        records['all_128_cells'] = bool(diff.size == 128 and np.isfinite(got).all())
        # The MMA's internal dot rounding differs from NumPy's f64 dot by a
        # few f32 units in the last place.  Bind the hardware to the
        # independent CPU model with a tight absolute bound; do not claim bit
        # identity for an unspecified internal MMA rounding order.
        rounding_bound = 1e-6
        rounding_ok = bool((diff <= rounding_bound).all())
        records['rounding_bound'] = rounding_bound
        records['rounding_ok'] = rounding_ok
        # Lane ownership is separate from arithmetic: every one of the 128 C
        # cells must be owned by exactly one lane's four-cell fragment map.
        owners = {}
        for lane in range(32):
            for row, col in lane_map(lane)['c']:
                owners[(row, col)] = owners.get((row, col), 0) + 1
        lane_partition = bool(len(owners) == 128 and set(owners.values()) == {1} and
                               set(owners) == {(r, c) for r in range(16) for c in range(8)})
        lane_fragments = True
        for lane in range(32):
            for (row, col), frag in zip(lane_map(lane)['c'], emitted.lanes[lane]['c']):
                if abs(float(got[row, col]) - float(np.float32(frag))) > rounding_bound:
                    lane_fragments = False
        records['lane_partition_exact'] = lane_partition
        records['lane_fragments_within_bound'] = lane_fragments
        records['pass'] = (records['all_128_cells'] and lane_partition and
                           lane_fragments and rounding_ok)
    finally:
        server.terminate()
        try: server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill(); server.wait()
    (args.out/'result.json').write_text(json.dumps(records, indent=2)+'\n')
    print(json.dumps(records, indent=2))
    return int(not records['pass'])


if __name__ == '__main__':
    raise SystemExit(main())
