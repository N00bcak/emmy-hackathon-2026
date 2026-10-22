#!/usr/bin/env python3
"""Durable typed Kernel IR correctness variants for the direct PTX compiler.

Every case is built from typed Emmy nodes and checked by a plain NumPy oracle.
The emitted metadata uses the evaluator ABI, so the cases run through the same
CUDA worker as published or generated research cases.
"""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
from emmy.compiler.dtype import F32, F64, I32, U32
from emmy.compiler.graph import Graph
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var
from emmy.compiler.ir.kernel.ir import (
    Accum, Cond, CpAsyncCommit, CpAsyncCopy, CpAsyncWait, KernelOp, Load, Loop,
    Smem, StridedLoop, Sync, Tile, TreeHalve, WarpShuffle, Write,
)
from emmy.compiler.ir.stmt.leaves import Assign, Let, Select, SelectBranch
from emmy.compiler.tensor import Tensor

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'emmy-variant-suite-v1'
NP = {'f32': np.float32, 'f64': np.float64, 'i32': np.int32}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'


def graph_canonical(value):
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=repr) + '\n'


def var(name):
    return Var(name)


def lit(value):
    return Literal(value, 'int' if type(value) is int else 'float')


def op(kind, left, right):
    return BinaryExpr(kind, left, right)


def assign(name, kind, *args, dtype=F32):
    return Assign(name, ElementwiseImpl(kind), tuple(args), dtype)


def load(name, buffer, index, dtype=F32):
    return Load(name, buffer, tuple(index), dtype=dtype)


def store(output, index, value, dtype=F32):
    return Write(output, tuple(index), value, value_dtype=dtype)


class Case:
    def __init__(self, name, block):
        self.name, self.block = name, block
        self.graph = Graph()
        self.input_names, self.output_names = [], []
        self.arrays, self.kernels, self.expected = {}, [], {}

    def input(self, name, array, dtype=F32):
        array = np.ascontiguousarray(array, dtype=NP[dtype.name])
        self.graph.add_node(InputOp(), [], Tensor(name, array.shape, dtype), node_id=name)
        self.graph.inputs.append(name)
        self.arrays[name] = array
        self.input_names.append(name)
        return name

    def output(self, name, shape, dtype=F32):
        self.graph.add_node(KernelOp(body=(), name='placeholder'), [], Tensor(name, tuple(shape), dtype), node_id=name)
        self.graph.outputs.append(name)
        self.output_names.append(name)
        return name

    def kernel(self, name, node, inputs, args, body, *, smem=0, grid=None):
        if len(body) != 1 or type(body[0]).__name__ != 'Tile':
            raise AssertionError('variant needs exactly one top-level Tile')
        tile = body[0]
        cells = math.prod(int(a.extent.as_static()) for a in tile.axes)
        self.graph.nodes[node].op = KernelOp(body=tuple(body), name=name)
        dtype = self.graph.nodes[node].outputs[0].dtype.name
        shape = [int(d.as_static()) if hasattr(d, 'as_static') else int(d)
                 for d in self.graph.nodes[node].outputs[0].shape]
        rows = [{'name': n, 'dtype': self.graph.nodes[n].outputs[0].dtype.name,
                 'shape': list(self.arrays[n].shape)} for n in inputs]
        rows.append({'name': node, 'dtype': dtype, 'shape': list(shape), 'output': True})
        self.kernels.append({
            'node': node, 'entry': name, 'arg_order': list(args), 'args': rows,
            'grid': list(grid or [math.ceil(cells / self.block), 1, 1]),
            'block': [self.block, 1, 1], 'smem_bytes': smem,
            'zero_outputs': [], 'final': True,
        })

    def finish(self, description, tolerances=(1e-5, 1e-6)):
        output = self.output_names[0]
        dtype = self.graph.nodes[output].outputs[0].dtype.name
        self.expected[output] = np.asarray(self.expected[output], dtype=NP[dtype])
        meta = {'id': self.name, 'graph_inputs': list(self.input_names),
                'graph_outputs': list(self.output_names), 'rtol': float(tolerances[0]),
                'atol': float(tolerances[1]), 'kernels': self.kernels}
        return {'name': self.name, 'description': description, 'graph': self.graph.to_dict(),
                'meta': meta, 'arrays': self.arrays, 'expected': self.expected}


def scalar_abi():
    c = Case('scalar_abi', 8)
    x = c.input('x', np.linspace(-1.5, 2.5, 23))
    out = c.output('out', (23,))
    body = [load('v', x, (var('i'),)), assign('scaled', 'multiply', 'v', 'scale'),
            assign('shifted', 'add', 'scaled', 'bias'),
            store(out, (var('i'),), 'shifted')]
    c.kernel('k_scalar', out, [x], ['out', x, 'scale', 'bias'],
             [Tile((Axis('i', 23),), tuple(body), block_threads=8)], grid=[3, 1, 1])
    c.kernels[0]['args'] += [
        {'name': 'scale', 'dtype': 'f32', 'shape': [], 'scalar': True, 'value': 1.75},
        {'name': 'bias', 'dtype': 'f32', 'shape': [], 'scalar': True, 'value': -0.5}]
    c.expected[out] = c.arrays[x] * np.float32(1.75) - np.float32(0.5)
    return c.finish('scalar-by-value ABI, non-unit tail, and extra-CTA masking')


def integer_address():
    c = Case('integer_address', 8)
    x = c.input('x', np.arange(64, dtype=np.int32).reshape(8, 8), I32)
    out = c.output('out', (8, 8), I32)
    body = [load('v', x, (var('i'), var('j')), I32),
            assign('copy', 'copy', 'v', dtype=I32),
            store(out, (var('i'), var('j')), 'copy', I32)]
    c.kernel('k_int_copy', out, [x], [x, out],
             [Tile((Axis('i', 8), Axis('j', 8)), tuple(body), block_threads=8)])
    c.expected[out] = c.arrays[x]
    return c.finish('integer copy and signed 2-D address flattening', tolerances=(0, 0))


def f64_values():
    c = Case('f64_values', 8)
    x = c.input('x', np.array([0.25, 1.5, -2.25, 3.0], dtype=np.float64), F64)
    out = c.output('out', (4,), F64)
    body = [load('v', x, (var('i'),), F64),
            assign('sum', 'add', 'v', 'delta', dtype=F64),
            store(out, (var('i'),), 'sum', F64)]
    c.kernel('k_f64', out, [x], [x, out, 'delta'],
             [Tile((Axis('i', 4),), tuple(body), block_threads=8)])
    c.kernels[0]['args'].append(
        {'name': 'delta', 'dtype': 'f64', 'shape': [], 'scalar': True, 'value': 0.125})
    c.expected[out] = c.arrays[x] + np.float64(0.125)
    return c.finish('f64 scalar literal, loads, arithmetic, and stores', tolerances=(0, 0))


def flatten_2d():
    c = Case('flatten_2d', 32)
    x = c.input('x', np.arange(12, dtype=np.float32).reshape(4, 3))
    out = c.output('out', (4, 3))
    body = [load('v', x, (var('i'), var('j'))),
            assign('twice', 'multiply', 'v', 'two'),
            store(out, (var('i'), var('j')), 'twice')]
    c.kernel('k_flat', out, [x], [x, out, 'two'],
             [Tile((Axis('i', 4), Axis('j', 3)), tuple(body), block_threads=32)], grid=[1, 1, 1])
    c.kernels[0]['args'].append(
        {'name': 'two', 'dtype': 'f32', 'shape': [], 'scalar': True, 'value': 2.0})
    c.expected[out] = c.arrays[x] * np.float32(2)
    return c.finish('non-unit 2-D Tile flattening and partial CTA')


def reduction_partial():
    c = Case('reduction_partial', 32)
    values = (np.arange(127, dtype=np.float32) * np.float32(0.37) - np.float32(9.0)).astype(np.float32)
    x = c.input('x', values)
    out = c.output('out', (), F32)
    # 24 active lanes cover the whole reduction in 24-element strides; eight
    # lanes of the first CTA and every lane of the extra CTA are inactive.
    loop = StridedLoop(Axis('k', 127), var('co'), lit(24),
                       (load('v', x, (var('k'),)),
                        Accum('best', 'v', ElementwiseImpl('maximum'), F32, axes=('k',))))
    shuffle = WarpShuffle(('best',), ('other',),
                          (assign('best', 'maximum', 'best', 'other'),), 32, F32)
    body = [loop, shuffle,
            Smem('best_smem', (1,), 'float', 4),
            Cond(op('==', var('lane'), lit(0)),
                 (store('best_smem', (var('warp'),), 'best'),), ()),
            Sync(0),
            TreeHalve(('best_smem',), ('best',), ('other',),
                      (assign('best', 'maximum', 'best', 'other'),), 1, 'warp', F32),
            Cond(op('==', var('co'), lit(0)), (store(out, (), 'best'),), ())]
    c.kernel('k_max', out, [x], [x, out],
             [Tile((Axis('co', 24),), tuple(body), block_threads=32)], grid=[2, 1, 1])
    c.expected[out] = np.asarray(values.max())
    return c.finish('non-sum reduction with partial-CTA and extra-CTA masking', tolerances=(1e-6, 1e-6))


def transcendental():
    c = Case('transcendental', 8)
    x = c.input('x', np.array([.25, .5, 1, 2, 4, 8, 16, 32], dtype=np.float32))
    out = c.output('out', (8,))
    body = [load('v', x, (var('i'),)), assign('ln', 'log', 'v'),
            assign('th', 'tanh', 'ln'), store(out, (var('i'),), 'th')]
    c.kernel('k_transcendental', out, [x], [x, out],
             [Tile((Axis('i', 8),), tuple(body), block_threads=8)])
    c.expected[out] = np.tanh(np.log(c.arrays[x].astype(np.float64))).astype(np.float32)
    return c.finish('natural log and tanh', tolerances=(2e-5, 2e-5))


def casts_and_index_expr():
    c = Case('casts_index', 8)
    x = c.input('x', np.array([[-3.5, 2.25, 7.0, -0.5],
                               [1.0, 4.0, -9.0, 0.0]], dtype=np.float32))
    out = c.output('out', (2, 4))
    i, j = var('i'), var('j')
    body = [Let('fi', op('/', i, lit(2)), I32),
            Let('in_first', op('<', var('fi'), lit(1)), None),
            load('v', x, (i, j)),
            Let('zero', lit(0.0), F32),
            Let('nonnegative', op('>=', var('v'), lit(0.0)), None),
            assign('choice', 'where', 'nonnegative', 'v', 'zero'),
            store(out, (i, j), 'choice')]
    c.kernel('k_cast_index', out, [x], [x, out],
             [Tile((Axis('i', 2), Axis('j', 4)), tuple(body), block_threads=8)])
    c.expected[out] = np.where(c.arrays[x] >= 0, c.arrays[x], 0).astype(np.float32)
    return c.finish('floor division, predicate selection, and typed writes')


def conversion():
    c = Case('write_conversion', 8)
    x = c.input('x', np.array([.125, 1.25, -2.5, 4.0, 8.0, 16.0, 32.0, 64.0], dtype=np.float32))
    out = c.output('out', (8,), F64)
    body = [load('v', x, (var('i'),)), assign('wide', 'copy', 'v', dtype=F64),
            store(out, (var('i'),), 'wide', F64)]
    c.kernel('k_convert', out, [x], [x, out],
             [Tile((Axis('i', 8),), tuple(body), block_threads=8)])
    c.expected[out] = c.arrays[x].astype(np.float64)
    return c.finish('f32 to f64 conversion at the SSA and write boundaries', tolerances=(0, 0))


def select_and_ternary():
    c = Case('select_ternary', 4)
    x = c.input('x', np.array([1.25, -2.5, 3.75, -4.0], dtype=np.float32))
    out = c.output('out', (4,))
    i = var('i')
    body = [load('v', x, (i,)), Let('zero', lit(0.0), F32),
            Let('low', op('<', i, lit(2)), None),
            Let('high', op('>=', i, lit(2)), None),
            Select('selected', (SelectBranch('v', var('low')),
                                SelectBranch('zero', var('high')))),
            Let('ternary', TernaryExpr(op('<', i, lit(2)), var('v'), lit(0.0)), F32),
            assign('sum', 'add', 'selected', 'ternary'),
            store(out, (i,), 'sum')]
    c.kernel('k_select', out, [x], [x, out],
             [Tile((Axis('i', 4),), tuple(body), block_threads=4)])
    values = c.arrays[x]
    c.expected[out] = np.where(np.arange(4) < 2, values, np.float32(0)) * np.float32(2)
    return c.finish('Select catch-all and TernaryExpr predicate', tolerances=(0, 0))


def integer_division():
    c = Case('integer_division', 8)
    x = c.input('x', np.array([-17, -9, -1, 0, 1, 9, 17, 33], dtype=np.int32), I32)
    out = c.output('out', (8,), I32)
    body = [load('v', x, (var('i'),), I32), Let('three', lit(-3), I32),
            assign('q', 'floor_divide', 'v', 'three', dtype=I32),
            assign('r', 'remainder', 'v', 'three', dtype=I32),
            assign('sum', 'add', 'q', 'r', dtype=I32),
            store(out, (var('i'),), 'sum', I32)]
    c.kernel('k_int_div', out, [x], [x, out],
             [Tile((Axis('i', 8),), tuple(body), block_threads=8)])
    values = c.arrays[x].astype(np.int64)
    q = np.floor_divide(values, -3)
    r = values - q * (-3)
    c.expected[out] = (q + r).astype(np.int32)
    return c.finish('signed floor division and floor remainder', tolerances=(0, 0))


def bitcast_roundtrip():
    c = Case('bitcast_roundtrip', 8)
    x = c.input('x', np.array([1.25, -2.5, 0.0, -0.0, 3.5, -7.0, 1e-8, -1e8], dtype=np.float32))
    out = c.output('out', (8,))
    body = [load('v', x, (var('i'),)), assign('bits', 'bitcast', 'v', dtype=I32),
            assign('back', 'bitcast', 'bits', dtype=F32),
            store(out, (var('i'),), 'back')]
    c.kernel('k_bitcast', out, [x], [x, out],
             [Tile((Axis('i', 8),), tuple(body), block_threads=8)])
    c.expected[out] = c.arrays[x]
    return c.finish('same-width bitcast round trip', tolerances=(0, 0))


def partial_warp_reduction(kind):
    name = f'partial_warp_{kind}'
    c = Case(name, 6)
    values = np.array([-9.0, -7.0, -5.0, -3.0, -2.0, -1.0], dtype=np.float32)
    x = c.input('x', values)
    out = c.output('out', (1,))
    combine = 'maximum' if kind == 'max' else 'add'
    body = [load('v', x, (var('i'),)),
            WarpShuffle(('v',), ('other',),
                        (assign('v', combine, 'v', 'other'),), 8, F32),
            Cond(op('==', var('lane'), lit(0)),
                 (store(out, (lit(0),), 'v'),), ())]
    c.kernel(f'k_{kind}', out, [x], [x, out],
             [Tile((Axis('i', 6),), tuple(body), block_threads=6)])
    c.expected[out] = np.asarray(values.max() if kind == 'max' else values.sum())
    return c.finish('partial-warp shuffle with absent peers', tolerances=(0, 0))


def vector_copy(aligned):
    name = 'vector_copy_aligned' if aligned else 'vector_copy_scalar'
    c = Case(name, 32)
    x = c.input('x', np.arange(129, dtype=np.float32))
    out = c.output('out', (32,))
    shift = 0 if aligned else 1
    source = op('+', op('*', var('tid'), lit(4)), lit(shift))
    destination = op('*', var('tid'), lit(4))
    body = [Smem('scratch', (128,), 'float', 16),
            CpAsyncCopy('scratch', (destination,), x, (source,), nbytes=16),
            CpAsyncCommit(), CpAsyncWait(0), Sync(0),
            load('a', 'scratch', (destination,)),
            load('b', 'scratch', (op('+', destination, lit(1)),)),
            load('c', 'scratch', (op('+', destination, lit(2)),)),
            load('d', 'scratch', (op('+', destination, lit(3)),)),
            assign('ab', 'add', 'a', 'b'), assign('cd', 'add', 'c', 'd'),
            assign('sum', 'add', 'ab', 'cd'),
            store(out, (var('tid'),), 'sum')]
    c.kernel('k_vec', out, [x], [x, out],
             [Tile((Axis('tid', 32),), tuple(body), block_threads=32)],
             smem=512, grid=[1, 1, 1])
    groups = c.arrays[x][:128].reshape(32, 4)
    c.expected[out] = groups[:, 0] + groups[:, 1] + groups[:, 2] + groups[:, 3] if aligned else \
        c.arrays[x][1:129].reshape(32, 4).sum(axis=1)
    return c.finish('aligned 16-byte vector copy' if aligned else 'unaligned scalar-copy fallback',
                    tolerances=(2e-6, 2e-6))


SPECS = [scalar_abi, integer_address, f64_values, flatten_2d, reduction_partial,
         transcendental, casts_and_index_expr, conversion, select_and_ternary,
         integer_division, bitcast_roundtrip, lambda: partial_warp_reduction('sum'),
         lambda: partial_warp_reduction('max'), lambda: vector_copy(True),
         lambda: vector_copy(False)]


def build(destination):
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise SystemExit(f'refusing to overwrite nonempty {destination}')
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for factory in SPECS:
        case = factory()
        folder = destination / case['name']
        folder.mkdir()
        graph = case['graph']
        Graph.from_dict(graph)
        ir_name = case['meta']['kernels'][0]['entry'] + '.ir.json'
        (folder / ir_name).write_text(graph_canonical(graph))
        (folder / 'meta.json').write_text(canonical(case['meta']))
        np.savez(folder / 'inputs.npz', **case['arrays'])
        np.savez(folder / 'expected.npz', **case['expected'])
        rows.append({'name': case['name'], 'description': case['description'],
                     'ir': ir_name, 'source_sha256': sha(ROOT / 'compile.py'),
                     'files': {p.name: sha(p) for p in sorted(folder.iterdir())}})
    manifest = {'schema': SCHEMA, 'cases': rows}
    (destination / 'manifest.json').write_text(canonical(manifest))
    return manifest


def compile_case(folder, compiler):
    compiler = Path(compiler).resolve()
    rows = []
    for ir in sorted(Path(folder).glob('*.ir.json')):
        ptx = ir.with_suffix('.ptx')
        result = subprocess.run([sys.executable, str(compiler), str(ir)],
                                capture_output=True, text=True, timeout=30)
        if result.returncode:
            raise RuntimeError(f'{ir}: compiler failed: {result.stderr}')
        ptx.write_text(result.stdout)
        cubin = ir.with_suffix('.cubin')
        assembled = subprocess.run(['ptxas', '-arch=sm_89', str(ptx), '-o', str(cubin)],
                                   capture_output=True, text=True, timeout=30)
        if assembled.returncode:
            raise RuntimeError(f'{ptx}: ptxas failed: {assembled.stderr}')
        rows.append({'ir': ir.name, 'ptx': ptx.name, 'cubin': cubin.name,
                     'ptx_sha256': sha(ptx), 'cubin_sha256': sha(cubin)})
    return rows


def gpu_specs(root):
    return [{'id': row['name'], 'variant': 'variant', 'case_dir': str(Path(root) / row['name']),
             'cubins': [str(next((Path(root) / row['name']).glob('*.cubin')).resolve())]}
            for row in json.loads((Path(root) / 'manifest.json').read_text())['cases']]


def run_gpu(root, out, repetitions=3, width=1):
    from tools.research_gpu import Driver, run_job
    specs = gpu_specs(root)
    batches = []
    driver = Driver()
    try:
        for start in range(0, len(specs), 8):
            subset = specs[start:start + 8]
            result = run_job({'mode': 'correctness', 'cases': subset, 'width': width,
                              'repetitions': repetitions, 'out': str(out)}, driver=driver)
            batches.append(result)
            if not result['ok']:
                break
    finally:
        driver.close()
    report = {'schema': SCHEMA, 'gpu': driver.snapshot, 'batches': batches,
              'compiler_sha256': sha(ROOT / 'compile.py'),
              'all_ok': bool(batches) and all(row['ok'] for row in batches)}
    Path(out).write_text(canonical(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--compiler', type=Path, default=ROOT / 'compile.py')
    parser.add_argument('--run-gpu', action='store_true')
    parser.add_argument('--repetitions', type=int, default=3)
    args = parser.parse_args()
    manifest = build(args.destination)
    for row in manifest['cases']:
        row['artifacts'] = compile_case(args.destination / row['name'], args.compiler)
    (args.destination / 'manifest.json').write_text(canonical(manifest))
    if args.run_gpu:
        report = run_gpu(args.destination, args.destination / 'gpu-results.json',
                         repetitions=args.repetitions)
        print('GPU', 'PASS' if report['all_ok'] else 'FAIL')
        if not report['all_ok']:
            raise SystemExit(1)
    print(args.destination)


if __name__ == '__main__':
    main()
