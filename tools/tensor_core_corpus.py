#!/usr/bin/env python3
"""Freeze an independent typed Kernel IR corpus for tensor-core recognition.

The generator uses Emmy's typed nodes directly.  It never calls Emmy's
renderer, emitter, or compiler, and every positive case has a plain NumPy
float64 reference.  Corpus bytes are content-addressed by ``manifest.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from emmy.compiler.dtype import F32
from emmy.compiler.graph import Graph
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.kernel.ir import (
    Accum, Assign, CpAsyncCommit, CpAsyncCopy, CpAsyncWait, KernelOp, Load, Loop,
    Smem, StridedLoop, Sync, Tile, Write,
)
from emmy.compiler.ir.stmt.leaves import Init
from emmy.compiler.tensor import Tensor

VERSION = 'tensor-core-corpus-v2'
RTL, ATOL = 1e-3, 1e-3


def canonical(value):
    # Emmy's published IR dumps use the Stmt constructor-repr the deserializer
    # intentionally accepts; the generator uses the same required typed path.
    return json.dumps(value, sort_keys=True, separators=(',', ':'), default=repr).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def save_arrays(path, arrays):
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, array in sorted(arrays.items()):
            data = io.BytesIO()
            np.lib.format.write_array(data, np.ascontiguousarray(array), allow_pickle=False)
            item = zipfile.ZipInfo(name + '.npy', date_time=(1980, 1, 1, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(item, data.getvalue())


def v(name):
    return Var(name)


def lit(value):
    return Literal(int(value), 'int')


def add(a, b):
    return BinaryExpr('+', a, b)


def sub(a, b):
    return BinaryExpr('-', a, b)


def mul(a, b):
    return BinaryExpr('*', a, b)


def div(a, b):
    return BinaryExpr('/', a, b)


def mod(a, b):
    return BinaryExpr('%', a, b)


def load(name, source, index):
    return Load(name, source, tuple(index))


def product(name, left, right):
    return Assign(name, ElementwiseImpl('multiply'), (left, right), F32)


def fold(name, value, axis='k'):
    return Accum(name, value, ElementwiseImpl('add'), F32, axes=(axis,), base=None)


class Case:
    def __init__(self, ident, expected, m, n, k, *, staged=False,
                 a_transpose=False, b_transpose=False, split=1, atomic=False,
                 warps=None, tail=False, missing_post_sync=False, same_buffer=False,
                 nonaffine=False, unrelated=False, epilogue=False):
        self.ident, self.expected = ident, expected
        self.m, self.n, self.k = m, n, k
        self.staged = staged
        self.a_transpose, self.b_transpose = a_transpose, b_transpose
        self.split, self.atomic = split, atomic
        self.warps = warps
        self.tail = tail
        self.missing_post_sync = missing_post_sync
        self.same_buffer = same_buffer
        self.nonaffine = nonaffine
        self.unrelated = unrelated
        self.epilogue = epilogue

    def tile(self):
        if self.warps is not None:
            # A forced warp count keeps its atom grid at the natural 16x8
            # tile; the extra launch warp is an independent near miss.
            return 16, 8, self.warps
        if self.tail:
            return 16, 8, 1
        if self.split > 1 or self.warps is not None:
            return 16, 8, (self.warps or 1)
        # Dedicated positives span multiple independently factorable CTA
        # tiles: 1x1, 2x1, 1x2, 2x2, 2x4, and 2x8 atoms.
        # Keep the frozen v1 atom grids: one warp per CTA for these samples.
        # This produces 1x1, 2x1, 1x2, 2x2, 1x4, and 1x16 CTA grids without
        # changing any reference values or tolerances.
        possibilities = ((16, 8), (32, 8), (16, 16), (32, 16),
                         (16, 32), (32, 32), (64, 32), (32, 64))
        for bm, bn in possibilities:
            if self.m % bm == 0 and self.n % bn == 0:
                return bm, bn, (bm // 16) * (bn // 8)
        return 16, 8, 1

    def build(self):
        bm, bn, warps = self.tile()
        atoms_m, atoms_n = bm // 16, bn // 8
        mtiles, ntiles = (self.m + bm - 1) // bm, (self.n + bn - 1) // bn
        block = warps * 32
        axes = [Axis('cta_m', mtiles), Axis('cta_n', ntiles),
                Axis('warp', warps), Axis('lane', 32)]
        if self.split > 1:
            axes.insert(0, Axis('split', self.split))
        grid = mtiles * ntiles * self.split

        g = div(v('lane'), lit(4))
        t = mod(v('lane'), lit(4))
        if self.warps is not None and self.warps > atoms_m * atoms_n:
            # Every launched warp maps to a legal cell family, but only the
            # first atom's warps have the required one-atom-per-warp role.
            warp_m = mod(v('warp'), lit(atoms_m))
            warp_n = mod(v('warp'), lit(atoms_n))
        else:
            warp_m = div(v('warp'), lit(atoms_n))
            warp_n = mod(v('warp'), lit(atoms_n))
        row0 = add(mul(v('cta_m'), lit(bm)), mul(warp_m, lit(16)))
        col0 = add(mul(v('cta_n'), lit(bn)), mul(warp_n, lit(8)))
        coords = {
            'c0': (add(row0, g), add(col0, mul(t, lit(2)))),
            'c1': (add(row0, g), add(add(col0, mul(t, lit(2))), lit(1))),
            'c2': (add(add(row0, g), lit(8)), add(col0, mul(t, lit(2)))),
            'c3': (add(add(row0, g), lit(8)), add(add(col0, mul(t, lit(2))), lit(1))),
        }
        a_row0 = add(row0, g)
        a_row1 = add(add(row0, g), lit(8))
        b_col0 = add(col0, mul(t, lit(2)))
        b_col1 = add(b_col0, lit(1))
        if self.tail:
            used = ('c0', 'c1')
        else:
            used = ('c0', 'c1', 'c2', 'c3')

        acc = [f'acc_{name}' for name in used]
        body = [Init(name, 0.0, dtype=F32) for name in acc]
        a_shape = (self.k, self.m) if self.a_transpose else (self.m, self.k)
        b_shape = (self.n, self.k) if self.b_transpose else (self.k, self.n)
        a_source = 'lhs' if not self.same_buffer else 'lhs'
        b_source = 'lhs' if self.same_buffer else 'rhs'

        def a_index(row, kk):
            return (kk, row) if self.a_transpose else (row, kk)

        def b_index(kk, col):
            if self.nonaffine:
                # A data-dependent gather is representable in IR and affine in
                # the loop coordinate only if a compiler linearizes mod.
                col = add(col, mod(mul(col, lit(3)), lit(5)))
            return (col, kk) if self.b_transpose else (kk, col)

        if self.split > 1:
            kk = add(mul(v('split'), lit(self.k // self.split)), v('k'))
        else:
            kk = v('k')

        if self.staged:
            body.extend([
                Smem('_a_smem', (bm, self.k), 'float', 16),
                Smem('_b_smem', (self.k, bn), 'float', 16),
                StridedLoop(Axis('a_copy', bm * self.k), start=lit(0), step=lit(block), seed=True,
                            body=(CpAsyncCopy('_a_smem', (div(add(v('a_copy'), v('tid')), lit(self.k)), mod(add(v('a_copy'), v('tid')), lit(self.k))),
                                              a_source,
                                              a_index(add(mul(v('cta_m'), lit(bm)), div(add(v('a_copy'), v('tid')), lit(self.k))),
                                                      mod(add(v('a_copy'), v('tid')), lit(self.k))), 4),)),
                StridedLoop(Axis('b_copy', self.k * bn), start=lit(0), step=lit(block), seed=True,
                            body=(CpAsyncCopy('_b_smem', (div(add(v('b_copy'), v('tid')), lit(bn)), mod(add(v('b_copy'), v('tid')), lit(bn))),
                                              b_source,
                                              b_index(div(add(v('b_copy'), v('tid')), lit(bn)),
                                                      add(mul(v('cta_n'), lit(bn)), mod(add(v('b_copy'), v('tid')), lit(bn)))), 4),)),
                CpAsyncCommit(), CpAsyncWait(0), Sync(0),
            ])
            compute_loads = {
                'a0': load('a0', '_a_smem', (add(mul(warp_m, lit(16)), g), kk)),
                'a1': load('a1', '_a_smem', (add(add(mul(warp_m, lit(16)), g), lit(8)), kk)),
                'b0': load('b0', '_b_smem', (kk, add(mul(warp_n, lit(8)), mul(t, lit(2))))),
                'b1': load('b1', '_b_smem', (kk, add(add(mul(warp_n, lit(8)), mul(t, lit(2))), lit(1)))),
            }
        else:
            compute_loads = {
                'a0': load('a0', a_source, a_index(a_row0, kk)),
                'a1': load('a1', a_source, a_index(a_row1, kk)),
                'b0': load('b0', b_source, b_index(kk, b_col0)),
                'b1': load('b1', b_source, b_index(kk, b_col1)),
            }
        if self.unrelated:
            compute_body = [load('u0', a_source, (v('k'), g)), fold('acc_c0', 'u0')]
        else:
            compute_body = [compute_loads['a0'], compute_loads['a1'], compute_loads['b0'], compute_loads['b1']]
            for index, name in enumerate(used):
                left = 'a0' if index < 2 else 'a1'
                right = 'b0' if index in (0, 2) else 'b1'
                compute_body.extend([product('prod_' + name, left, right), fold('acc_' + name, 'prod_' + name)])
        body.extend([Loop(Axis('k', self.k // self.split), body=tuple(compute_body), unroll=True, seed=False)])
        if self.staged and not self.missing_post_sync:
            body.append(Sync(0))
        if self.epilogue:
            body.append(Assign('epilogue_junk', ElementwiseImpl('add'), ('a0', 'b0'), F32))
            values = {name: 'epilogue_junk' for name in used}
        else:
            values = {name: 'acc_' + name for name in used}
        for name in used:
            body.append(Write('result', coords[name], values[name], value_dtype=F32, atomic=self.atomic))
        # An extra warp deliberately has no role.  The Tile still launches it,
        # so launch factorization, not an accidental duplicate, must reject it.

        graph = Graph()
        graph.add_node(InputOp(), [], Tensor('lhs', a_shape, F32), node_id='lhs')
        graph.add_node(InputOp(), [], Tensor('rhs', b_shape, F32), node_id='rhs')
        graph.inputs.extend(['lhs', 'rhs'])
        graph.add_node(KernelOp(body=(Tile(tuple(axes), tuple(body), block_threads=block),),
                                name='kernel'), ['lhs', 'rhs'],
                       Tensor('result', (self.m, self.n), F32), node_id='result')
        rng = np.random.default_rng(0x5EED + sum(map(ord, self.ident)))
        lhs = rng.uniform(-1.5, 1.5, a_shape).astype('float32')
        rhs = (rng.uniform(-1.5, 1.5, b_shape).astype('float32') if not self.same_buffer else lhs)
        lhs += np.arange(a_shape[0], dtype=np.float32)[:, None] * np.float32(.003)
        if not self.same_buffer:
            rhs += np.arange(rhs.shape[1], dtype=np.float32)[None, :] * np.float32(.005)
        if self.same_buffer:
            reference = lhs.astype(np.float64) @ lhs.astype(np.float64)
        else:
            a = lhs.astype(np.float64).T if self.a_transpose else lhs.astype(np.float64)
            b = rhs.astype(np.float64).T if self.b_transpose else rhs.astype(np.float64)
            reference = a @ b
        # A negative's expected output is diagnostic only; Plan cases are executed.
        order = ['rhs', 'lhs', 'result'] if sum(map(ord, self.ident)) % 2 else ['lhs', 'rhs', 'result']
        meta = {'id': self.ident, 'rtol': RTL, 'atol': ATOL,
                'graph_inputs': ['lhs', 'rhs'], 'graph_outputs': ['result'],
                'kernels': [{'node': 'result', 'entry': 'kernel', 'arg_order': order,
                             'grid': [grid, 1, 1], 'block': [block, 1, 1],
                             'smem_bytes': (bm * self.k + self.k * bn) * 4 if self.staged else 0,
                             'zero_outputs': ['result'] if self.atomic else [],
                             'args': [{'name': 'lhs', 'dtype': 'f32', 'shape': list(a_shape)},
                                      {'name': 'rhs', 'dtype': 'f32', 'shape': list(b_shape)},
                                      {'name': 'result', 'dtype': 'f32',
                                       'shape': [self.m, self.n], 'output': True}]}]}
        return graph, meta, {'lhs': lhs, 'rhs': rhs}, {'result': np.asarray(reference, dtype=np.float32)}


def cases():
    return [
        Case('p01-direct-16x8', 'Plan', 16, 8, 8),
        Case('p02-direct-32x16', 'Plan', 32, 16, 16),
        Case('p03-direct-64x32', 'Plan', 64, 32, 16),
        Case('p04-staged-32x16', 'Plan', 32, 16, 16, staged=True),
        Case('p05-staged-64x32', 'Plan', 64, 32, 16, staged=True),
        Case('p06-staged-32x64', 'Plan', 32, 64, 16, staged=True),
        Case('p07-b-transpose-32x16', 'Plan', 32, 16, 16, b_transpose=True),
        Case('p08-a-transpose-16x16', 'Plan', 16, 16, 8, a_transpose=True),
        Case('p09-direct-32x8', 'Plan', 32, 8, 8),
        Case('p10-staged-16x32', 'Plan', 16, 32, 8, staged=True),
        Case('p11-direct-16x32', 'Plan', 16, 32, 8),
        Case('p12-staged-b-transpose', 'Plan', 32, 16, 16, staged=True, b_transpose=True),
        Case('p13-direct-48x24', 'Plan', 48, 24, 24),
        Case('p14-staged-48x24', 'Plan', 48, 24, 24, staged=True),
        Case('n01-same-buffer', 'NoMatch', 16, 16, 16, same_buffer=True),
        Case('n02-nonaffine-gather', 'NoMatch', 16, 8, 8, nonaffine=True),
        Case('n03-unrelated-reduction', 'NoMatch', 16, 8, 8, unrelated=True),
        Case('n04-unsupported-epilogue', 'NoMatch', 16, 8, 8, epilogue=True),
        Case('n05-partial-collective', 'UnsupportedMatch', 16, 8, 8, staged=True, missing_post_sync=True),
        Case('n06-split-without-atomics', 'UnsupportedMatch', 16, 8, 16, split=2),
        Case('n07-split-atomic', 'UnsupportedMatch', 16, 8, 16, split=2, atomic=True),
        Case('n08-launch-mismatch', 'UnsupportedMatch', 16, 8, 8, warps=2),
        Case('n09-tail-partial', 'UnsupportedMatch', 8, 8, 8, tail=True),
    ]


def generate(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    records = []
    for case in cases():
        folder = destination / case.ident
        folder.mkdir()
        graph, meta, inputs, expected = case.build()
        (folder / 'case.ir.json').write_bytes(canonical(graph.to_dict()) + b'\n')
        (folder / 'meta.json').write_bytes(canonical(meta) + b'\n')
        save_arrays(folder / 'inputs.npz', inputs)
        save_arrays(folder / 'expected.npz', expected)
        files = {path.name: sha(path.read_bytes()) for path in sorted(folder.iterdir())}
        record = {'id': case.ident, 'expected': case.expected,
                  'version': VERSION, 'rtol': RTL, 'atol': ATOL, 'files': files}
        (folder / 'case.json').write_bytes(canonical(record) + b'\n')
        records.append(record)
    manifest = {'version': VERSION, 'rtol': RTL, 'atol': ATOL, 'cases': records}
    (destination / 'manifest.json').write_bytes(canonical(manifest) + b'\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    manifest = generate(args.destination)
    for record in manifest['cases']:
        print(record['id'], record['expected'], sha(canonical(record)))


if __name__ == '__main__':
    main()
