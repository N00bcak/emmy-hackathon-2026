"""Construct semantically explicit typed-IR probes for the agent-2 audit.

This module is read-only with respect to the compiler.  It builds small
contractions directly from Emmy typed nodes, so every probe can be reasoned
about without treating the production recognizer as an oracle.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from emmy.compiler.dtype import F32
from emmy.compiler.graph import Graph
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var
from emmy.compiler.ir.kernel.ir import (
    Accum, Assign, CpAsyncCommit, CpAsyncCopy, CpAsyncWait, KernelOp, Load, Loop,
    Smem, StridedLoop, Sync, Tile, Write,
)
from emmy.compiler.ir.stmt.leaves import Init
from emmy.compiler.tensor import Tensor


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


def cond(a, op, b):
    return BinaryExpr(op, a, b)


def choose(c, yes, no):
    return TernaryExpr(c, yes, no)


def load(name, source, index):
    return Load(name, source, tuple(index))


def prod(name, left, right):
    return Assign(name, ElementwiseImpl('multiply'), (left, right), F32)


def fold(name, value, axis='k'):
    return Accum(name, value, ElementwiseImpl('add'), F32, axes=(axis,), base=None)


def base_body(m, n, k, *, a_source='A', b_source='B', output='C', a_shape=None,
              b_shape=None, a_transpose=False, b_transpose=False, a_row=None,
              b_col=None, a_k=None, b_k=None, axes=('cta_m', 'cta_n', 'warp', 'lane'),
              atom_grid=(1, 1), tile=(16, 8, 1)):
    """Build one 16x8 atom per warp with explicit (c0,c1,c2,c3) cells."""
    bm, bn, warps = tile
    cta_m, cta_n, warp_axis, lane_axis = axes
    g = div(v(lane_axis), lit(4))
    tt = mod(v(lane_axis), lit(4))
    warp_m = div(v(warp_axis), lit(atom_grid[1])) if atom_grid[1] != 1 else v(warp_axis)
    warp_n = mod(v(warp_axis), lit(atom_grid[1])) if atom_grid[1] != 1 else lit(0)
    row0 = add(mul(v(cta_m), lit(bm)), mul(warp_m, lit(16)))
    col0 = add(mul(v(cta_n), lit(bn)), mul(warp_n, lit(8)))
    r0, r1 = add(row0, g), add(add(row0, g), lit(8))
    c0, c1 = add(col0, mul(tt, lit(2))), add(add(col0, mul(tt, lit(2))), lit(1))
    vr0 = a_row(r0) if a_row else r0
    vr1 = a_row(r1) if a_row else r1
    vc0 = b_col(c0) if b_col else c0
    vc1 = b_col(c1) if b_col else c1
    ak0 = a_k if a_k is not None else v('k')
    bk0 = b_k if b_k is not None else v('k')
    ai0 = (ak0, vr0) if a_transpose else (vr0, ak0)
    ai1 = (ak0, vr1) if a_transpose else (vr1, ak0)
    bi0 = (vc0, bk0) if b_transpose else (bk0, vc0)
    bi1 = (vc1, bk0) if b_transpose else (bk0, vc1)
    fragments = ((r0, c0, 'a0', 'b0'), (r0, c1, 'a0', 'b1'),
                 (r1, c0, 'a1', 'b0'), (r1, c1, 'a1', 'b1'))
    names = ('c0', 'c1', 'c2', 'c3')
    inits = [Init('acc_' + x, 0.0, dtype=F32) for x in names]
    loads = [load('a0', a_source, ai0), load('a1', a_source, ai1),
             load('b0', b_source, bi0), load('b1', b_source, bi1)]
    compute = list(loads)
    for name, (_, _, aa, bb) in zip(names, fragments):
        compute.extend([prod('p_' + name, aa, bb), fold('acc_' + name, 'p_' + name)])
    loop = Loop(Axis('k', k), body=tuple(compute), unroll=True, seed=False)
    writes = []
    for name, (rr, cc, _, _) in zip(names, fragments):
        writes.append(Write(output, (rr, cc), 'acc_' + name, value_dtype=F32))
    body = tuple(inits + [loop] + writes)
    grid_m, grid_n = m // bm, n // bn
    tile_stmt = Tile(tuple(Axis(x, e) for x, e in zip(
        axes, (grid_m, grid_n, warps, 32))), body, block_threads=32 * warps)
    g = Graph()
    g.add_node(InputOp(), [], Tensor(a_source, tuple(a_shape or ((k, m) if a_transpose else (m, k))), F32), node_id=a_source)
    g.add_node(InputOp(), [], Tensor(b_source, tuple(b_shape or ((n, k) if b_transpose else (k, n))), F32), node_id=b_source)
    g.inputs.extend([a_source, b_source])
    g.add_node(KernelOp(body=(tile_stmt,), name='kernel_' + output),
               [a_source, b_source], Tensor(output, (m, n), F32), node_id=output)
    g.outputs.append(output)
    return g, tile_stmt


def metadata(ident, m, n, k, *, grid=1, block=32, a_shape=None, b_shape=None,
             a='A', b='B', output='C', order=(0, 1, 2), smem=0, zero=()):
    a_shape = a_shape or (m, k)
    b_shape = b_shape or (k, n)
    args = [{'name': a, 'dtype': 'f32', 'shape': list(a_shape)},
            {'name': b, 'dtype': 'f32', 'shape': list(b_shape)},
            {'name': output, 'dtype': 'f32', 'shape': [m, n], 'output': True}]
    args = [args[i] for i in order]
    return {'id': ident, 'rtol': 1e-3, 'atol': 1e-3,
            'graph_inputs': [a, b], 'graph_outputs': [output],
            'kernels': [{'node': output, 'entry': 'kernel_' + output,
                         'arg_order': [x['name'] for x in args],
                         'grid': [grid, 1, 1], 'block': [block, 1, 1],
                         'smem_bytes': smem, 'zero_outputs': list(zero), 'args': args}]}


def kernel_node(graph, name='C'):
    return graph.nodes[name]


def replace_op(node, **changes):
    return dataclasses.replace(node, op=dataclasses.replace(node.op, **changes))


def replace_tile(node, tile):
    return replace_op(node, body=(tile,))


def map_statements(statement, fn):
    changes = {}
    for field in dataclasses.fields(statement):
        if field.name in ('body', 'else_body', 'combine_states'):
            nested = getattr(statement, field.name)
            if nested:
                changes[field.name] = tuple(map_statements(x, fn) for x in nested)
    if changes:
        statement = dataclasses.replace(statement, **changes)
    return fn(statement)


def map_tile_body(tile, fn):
    return dataclasses.replace(tile, body=tuple(map_statements(x, fn) for x in tile.body))


def insert_before_first(tile, predicate, additions):
    body = []
    inserted = False
    for statement in tile.body:
        if not inserted and predicate(statement):
            body.extend(additions)
            inserted = True
        body.append(statement)
    if not inserted:
        raise AssertionError('insertion point absent')
    return dataclasses.replace(tile, body=tuple(body))
