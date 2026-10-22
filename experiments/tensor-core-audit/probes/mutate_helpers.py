"""Helpers for adversarial typed-IR mutations. Read-only w.r.t. compiler files."""
from __future__ import annotations
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.tensor_core_corpus import (  # noqa: E402
    Case, add, div, lit, mod, mul, sub, v,
)
from emmy.compiler.graph import Graph  # noqa: E402
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var  # noqa: E402
from emmy.compiler.ir.kernel.ir import CpAsyncCopy  # noqa: E402


def replace_body(node, body):
    return dataclasses.replace(node, op=dataclasses.replace(node.op, body=tuple(body)))


def replace_tile(node, tile):
    return replace_body(node, (tile,))


def mutate_statements(statement, fn):
    """Bottom-up typed statement mutation; fn(stmt) may return a replacement."""
    changes = {}
    for field in dataclasses.fields(statement):
        if field.name in ('body', 'else_body', 'combine_states'):
            nested = getattr(statement, field.name)
            if nested:
                changes[field.name] = tuple(mutate_statements(x, fn) for x in nested)
    if changes:
        statement = dataclasses.replace(statement, **changes)
    return fn(statement)


def replace_tile_statements(tile, fn):
    return dataclasses.replace(
        tile, body=tuple(mutate_statements(x, fn) for x in tile.body))


def copies(body):
    found = []
    def walk(statements):
        for statement in statements:
            if type(statement).__name__ == 'CpAsyncCopy':
                found.append(statement)
            for field in ('body', 'else_body', 'combine_states'):
                nested = getattr(statement, field, None)
                if nested:
                    walk(nested)
    walk(body)
    return found


def save_case(folder, graph, meta, inputs=None, expected=None):
    import io, zipfile
    import numpy as np
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'), default=repr).encode()
    (folder / 'case.ir.json').write_bytes(canonical(graph.to_dict()) + b'\n')
    (folder / 'meta.json').write_bytes(canonical(meta) + b'\n')
    if inputs is not None:
        with zipfile.ZipFile(folder / 'inputs.npz', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for name, array in sorted(inputs.items()):
                data = io.BytesIO()
                np.lib.format.write_array(data, np.ascontiguousarray(array), allow_pickle=False)
                item = zipfile.ZipInfo(name + '.npy', date_time=(1980, 1, 1, 0, 0, 0))
                item.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(item, data.getvalue())
    if expected is not None:
        with zipfile.ZipFile(folder / 'expected.npz', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for name, array in sorted(expected.items()):
                data = io.BytesIO()
                np.lib.format.write_array(data, np.ascontiguousarray(array), allow_pickle=False)
                item = zipfile.ZipInfo(name + '.npy', date_time=(1980, 1, 1, 0, 0, 0))
                item.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(item, data.getvalue())
