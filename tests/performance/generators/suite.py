"""Frozen typed-IR builders and independent references for the scaled suite.

The published IRs define each family's schedule.  Scaling therefore rewrites
only shape-bearing fields in the typed graph and metadata, never the operation
or kernel decomposition.  References use NumPy directly in float64.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import fields
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CHALLENGE = ROOT / "emmy-hackathon-2026"
PROBLEMS = CHALLENGE / "problems"
HERE = Path(__file__).resolve().parents[1]
MANIFEST = HERE / "manifest.json"
VERSION = "perf-v1"
MAX_ELEMENTS = 2**26
RETURN_LIMIT = 2**22
DTYPES = {"f16": np.float16, "f32": np.float32, "f64": np.float64,
          "i32": np.int32, "i64": np.int64}

sys.path.insert(0, str(ROOT))
from tools.research_cases import canonical, digest, save_arrays, validate_case  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_manifest():
    return json.loads(MANIFEST.read_text())


def canonical_case(case, tier=None):
    """Return the manifest fields that define a frozen case identity."""
    return {k: case[k] for k in sorted(case) if k != "tier"}


def all_cases(tier=None):
    data = load_manifest()
    tiers = [tier] if tier else list(data["tiers"])
    return [(name, case) for name in tiers for case in data["tiers"][name]["cases"]]


def _dump(path, value, *, sort_keys=True):
    path.write_bytes((json.dumps(value, sort_keys=sort_keys, separators=(",", ":"),
                                allow_nan=False) + "\n").encode())


def _graph_dict(graph):
    """Use Emmy's serializer, then encode Stmt reprs like the published files."""
    return json.loads(json.dumps(graph.to_dict(), default=str, allow_nan=False))


def _published(path):
    return json.loads((PROBLEMS / path / "meta.json").read_text()), \
        next((PROBLEMS / path).glob("*.ir.json"))


def _tile_axes(graph):
    from emmy.compiler.ir.kernel.ir import KernelOp, Tile
    for node in graph.nodes.values():
        if isinstance(node.op, KernelOp):
            for stmt in node.op.body:
                if isinstance(stmt, Tile):
                    yield from stmt.axes


def _rescale_windows(stmt, old, new):
    """Recursively re-extent window parents, leaving local schedule factors alone."""
    from emmy.compiler.ir.axis import Axis
    from emmy.compiler.ir.kernel.ir import Loop, StridedLoop, Tile
    from emmy.compiler.dim import Dim
    if isinstance(stmt, Tile):
        axes = []
        for axis in stmt.axes:
            window = axis.window
            if window and window.parent and window.parent.name in new:
                parent = Axis(window.parent.name, Dim(new[window.parent.name]), window=window.parent.window)
                window = type(window)(parent=parent, base=window.base, bound=window.bound,
                                      partition=window.partition)
            axes.append(Axis(axis.name, axis.extent, window=window))
        return type(stmt)(tuple(axes), tuple(_rescale_windows(s, old, new) for s in stmt.body),
                          block_threads=stmt.block_threads, aux_threads=stmt.aux_threads,
                          raster_axes=stmt.raster_axes, raster_group=stmt.raster_group,
                          raster_orient=stmt.raster_orient)
    if isinstance(stmt, (Loop, StridedLoop)):
        return stmt
    body = getattr(stmt, "body", None)
    if body is not None:
        values = {f.name: getattr(stmt, f.name) for f in fields(stmt)}
        return type(stmt)(**{**values, "body": tuple(_rescale_windows(s, old, new) for s in body)})
    return stmt


def _rename_and_rescale(graph, names, shapes, extents=None, windows=None):
    """Change tensor extents and loop/tile extents while preserving the schedule."""
    from emmy.compiler.dim import Dim
    from emmy.compiler.ir.kernel.ir import KernelOp, Loop, StridedLoop, Tile
    for node in graph.nodes.values():
        node.outputs = tuple(type(t)(name=names.get(t.name, t.name),
                                    shape=tuple(Dim(s) for s in shapes[t.name])
                                    if t.name in shapes else t.shape,
                                    dtype=t.dtype)
                             for t in node.outputs)
    for node in graph.nodes.values():
        if not isinstance(node.op, KernelOp):
            continue
        def change(stmt):
            if isinstance(stmt, Tile):
                axes = tuple(type(a)(a.name,
                                     Dim(extents[a.name]) if extents and a.name in extents else a.extent,
                                     window=(type(a.window)(parent=type(a.window.parent)(a.window.parent.name, Dim(windows[a.window.parent.name]), window=a.window.parent.window), base=a.window.base, bound=a.window.bound, partition=a.window.partition) if a.window and a.window.parent and windows and a.window.parent.name in windows else a.window))
                             for a in stmt.axes)
                return type(stmt)(axes, tuple(change(s) for s in stmt.body),
                                  block_threads=stmt.block_threads,
                                  aux_threads=stmt.aux_threads,
                                  raster_axes=stmt.raster_axes,
                                  raster_group=stmt.raster_group,
                                  raster_orient=stmt.raster_orient)
            if isinstance(stmt, (Loop, StridedLoop)):
                axis = type(stmt.axis)(stmt.axis.name,
                                       Dim(extents[stmt.axis.name]) if extents and stmt.axis.name in extents
                                       else stmt.axis.extent, window=stmt.axis.window)
                if isinstance(stmt, StridedLoop):
                    return type(stmt)(axis, stmt.start, stmt.step,
                                      tuple(change(s) for s in stmt.body),
                                      unroll=stmt.unroll, end=stmt.end, seed=stmt.seed)
                return type(stmt)(axis, tuple(change(s) for s in stmt.body),
                                  unroll=stmt.unroll, seed=stmt.seed)
            body = getattr(stmt, "body", None)
            if body is not None:
                values = {f.name: getattr(stmt, f.name) for f in fields(stmt)}
                return type(stmt)(**{**values, "body": tuple(change(s) for s in body)})
            return stmt
        from emmy.compiler.ir.stmt import Body
        node.op = type(node.op)(name=node.op.name,
                                body=Body(tuple(change(s) for s in node.op.body)),
                                zero_delegated=node.op.zero_delegated, serial=node.op.serial)
    return graph


def case_shapes(case):
    """Return semantic input/output shapes, independent of generated payloads."""
    family, shape = case["family"], case["shape"]
    if family == "pointwise":
        n = shape[0]
        if case["anchor"] in ("p00_scalar_add", "p01_vector_add"):
            return {"x0": (n,), "x1": (n,)}, {"add": (n,)}
        if case["anchor"] == "p02_scale_shift":
            return {"x": (n,)}, {"add": (n,)}
        if case["anchor"] == "p03_relu_mul":
            return {"x": (n,), "y": (n,)}, {"mul": (n,)}
        raise ValueError("unknown pointwise anchor")
    if family == "reduction":
        return {"x": (shape[0],)}, {"sum_1": (1,)}
    if family == "row_sum":
        m, n = shape
        return {"x": (m, n)}, {"sum_1": (m,)}
    if family == "matmul":
        m, n, k = shape
        return {"x0": (m, k), "x1": (k, n)}, {"matmul": (m, n)}
    if family in ("rmsnorm", "l2norm"):
        m, n = shape
        inputs = {"x": (m, n)}
        if family == "rmsnorm":
            inputs["w"] = (n,)
        return inputs, {("mul_1" if family == "rmsnorm" else "div"): (m, n)}
    if family == "softmax":
        m, n = shape
        return {"x": (m, n)}, {"softmax": (m, n)}
    if family == "attention":
        seq, width = shape
        return ({"q": (seq, width), "kt": (width, seq), "v": (seq, width)},
                {"matmul_1": (seq, width)})
    raise ValueError(f"unknown family {family}")


def output_elements(case):
    """Largest number of values returned by a case."""
    _, outputs = case_shapes(case)
    return max(math.prod(shape) for shape in outputs.values())


def stable_softmax(x):
    z = x.astype(np.float64)
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def _reference(case, inputs):
    family = case["family"]
    if family == "pointwise":
        if case["anchor"] in ("p00_scalar_add", "p01_vector_add"):
            value = inputs["x0"].astype(np.float64) + inputs["x1"].astype(np.float64)
            return {"add": value.astype(np.float32)}
        if case["anchor"] == "p02_scale_shift":
            value = inputs["x"].astype(np.float64) * 2.0 + 1.0
            return {"add": value.astype(np.float32)}
        if case["anchor"] == "p03_relu_mul":
            value = (np.maximum(inputs["x"].astype(np.float64), 0.0)
                     * inputs["y"].astype(np.float64))
            return {"mul": value.astype(np.float32)}
        raise ValueError("unknown pointwise anchor")
    if family == "reduction":
        return {"sum_1": np.asarray([inputs["x"].astype(np.float64).sum()], dtype=np.float32)}
    if family == "row_sum":
        return {"sum_1": inputs["x"].astype(np.float64).sum(axis=1).astype(np.float32)}
    if family == "matmul":
        value = inputs["x0"].astype(np.float64) @ inputs["x1"].astype(np.float64)
        return {"matmul": value.astype(np.float32)}
    if family == "rmsnorm":
        x = inputs["x"].astype(np.float64)
        value = x / np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + 1e-5) * inputs["w"]
        return {"mul_1": value.astype(np.float32)}
    if family == "l2norm":
        x = inputs["x"].astype(np.float64)
        value = x / np.sqrt(np.sum(x ** 2, axis=-1, keepdims=True))
        return {"div": value.astype(np.float32)}
    if family == "softmax":
        return {"softmax": stable_softmax(inputs["x"]).astype(np.float32)}
    if family == "attention":
        scale = math.sqrt(case["shape"][1])
        scores = inputs["q"].astype(np.float64) @ inputs["kt"].astype(np.float64) / scale
        return {"matmul_1": (stable_softmax(scores) @ inputs["v"]).astype(np.float32)}
    raise ValueError(f"unknown family {family}")

FAMILY_SOURCE = {
    "pointwise": "p00_scalar_add", "reduction": "r01_array_sum",
    "row_sum": "r04_row_sum", "matmul": "m01_matmul_512",
    "rmsnorm": "f01_rmsnorm", "softmax": "f02_row_softmax",
    "l2norm": "f03_l2norm", "attention": "a01_attention",
}

ALIASES = {
    "pointwise": {"x": "x0", "y": "x1", "scale": "mul_c1", "shift": "add_c1",
                  "out": "add"},
    "reduction": {"x": "x", "out": "sum_1"},
    "row_sum": {"x": "x", "out": "sum_1"},
    "matmul": {"x0": "x0", "x1": "x1", "matmul": "matmul"},
    "rmsnorm": {"x": "x", "w": "w", "out": "mul_1"},
    "l2norm": {"x": "x", "out": "div"},
    "softmax": {"x": "x", "out": "softmax"},
    "attention": {"q": "q", "kt": "kt", "v": "v", "scale": "div_c1", "out": "matmul_1"},
}


def _scaled_metadata(source_meta, case, shapes):
    meta = json.loads(json.dumps(source_meta))
    meta["id"] = case["id"]
    meta["rtol"], meta["atol"] = case["tolerance"]
    seed_index = 0
    for kernel in meta["kernels"]:
        for arg in kernel["args"]:
            alias = arg["name"]
            if alias in shapes:
                arg["shape"] = list(shapes[alias])
                if not arg.get("output") and "data" not in arg:
                    arg["seed"] = int(case["seed"]) + seed_index
                    seed_index += 1
            elif arg["name"] == "mean_count":
                arg["data"] = [float(case["shape"][1])]
            elif arg["name"] == "div_c1":
                arg["data"] = [math.sqrt(case["shape"][1])]
            elif arg["name"] not in shapes and not arg.get("output") and "data" not in arg:
                raise ValueError(f"unmapped metadata constant {arg['name']}")
    return meta


def scaled_metadata(case):
    source_meta, _ = _published(case["anchor"])
    input_shapes, output_shapes = case_shapes(case)
    return _scaled_metadata(source_meta, case, {**input_shapes, **output_shapes})


def seeded_input_plan(case):
    """Return ``name -> (dtype, shape, seed)`` from the rewritten launch ABI."""
    input_shapes, _ = case_shapes(case)
    plan = {}
    for kernel in scaled_metadata(case)["kernels"]:
        for arg in kernel["args"]:
            if arg.get("output") or arg.get("scalar") or "seed" not in arg:
                continue
            dtype = arg.get("dtype", "f32")
            definition = (dtype, tuple(arg["shape"]), int(arg["seed"]))
            if arg["name"] in plan and plan[arg["name"]] != definition:
                raise ValueError(f"inconsistent seed definition for {arg['name']}")
            plan[arg["name"]] = definition
    expected = set(input_shapes)
    if set(plan) != expected:
        raise ValueError(f"seed plan/input mismatch: {sorted(set(plan) ^ expected)}")
    for name, (_, shape, _) in plan.items():
        if shape != tuple(input_shapes[name]):
            raise ValueError(f"seed plan shape mismatch for {name}")
    return plan


def materialize_seeded(plan):
    """Match the runner ABI exactly; never introduce a second draw convention."""
    return {name: np.ascontiguousarray(
        np.random.default_rng(seed).standard_normal(shape).astype(DTYPES[dtype]))
        for name, (dtype, shape, seed) in plan.items()}


def generate_case(case, out):
    source = case["anchor"]
    source_meta, source_ir = _published(source)
    from emmy.compiler.graph import Graph
    graph = Graph.from_dict(json.loads(source_ir.read_text()))
    input_shapes, output_shapes = case_shapes(case)
    shapes = {**input_shapes, **output_shapes}
    meta = _scaled_metadata(source_meta, case, shapes)
    plan = seeded_input_plan(case)
    inputs = materialize_seeded(plan)
    expected = _reference(case, inputs)
    if {name: value.shape for name, value in expected.items()} != output_shapes:
        raise ValueError("reference output shape mismatch")
    extents = {}
    windows = None
    if case["family"] == "reduction":
        extents = {"a0": case["shape"][0]}
    elif case["family"] in ("row_sum", "rmsnorm", "l2norm", "softmax"):
        # Row kernels have one loop for reduction and another for the output
        # pass. Leaving the second at the published width silently truncates
        # the scaled row; update every width-bearing axis.
        extents = {"a0": case["shape"][0]}
        for key in ("a1", "a2", "a3"):
            extents[key] = case["shape"][1]
    if case["family"] == "pointwise":
        n = case["shape"][0]
        extents["a0"] = math.ceil(n / case["vector_width"])
    elif case["family"] == "matmul":
        m, n, k = case["shape"]
        extents.update({"a0_b": m // 32, "a1_b": n // 64, "_ks": k})
        windows = {"a0": m, "a1": n}
    elif case["family"] == "attention":
        seq, width = case["shape"]
        extents.update({"a0_b": math.ceil(seq / 32), "a5_b": math.ceil(width / 32),
                        "a1": seq})
    _rename_and_rescale(graph, {}, shapes, extents, windows=windows)
    if case["family"] == "attention":
        seq, width = case["shape"]
        meta["kernels"][0]["grid"] = [math.ceil(seq / 32) * math.ceil(width / 32), 1, 1]
    elif case["family"] == "matmul":
        m, n, _ = case["shape"]
        meta["kernels"][0]["grid"] = [math.ceil(m / 32) * math.ceil(n / 64), 1, 1]
    elif case["family"] == "pointwise":
        tile = next(a for a in _tile_axes(graph) if a.name == "a0")
        meta["kernels"][0]["grid"] = [math.ceil(tile.extent.as_static() / case["block"]), 1, 1]
    elif case["family"] in ("row_sum", "rmsnorm", "l2norm", "softmax"):
        meta["kernels"][0]["grid"] = [case["shape"][0], 1, 1]
    tier = next(name for name, data in load_manifest()["tiers"].items()
                if case in data["cases"])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    ir_paths = []
    for kernel in meta["kernels"]:
        path = out / (kernel["entry"] + ".ir.json")
        _dump(path, _graph_dict(graph))
        Graph.from_dict(json.loads(path.read_text()))
        ir_paths.append(path.name)
    _dump(out / "meta.json", meta)
    canonical_inputs = dict(inputs)
    for kernel in meta["kernels"]:
        for arg in kernel["args"]:
            if "data" in arg and not arg.get("output"):
                canonical_inputs.setdefault(arg["name"], np.asarray(arg["data"], dtype=np.float32).reshape(arg["shape"]))
    canonical_expected = dict(expected)
    save_arrays(out / "inputs.npz", canonical_inputs)
    save_arrays(out / "expected.npz", canonical_expected)
    files = {p.name: digest(p.read_bytes()) for p in sorted(out.iterdir())}
    record = {
        "version": VERSION, "tier": tier, "spec": case, "id": digest(canonical(files)),
        "files": files, "ir_paths": ir_paths,
        "tolerance_justification": "Published family tolerance; independent plain NumPy float64 reference.",
        "requires_sanitizer": False, "requires_isolated_first": False,
        "hosted_eligible": output_elements(case) <= RETURN_LIMIT,
        "local_only_reason": None if not case["local_only"] else "full output exceeds hosted 2^22 return cap",
        "input_seed_plan": {name: {"dtype": row[0], "shape": list(row[1]), "seed": row[2]}
                            for name, row in sorted(plan.items())},
        "source_graph_sha256": sha(source_ir), "source_meta_sha256": sha(PROBLEMS / source / "meta.json"),
    }
    _dump(out / "case.json", record)
    validate_case(out)
    return out
