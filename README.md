# Emmy compiler discovery workspace

A general **Kernel IR → PTX** baseline and a local CUDA evaluator. The baseline
passes all eight pointwise/reduction MVP problems, RMSNorm and L2Norm on the tested
local GPU. Five open cases remain explicitly unsupported: both matmuls, row
softmax, and both attention cases. Hidden-set correctness is unverified.

The lowering is ours, using Emmy's typed IR loader and sibling metadata for ABI
and launch contracts. It does not delegate IR lowering to Emmy's renderer, nvcc,
or another compiler. It emits PTX 8.0 targeting sm_89. Scalarized vector accesses
and simple register/index emission are intentional baseline limitations; no
comparative speedup is claimed.

```sh
uv venv --python 3.13
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python compile.py emmy-hackathon-2026/problems/p00_scalar_add/k_add_pointwise.ir.json > out.ptx
```

Python 3.12+ is required. The challenge is already present as a pinned public
submodule. The owner performs any future networked Git operations. The Mac's
requirements.lock captures its installed environment; the private Linux runtime
has a separate dependency lock. Torch is unnecessary for this baseline.

Read AGENTS.md and the submodule instructions before implementation. Codex and
Claude Code share kernel-strategy, kernel-experiment and kernel-reflection.
Claude Code loads AGENTS.md through CLAUDE.md; its skills directory is a symlink
to the canonical directory. The machine-specific runtime skill is Git-ignored.

Compiler prototypes are developed as plain source snapshots on the GPU host.
Only GPU-evaluated shortlisted candidates are copied back here. No Git commands
run on that host; agents do not commit, manage remotes, or perform Git network
actions here. The owner decides what stays.

[Evaluator contract and verification](docs/evaluator-contract.md) describes the
local loopback service, the unchanged upstream client, benign hosted checks, and
known limits. tools/evaluate.py now defaults to the local service and records
file hashes without invoking Git. A missing local service never triggers public
fallback. tools/probe_endpoint.py deliberately spends a bounded number of public
requests when explicitly invoked; it is not an inner-loop dependency.

GPU test commands must run inside the configured container through its admission
wrapper. CPU tests inside that container use:

```sh
python -m unittest discover -s tests -v
```

The local runner is a trusted research tool, not a sandbox for hostile kernels.
It keeps expected-output scoring independent from execution. Its timings are
indicative and do not reproduce or predict hosted GPU latencies.

[Private repository setup](docs/version-control.md) lists the review, commit and
push commands for you to run. No remote has been added and no commit/push has
been performed by the agent.
