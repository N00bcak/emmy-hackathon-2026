# Tensor-core TF32 matmul: frozen inputs and evidence

Status: correctness work only. No benchmark, speedup, promotion, TMA,
`ldmatrix`, `stmatrix`, or `wgmma` claim is made here.

## Current source identity

The validated implementation is bound to these source hashes:

```text
compile.py                 996cf5fcd1b6fd70608d384856e97a9057f20b06b55b99087976c89d947c08d4
tensor_core.py             ca5af9c9f41826f9e2000de7a8537745c0393007d8eaaaf9fd212b3bfd3364a6
tf32_emitter.py            6a9e1af929985911d97de3ab9d967326dd1db0b57a849427b595ed79eb655e9d
tests/test_tensor_core.py  d81f23875828f874f210b29fdfebbb51cf4124ae4afcefc6805bf23907b52987
corpus manifest             0aa2c60421b34cf7135494b15fada15376488a0891e1fcf495906f07637626b8
```

m01 is `Plan` with PTX hash
`84e26af55cd3c696a891d583a04e13b20d48a2f349223f8140311e3de9f14b9b`.
m02 is `UnsupportedMatch` (four atoms for eight warps); its fallback PTX hash is
`3fabbf1e81c3f4ca83574876721d851910b8337a6cbd98d7f38bda9d2e7f5071`.
The full per-case table, source/corpus hashes, and execution results are in
`evidence-sm89-sm120/summary.json`.

## Frozen baseline and corpus

`baseline/baseline.json` binds the 15 published and 21 v1 corpus PTX artifacts
to compiler SHA-256
`6db2dad4944d87ae46194fd4deaf9331327cda21b21adaa434a1db41f515e6ba`.
`baseline/baseline-v2.json` separately binds the 23 independent v2 corpus IRs
and their pre-change PTX. Per-case files are in `baseline/published/`,
`baseline/corpus/`, and `baseline/corpus-v2/`. Together they are the
byte-identity SIMT fallback oracle. These are deliberately frozen pre-change
snapshots and are not rewritten to the current compiler hash; corpus-v2 is the
authoritative 23-case corpus described next.

`corpus-v2/` is frozen at `tensor-core-corpus-v2`. Its `manifest.json` records
exact file hashes, expected classifications, and unchanged `1e-3/1e-3`
tolerances. It contains 14 positive samples and 9 near misses. The generator
`tools/tensor_core_corpus.py` constructs typed Emmy nodes directly (no renderer
or compiler callback), reproduces the frozen files byte-for-byte, and computes
every reference with plain NumPy float64.

This directory also retains earlier development snapshots (`corpus/`, the
`corpus-draft*`/numbered drafts, `evaluate-sm89*`, `published-sm89/`, the
`synthetic-*` runs, and `evidence-final*`). They are provenance, not current
acceptance criteria. `corpus-v2/` and `evidence-sm89-sm120/` are the
authoritative pair; use their manifest and checksum set for replay.

## Validated path

- m01 is `Plan`, and all 14 corpus positives are `Plan` across direct-global,
  simple shared/`cp.async`, and m01's typed rotating pipeline; atom grids cover
  1x1, 2x1, 1x2, 2x2, 1x4, and 3x3 CTA factorizations, A/B transposes, and
  M/N/K extents.
- `NoMatch` and `UnsupportedMatch` cases, including m02, preserve their frozen
  PTX bytes. m02 is explicitly unsupported because its 16x32 tile has four
  atoms for eight launched warps.
- The one-atom emitter uses `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`
  three times with independent `hh`, `hl`, and `lh` C fragments and
  `cvt.rna.tf32.f32` conversions.
- The synthetic 128-cell check passes against an asymmetric CPU oracle. The
  integrated 15 Plan cases pass independent float64 CPU references and the
  local CUDA runner at unchanged tolerances.
- All 15 published cases pass, including m02, through the unchanged client.
- `sm_89` and `sm_120` evidence are kept in separate artifacts; both assembly
  checks contain TF32 `HMMA` in SASS.

## Adversarial audits

Two independent read-only audits probed the checker with typed IR whose moral
meaning differed from its spelling:

- [`tensor-core-audit/REPORT.md`](../tensor-core-audit/REPORT.md) found
  CTA-dependent direct-operand drift, output/input aliasing, and an unproved
  pipeline copy mapping. Those findings were repaired and replayed.
- [`tensor-core-audit-agent2/REPORT.md`](../tensor-core-audit-agent2/REPORT.md)
  attacked the repairs and found three more false accepts (interior-CTA-only
  pipeline mutation, hidden K phase, and partial-CTA barriers) plus a valid
  staged A-transpose rejection. The false accepts were repaired; staged A
  transpose is now accepted. Its P08 nested-K probe is malformed (loads still
  reference `k` after the loop axis was renamed `k0`) and remains fail-closed;
  it is not counted as a confirmed valid rejection.

Both reports carry their historical pre-repair hash tables. Read their status
banners and disposition sections before using their probe results as current
evidence.

Evidence is generated with:

```sh
.venv/bin/python tools/tensor_core_check.py --out <synthetic>
.venv/bin/python tools/tensor_core_evaluate.py --out <execution> --arch sm_89
.venv/bin/python tools/tensor_core_evidence.py --out <assembly>
.venv/bin/python tools/check_local.py --compiler compile.py --out <published>
.venv/bin/python -m unittest discover -s tests -v
```

`tools/tensor_core_evaluate.py` currently executes the local `sm_89` device;
`sm_120` assembly evidence is recorded separately and is not an execution claim.
The consolidated machine-readable report, exact hashes, PTX/cubin artifacts, and
test logs are under `evidence-sm89-sm120/`; read `summary.json` first and verify
the bundle with `cd evidence-sm89-sm120 && sha256sum -c SHA256SUMS`. The 23
corpus fallback rows are checked by `tests/test_tensor_core.py` against
`baseline.json` and `baseline-v2.json`; the tracked `.cubin` files under the
evidence directory are documentation artifacts and are not used by the compiler.
This bundle makes no benchmark, speedup, promotion, TMA, matrix-transport, or
hidden-set claim. The atomic split-K path remains unimplemented and outside the
one-atom emitter.

To replay the corpus and verify the bundle without a GPU:

```sh
.venv/bin/python tools/tensor_core_corpus.py /tmp/corpus-replay
sha256sum /tmp/corpus-replay/manifest.json
cd experiments/tensor-core-v1/evidence-sm89-sm120
sha256sum -c SHA256SUMS
```

The first hash must equal the corpus manifest hash above. The checksum manifest
covers the retained PTX, cubins, logs, classification rows, synthetic result,
execution result, and assembly/SASS report. The two audit directories are kept
separately because they are historical probe records, not part of the final
compiler-evidence checksum set.
