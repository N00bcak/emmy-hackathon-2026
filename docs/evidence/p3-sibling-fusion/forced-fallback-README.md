# P3 forced-fallback SASS audit

Candidate: `cand/compile.py`, SHA-256
`1f220a4dabcc436c49ab933bcc505e0e4d6ad1e04aef347a14a0b6cbd2fd0d1d`.

Forced fallback: `fallback/compile.py`, SHA-256
`ab85845e77c989172a140a3c2d7508472d5daac5389d21da60462ca55215bab8`.
It is mechanically identical to the candidate except for one construction-time
disable in `Emitter.__init__`:

```python
self.sibling_fusion_disabled = True  # forced-fallback diagnostic arm only
```

Both arms were compiled from complete directories containing the same
`tensor_core.py` and `tf32_emitter.py`, with the same commands and flags:

```sh
python <arm>/compile.py <scaled-case>/k_matmul_softmax_reduce_b55365.ir.json > kernel.ptx
ptxas -arch=<sm_89|sm_120> kernel.ptx -o kernel.<arch>.cubin
cuobjdump -sass kernel.<arch>.cubin > kernel.<arch>.sass
ptxas -v -arch=<sm_89|sm_120> kernel.ptx -o kernel.verbose.<arch>.cubin
```

`kernel.verbose.*` differs from the primary cubin only in metadata; its
disassembly hash is identical. Counts and hashes are in
`sass-summary-forced-fallback.json`.

| shape | target | instructions | LDG | STG | FFMA | IMAD | regs | spill store/load | SASS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 256 | sm_89 | 3968→3968 | 897→897 | 8→8 | 658→658 | 856→856 | 168→168 | 0/0→0/0 | byte-identical |
| 256 | sm_120 | 4096→4096 | 897→897 | 8→8 | 658→658 | 827→827 | 184→184 | 0/0→0/0 | byte-identical |
| 512 | sm_89 | 7872→6248 | 1737→1282 | 8→8 | 1937→1137 | 1492→1295 | 255→128 | 12/20→0/0 | changed |
| 512 | sm_120 | 7744→6280 | 1737→1282 | 8→8 | 1937→1137 | 1461→1294 | 255→168 | 16/12→0/0 | changed |

At 256x64 the fusion is structurally inactive (complete-run guard), so the
forced-fallback and candidate PTX, cubin, and final SASS are byte-identical on
both targets. This is a fail-closed result, not an activation claim.

At 512x64 the final SASS retains the intended shared-load fusion. Representative
sm_89 candidate sequence:

```text
/*0460*/ LDG.E.128 R8, [R8.64] ;
/*04e0*/ FFMA R27, R8, R57.reuse, R27 ;
/*04f0*/ FFMA R40, R9, R57.reuse, R40 ;
/*0500*/ FFMA R41, R10, R57.reuse, R41 ;
/*0510*/ FFMA R42, R11, R57, R42 ;
```

One four-wide global load feeds multiple sibling accumulators. The fallback
uses a separate scalar/load-and-fold sequence per accumulator. The corresponding
address-form changes are sm_89 IADD3 `1481→1348`, LOP3 `55→27`, IMAD
`1492→1295`; sm_120 IMAD `1461→1294`. STG remains 8 on both arms and both
targets.

SASS hashes: 256 sm_89 `b6104069...`, sm_120 `ef1d3bca...` (both arms identical);
512 sm_89 fallback `90f8eab6...` → candidate `fd07ba48...`; sm_120 fallback
`8a4674cd...` → candidate `153d8a54...`. Full 64-character hashes, PTX and cubin
identities, and opcode counts are in `sass-summary-forced-fallback.json`. The
512 candidate still passes scaled full-output correctness; no new timing was
needed or run for this forced-fallback audit.
Note: `SUPERSEDED-forced-fallback-sass-flattened/` was an intermediate copy with colliding arm filenames. It is retained only for audit and MUST NOT be used. The authoritative evidence is `forced-fallback-sass-per-arm/` plus `forced-fallback-sass-summary.json`.
