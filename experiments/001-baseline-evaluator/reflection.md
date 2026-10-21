# Baseline and evaluator reflection

Decision: keep this baseline as a shortlisted candidate for the owner's review;
no agent commit or publication. Source identity and result summary are in
verification.json. Raw data stays in ignored artifacts and private NFS storage.

The general typed compiler passes the 8 MVP cases and f01/f03, plus a reversed
scale/shift ABI. Typed nonfixture tests exercise a 14-element vector tail and a
257-element reduction with reversed ABI, with independent NumPy outputs; both
pass on the local GPU. Five open cases remain explicitly unsupported as
listed in the summary. Hidden-set correctness is not established.

Sixteen CPU tests pass. During review, shallow detection of collectives and a
warp-specific shared-tree assumption needed conservative rejection guards. The
final source emits byte-identical PTX for all ten accepted open kernels; final
pointwise/reduction PTX also matches the recorded hosted requests exactly.

The hypothesis was supported for the tested execution contract: the same valid
request semantics work locally and on the hosted runner. Sixteen benign hosted
HTTP requests confirmed seeded/exact inputs, by-value scalar, compile ID stability,
truncation, fresh allocations, warmups/iterations and per-launch atomic resets.
The initial local output-sum diagnostic used float64 accumulation; the recorded
hosted f32 response established native float32 accumulation for that case. The
replica was corrected and its complete local contract suite rerun successfully.

Three representative kernels have local warmup=3/iters=10 samples saved. These
measure baseline behavior only, not a speedup. The hosted server's exact timing
method, hidden evaluator, undocumented quota and implementation remain unknown.
The local runner is a compatible research execution tool for the tested cases,
not an exact server clone or a security boundary for hostile kernels.

Next uncertainty: supported execution semantics are a foundation for local
research, but generalizing coupled reduction states and asynchronous transport
is needed for the remaining problem families. Microprobes and independent
references should test those mechanisms before expensive compiler integration.
