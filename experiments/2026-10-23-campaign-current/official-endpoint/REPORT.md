The official RTX 5090 runner passed **14 of 15 published problems**.

Only `m01_matmul_512` failed: 262,069 of 262,144 values differed, with maximum absolute error 144.748085 at the original `rtol=atol=0.001`. The recorded inputs, positional arguments and launch match the validated local case. An independent NumPy float64 matrix product, converted to float32, matches the published reference exactly. The cause of the remote failure remains unresolved.

The earlier 15/15 and 278-case correctness results describe the RTX 4090. This compiler has not passed all 15 problems on sm_120.

See [per-case results](per-case.csv), [summary](summary.json), and [independent diagnostic](m01-independent-diagnostic.json). All 16 HTTP attempts and exact responses are retained in `requests/`. The original numerical stop and continuation of nine unstarted cases are preserved. No request was retried; the compiler was unchanged. No commit or publication occurred.

These are published-suite results, not hidden-set scores. Reported timings are indicative and do not establish a paired speedup on the RTX 5090.
