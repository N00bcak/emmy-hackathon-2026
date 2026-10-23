"""Project exact retained evidence into a current table. No GPU/HTTP operations."""
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
CAMPAIGN = OUT.parent
OLD = ROOT / 'experiments/2026-10-23-kernel-performance-audit'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def quantile(values, p):
    values = sorted(values)
    i = (len(values) - 1) * p
    lower = int(i)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (i - lower)


current = read(CAMPAIGN / 'CURRENT.json')
receipt = read(CAMPAIGN / 'attention-vector-confirmed/independent-qualified-receipt.json')
report = receipt['report']
assessment = read(CAMPAIGN / 'attention-vector-confirmed/independent-assessment-raw-pairs.json')
official = read(CAMPAIGN / 'official-endpoint/result.json')
resources = read(OUT / 'resources.json')
assert sha(ROOT / 'compile.py') == current['compile_sha256'] == official['compiler_sha256']
assert report['source_id'] == current['source_id'] == resources['source_id'] == official['source_id']
assert report['suite_id'] == resources['suite_id']
assert report['published_gate_id'] == resources['gate_id']
assert report['evaluation_ok'] and report['benchmark_qualified'] and report['sanitizer_passed']
assert report['admission_status'] == 'observed_no_contention' and resources['source_4090_gate_valid']
assert resources['active_gpu'] is None and not resources['preparation']
assert len(report['timing']) == len(assessment['rows']) == 47
assert official['evaluated'] == 15 and official['passed'] == 14 and official['failed'] == 1
published = {c['case_id']: c for c in report['cases'] if c['scope'] == 'published'}
assert len(published) == 15 and all(c['status'] == 'pass' and c['isolated'] and c['guards_ok'] and c['inputs_ok'] for c in published.values())
accounting = {c['case']: c for c in read(OLD / 'per-case.json')['cases'] if c['scope'] == 'published'}
remote = {r['problem']: r for r in official['results']}
native = {r['case_id']: r for r in resources['resources']}
rows = []
for measured in assessment['rows']:
    cid = measured['case_id']
    assert measured['source_id'] == report['source_id'] and measured['baseline_id'] == report['baseline_id']
    assert measured['noise_threshold'] == report['per_case_noise'][cid]['threshold']
    raw = report['timing'][cid]
    assert raw['ok'] and len(raw['pairs']) == 32 and all(p['complete'] for p in raw['pairs'])
    times = [p['samples_us']['candidate'] for p in raw['pairs']]
    assert math.isclose(statistics.median(times), measured['candidate_median_us'], abs_tol=1e-9)
    row = dict(case=measured['case_name'], case_id=cid, source_id=current['source_id'],
               scope='published' if cid in published else 'generated',
               latency_us=statistics.median(times), p10_us=quantile(times, .1), p90_us=quantile(times, .9),
               pair_count=32, aa_pairs=len(measured['noise_pairs']),
               aa_relative_noise_threshold=measured['noise_threshold'],
               comparison_verdict=measured['verdict'], median_block_ratio=measured['median_block_ratio'],
               cache_mode=raw['cache_mode'], correctness_4090='pass', benchmark_qualified_4090=True,
               useful_flops=None, logical_bytes=None, executed_flops=None, dram_bytes=None, l2_bytes=None,
               shared_bytes_measured=None, observed_clocks=None, achieved_occupancy=None,
               actual_active_sms=None, reference_latency_us=None, reference_ratio=None,
               bottleneck='unresolved', bottleneck_confidence=None, recoverable_us=None,
               benchmark_qualified_5090=False, official_raw_latency_us=None,
               official_status=None, official_p10_us=None, official_p90_us=None, official_mfu_pct=None,
               official_timing_note='Entire sm120 performance evaluation unqualified: published all15 correctness gate failed. Raw endpoint medians retained only as diagnostics.',
               special_operations_useful=None, resources=None)
    if cid in published:
        name = published[cid]['problem']
        old = accounting[name]
        folder = ROOT / 'emmy-hackathon-2026/problems' / name
        meta = read(folder / 'meta.json')
        assert len(meta['kernels']) == 1 and old['case_id'] == cid
        kernel = meta['kernels'][0]
        ir = folder / (kernel['entry'] + '.ir.json')
        # Reuse only semantic accounting from independently audited, identical typed IR.
        # Older compiler timings, PTX, instruction counts and bottleneck claims are not reused.
        assert sha(ir) == old['ir_sha256'] and sha(folder / 'meta.json') == old['meta_sha256']
        binding = report['calibration_cases'][cid]
        for filename in ('meta.json', 'expected.npz'):
            assert sha(folder / filename) == binding['case_files_sha256'][filename]
        assert remote[name]['case_id'] == cid
        logical = sum(math.prod(a['shape']) * 4 for a in kernel['args'] if not a.get('scalar'))
        assert all(a['dtype'] == 'f32' for a in kernel['args']) and logical == old['logical_bytes']
        row.update(case=name, useful_flops=old['useful_flops'], logical_bytes=logical,
                   work_derivation=old['work_derivation'], special_operations_useful=old['special_operations_useful'],
                   accounting_evidence='Prior typed-IR audit; IR and metadata hashes revalidated against this exact measured case.',
                   ir_sha256=sha(ir), meta_sha256=sha(folder / 'meta.json'),
                   candidate_ptx_sha256=binding['candidate_ptx_sha256'], work_mode='fp32_simt',
                   compute_ceiling={'flops_s': 82.6e12, 'mode': 'fp32_simt', 'basis': 'theoretical',
                                    'source': 'NVIDIA Ada whitepaper Appendix A; https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf'},
                   grid=kernel['grid'], block=kernel['block'], blocks=math.prod(kernel['grid']),
                   threads_per_block=math.prod(kernel['block']), shared_launch_bytes=kernel['smem_bytes'],
                   max_simultaneous_sms_from_grid=min(math.prod(kernel['grid']), report['local_device']['multiprocessors']),
                   resources=native[cid]['resources'], official_raw_latency_us=remote[name]['median_us'],
                   official_status=remote[name]['status'], next_check='Distinguish same-launch schedule limits from compiler costs with a controlled reference; no hardware cause established by MFU alone.')
        if name == 'm01_matmul_512':
            row.update(bottleneck='sm120 correctness failure; performance attribution deferred',
                       next_check='Diagnose the saved exact-input sm120 mismatch before any new performance evaluation; lowering, assembler and runner causes remain unresolved.')
    rows.append(row)
save('metrics-input.json', {'cases': rows})
calculator = ROOT / '.agents/skills/kernel-performance/scripts/metrics.py'
result = subprocess.run([sys.executable, str(calculator), str(OUT / 'metrics-input.json')], check=True, capture_output=True, text=True)
(OUT / 'metrics-output.json').write_text(result.stdout)
calculated = json.loads(result.stdout)
for row, derived in zip(rows, calculated['cases'], strict=True):
    row['derived_metrics_4090'] = derived['metrics']
    row['useful_gflops_s'] = None if row['useful_flops'] is None else 1000 * derived['metrics']['useful_tflops_s']
save('per-case.json', {'source_id': current['source_id'], 'suite_id': report['suite_id'], 'cases': rows})
fields = ['case', 'scope', 'latency_us', 'p10_us', 'p90_us', 'aa_relative_noise_threshold',
          'comparison_verdict', 'useful_flops', 'useful_gflops_s', 'mfu_pct', 'logical_bytes',
          'logical_effective_bandwidth_gb_s', 'blocks', 'threads_per_block',
          'official_raw_latency_us', 'official_status', 'benchmark_qualified_5090']
with (OUT / 'per-case.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row['derived_metrics_4090'].get(k, row.get(k)) for k in fields})
shown = [r for r in rows if r['scope'] == 'published']
table = '| Case | 4090 µs | 4090 MFU % | 5090 raw µs† | 5090 check |\n|---|---:|---:|---:|---|\n'
for row in shown:
    table += f"| {row['case']} | {row['latency_us']:.3f} | {row['derived_metrics_4090']['mfu_pct']:.5f} | {row['official_raw_latency_us']:.3f} | {row['official_status'].upper()} |\n"
(OUT / 'published-table.md').write_text(table)
save('manifest.json', dict(source_id=current['source_id'], compiler_sha256=current['compile_sha256'],
                          suite_id=report['suite_id'], evidence_id=receipt['evidence_id'],
                          gate_4090=report['published_gate_id'], official_evidence_id=current['official_evidence_id'],
                          device_4090=report['local_device'], device_5090=official['health'],
                          published_rows=15, generated_rows=32, fresh_gpu_work=False, fresh_endpoint_calls=False,
                          accounting_source_sha256=sha(OLD / 'per-case.json'), calculator_sha256=sha(calculator),
                          driver_sha256=sha(__file__), gate_5090_passed=False))
description = '''Current retained compiler: f47c / source549b. This table projects existing exact-source evidence; no new GPU run, profiler, endpoint request, source change or discovery campaign was started.

The RTX4090 passed all15 published problems and its isolated/sanitizer gate remains valid. Its 47-case independently selected timing has32 pairs per case plus16 baseline A/A pairs. Published kernel and ordered-pipeline latencies coincide because every published problem has one kernel. Warm reused allocation GPU-event intervals exclude CPU compilation, assembly, transfers, host submission, reset and result-checking costs. Full raw evidence and source/suite identities are linked by manifest.json; p10/p90 and per-case A/A noise are retained in per-case.csv/json. Fourteen latest published comparisons were inconclusive; a01 alone was a resolved improvement against that measurement's parent. No minimal-control measurement is available, and no guessed overhead is subtracted.

The official RTX5090 passed14/15. Its entire performance evaluation is unqualified under the all15 rule. Raw medians from5warmup/20timed iterations remain diagnostic only, including21.696us for the incorrectm01 output. Its API does not provide the individual timing samples, dispersion, event/reset implementation, clocks, counters or an identified matching sustained ceiling. No5090MFU or useful-throughput claim is calculated. Different device protocols do not establish a cross-device speedup.

4090MFU uses useful FP32 work / time / theoretical82.6TFLOP/s non-Tensor ceiling from NVIDIA's Ada whitepaper AppendixA; it is not measured sustained utilization. Counts reuse prior typed-IR semantic accounting only after checking every IR/metadata hash and current case identity. Useful GEMM counts use2MNK; attention uses conventional QK+stable-softmax+PV. Exp/div/sqrt/comparisons are separate in per-case.json. Native instruction counts and diagnoses from the older compiler are not reused. Logical traffic assumes each distinct argument is accessed once; logical GB/s is not measured DRAM bandwidth. Actual DRAM/L2/shared traffic, achieved occupancy, observed clocks and reference latencies remain null. Cached physical register/spill records are source-bound assembly evidence; all15published kernels have zero recorded spills on both targets. No profiler was started.

Priority: restorem01 correctness onsm120; the saved exact-input independent diagnostic rules out an observed input/ABI/launch/reference mismatch but does not identify a compiler, assembler or runner cause. Recoverable time is unknown. The next diagnostic should distinguish those causes using immutable savedPTX/inputs before evaluating new speed work. Among correct4090cases, m01=25.60us, a02=23.552us, a01=18.432us andr02=7.76us are the largest current latency observations; this ranking alone does not establish recoverable compiler time. Existing schedule limits (oneCTA for r02/r03, fourCTAs for a01, andm01's72KiB shared allocation) are constraints, not proven bottlenecks.

No owned live jobs remain. Reproduce this offline table with python3 experiments/2026-10-23-campaign-current/current-performance/table.py. No compiler repair or new experiment was performed.
'''
(OUT / 'REPORT.md').write_text(description + '\n' + table)
(OUT / 'HANDOFF.md').write_text('Current offline table: read manifest.json, REPORT.md and per-case.json. No owned GPU/CPU jobs or active campaign. Before any new performance work, preserve the recorded sm120 all15 failure and obtain authorization for diagnosis/implementation as appropriate. Exact next offline command: python3 experiments/2026-10-23-campaign-current/current-performance/table.py\n')
print(table)
