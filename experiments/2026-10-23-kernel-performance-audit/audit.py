"""Offline audit of retained, source-bound timings; never dispatches GPU work."""
from pathlib import Path
import collections
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
EVIDENCE = ROOT / 'experiments/2026-10-22-correctness-main-integration'
sys.path.insert(0, str(ROOT))
import compile as compiler
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def useful_work(name, meta):
    """Documented mathematical workload counts, checked against typed IR below.

    These audit formulas describe published algorithms; they never affect lowering.
    Division, reciprocal, sqrt, exp and comparisons are counted separately.
    """
    shapes = {a['name']: a['shape'] for k in meta['kernels'] for a in k['args']}
    outputs = meta['graph_outputs']
    n = math.prod(shapes[outputs[0]])
    special = {}
    if name in ('p00_scalar_add', 'p01_vector_add'):
        f, derivation = n, 'one add per output'
    elif name == 'p02_scale_shift':
        f, derivation = 2*n, 'one multiply and one add per output'
    elif name == 'p03_relu_mul':
        f, derivation = n, 'one multiply per output; relu excluded from FLOPs'
        special = {'relu': n}
    elif name.startswith('r0'):
        size = math.prod(shapes['x'])
        f, derivation = size-n, 'input elements minus output rows: mathematical sum additions'
        if name == 'r03_mean':
            special = {'divide': 1}
    elif name.startswith('m0'):
        m, k = shapes['x0']; k2, cols = shapes['x1']; assert k == k2
        f, derivation = 2*m*k*cols, f'conventional GEMM 2*{m}*{k}*{cols}; atomic overhead excluded'
    elif name == 'f01_rmsnorm':
        rows, cols = shapes['x']
        f = rows*(4*cols+1)
        derivation = f'{rows}*(square {cols} + sum {cols-1} + mean/epsilon 2 + normalize/weight {2*cols})'
        special = {'sqrt': rows, 'reciprocal': 2*rows}
    elif name == 'f02_row_softmax':
        rows, cols = shapes['x']
        f, derivation = rows*(3*cols-1), f'{rows}*(subtract {cols} + sum {cols-1} + normalize multiply {cols}); ordinary stable softmax useful-work convention'
        special = {'exp': rows*cols, 'reciprocal': rows, 'maximum_comparisons': rows*(cols-1)}
    elif name == 'f03_l2norm':
        rows, cols = shapes['x']
        f, derivation = rows*(3*cols-1), f'{rows}*(square {cols} + sum {cols-1} + normalize multiply {cols})'
        special = {'sqrt': rows, 'reciprocal': rows}
    elif name.startswith('a0'):
        m, d = shapes['q']; d2, keys = shapes['kt']; keys2, v = shapes['v']
        assert d == d2 and keys == keys2
        contractions = 2*m*keys*(d+v)
        softmax = m*(4*keys-1)
        f = contractions + softmax
        derivation = f'QK+PV conventional FLOPs {contractions}; score scaling + stable-softmax subtract/sum/normalize multiply {softmax}; exp, reciprocal, maximum excluded'
        special = {'exp': m*keys, 'reciprocal': m+1, 'maximum_comparisons': m*(keys-1)}
    else:
        raise ValueError(name)
    return f, derivation, special


def attention_schedule_counts(body):
    """Exact typed-IR scalar operation instances for these branch-free schedules.

    Not hardware counters: assembler fusion/CSE and instruction expansion differ.
    """
    counts = collections.Counter()
    def walk(statements, weight=1):
        for s in statements:
            kind = type(s).__name__
            if kind == 'Tile':
                walk(s.body, weight*math.prod(a.extent.as_static() for a in s.axes))
            elif kind == 'Loop':
                walk(s.body, weight*s.axis.extent.as_static())
            elif kind in ('Assign', 'Accum'):
                counts[getattr(s.op, 'name', s.op)] += weight
            elif kind == 'Load':
                counts['load_elements:'+s.input] += weight*len(s.names)
            elif kind == 'Write':
                counts['store_elements:'+s.output] += weight*len(s.values)
            elif kind not in ('Let', 'Init'):
                raise ValueError('unaccounted attention statement: '+kind)
    walk(body)
    return dict(counts)


def main():
    source = sha(ROOT/'compile.py')
    summary = read(EVIDENCE/'independent-summary.json')
    receipt = read(EVIDENCE/'independent47-timing-receipt.json')
    report = receipt['report']
    fresh = read(EVIDENCE/'fresh47-correctness-receipt.json')['report']
    assert source == summary['compiler_sha256']
    source_manifest=read(EVIDENCE/'source-manifest.json')
    assert hashlib.sha256(json.dumps(source_manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()==report['source_id']
    assert report['source_id'] == summary['source_id'] == fresh['source_id']
    assert report['suite_id'] == fresh['suite_id']
    assert report['published_gate_id'] == fresh['published_gate_id']
    assert report['admission_status'] == 'observed_no_contention'
    assert report['benchmark_qualified'] and report['evaluation_ok'] and report['sanitizer_passed']
    assert fresh['evaluation_ok'] and fresh['sanitizer_passed']
    published = [r for r in report['cases'] if r.get('scope') == 'published']
    assert len(published) == 15 and len({r['problem'] for r in published}) == 15
    assert all(r['status']=='pass' and r['isolated'] and r['guards_ok'] and r['inputs_ok'] for r in report['cases'])
    measured = {r['case_id']:r for r in csv.DictReader((EVIDENCE/'independent-per-case.csv').open())}
    path = ROOT/'.agents/skills/kernel-performance/scripts/metrics.py'
    spec = importlib.util.spec_from_file_location('performance_metrics', path)
    metrics = importlib.util.module_from_spec(spec); spec.loader.exec_module(metrics)
    ptxdir = OUT/'ptx'; ptxdir.mkdir(exist_ok=True)
    rows = []
    for case in report['cases']:
        case_id = case['case_id']; timing = report['timing'][case_id]; pairs = timing['pairs']
        assert timing['ok'] and len(pairs)==32 and all(p['complete'] for p in pairs)
        times = [p['samples_us']['candidate'] for p in pairs]
        base = [p['samples_us']['baseline'] for p in pairs]
        median = statistics.median(times)
        assert math.isclose(median, float(measured[case_id]['candidate_median_us']), abs_tol=1e-9)
        name = case.get('problem') or measured[case_id]['case_name']
        row = {'case':name,'case_id':case_id,'scope':case['scope'],
               'latency_us':median,'p10_us':float(np.quantile(times,.1)),
               'p90_us':float(np.quantile(times,.9)), 'min_us':min(times),'max_us':max(times),
               'baseline_median_us':statistics.median(base),'pair_count':32,
               'aa_pairs':16,'aa_relative_noise_threshold':report['per_case_noise'][case_id]['threshold'],
               'comparison_verdict':measured[case_id]['verdict'],'cache_mode':timing['cache_mode'],
               'correctness':'retained isolated pass with guards/input checks',
               'useful_flops':None,'logical_bytes':None,'work_mode':None,
               'dram_bytes':None,'l2_bytes':None,'shared_bytes_measured':None,
               'executed_flops':None,'registers_per_thread':None,'spills':None,
               'achieved_occupancy':None,'actual_active_sms':None,
               'reference_latency_us':None,'physical_bandwidth_gb_s':None}
        if case['scope']=='published':
            folder = ROOT/'emmy-hackathon-2026/problems'/name
            meta = read(folder/'meta.json'); binding = report['calibration_cases'][case_id]
            for filename in ['meta.json','expected.npz']:
                assert sha(folder/filename) == binding['case_files_sha256'][filename]
            assert len(meta['kernels'])==1
            kernel = meta['kernels'][0]
            ir = folder/(kernel['entry']+'.ir.json')
            graph = compiler.Graph.from_dict(read(ir)); op = graph.nodes[kernel['node']].op
            assert op.name==kernel['entry'] and type(op.body[0]).__name__=='Tile'
            tile = op.body[0]
            ptx = subprocess.run([sys.executable,str(ROOT/'compile.py'),str(ir)],capture_output=True,check=True)
            assert hashlib.sha256(ptx.stdout).hexdigest()==binding['candidate_ptx_sha256'][0], name+' PTX differs from qualified receipt'
            (ptxdir/(name+'.ptx')).write_bytes(ptx.stdout)
            f, derivation, special = useful_work(name,meta)
            logical = sum(math.prod(a['shape'])*4 for a in kernel['args'] if not a.get('scalar'))
            assert all(a['dtype']=='f32' for a in kernel['args'])
            blocks=math.prod(kernel['grid']); threads=math.prod(kernel['block'])
            row.update(useful_flops=f, logical_bytes=logical,work_mode='fp32_simt',
                       work_derivation=derivation,special_operations_useful=special,
                       compute_ceiling={'flops_s':82.6e12,'mode':'fp32_simt','basis':'theoretical',
                                        'source':'NVIDIA Ada architecture whitepaper Appendix A: RTX4090 FP32 non-Tensor at boost; rounded82.6TFLOP/s'},
                       ir_sha256=sha(ir),meta_sha256=sha(folder/'meta.json'),
                       qualified_ptx_sha256=binding['candidate_ptx_sha256'][0],
                       grid=kernel['grid'],block=kernel['block'],blocks=blocks,threads_per_block=threads,
                       tile_cells=math.prod(a.extent.as_static() for a in tile.axes),
                       max_simultaneous_sms_from_grid=min(blocks,128),
                       max_simultaneous_sm_coverage_pct=100*min(blocks,128)/128,
                       dynamic_shared_launch_bytes=kernel.get('smem_bytes',0),
                       declared_shared_ir_bytes=sum(math.prod(compiler.static(e) for e in s.extents)*({'float':4,'double':8,'int':4}[s.dtype] if isinstance(s.dtype,str) else s.dtype.nbytes) for s in tile.body if type(s).__name__=='Smem'),
                       atomic_zero_outputs=kernel.get('zero_outputs',[]),
                       ptx_static_instruction_counts=dict(collections.Counter(
                           (line.strip().split()[1] if line.strip().startswith('@') else line.strip().split()[0]).rstrip(';') for line in ptx.stdout.decode().splitlines()
                           if line.strip() and line.strip().endswith(';') and not line.strip().startswith(('.', '//')))),
                       resource_note='PTX virtual registers are not physical allocation; no local cubin/SASS/counter artifact available.',
                       logical_byte_convention='Each distinct argument buffer read/written once; excludes atomic RMW/reset traffic, rereads and spills; ideal reuse, not measured DRAM traffic.')
            if name.startswith('a0'):
                counts = attention_schedule_counts(op.body)
                q = next(a['shape'] for a in kernel['args'] if a['name']=='q')
                kt = next(a['shape'] for a in kernel['args'] if a['name']=='kt')
                unique_products=q[0]*q[1]*kt[1]
                row.update(attention_ir_scalar_operation_instances=counts,
                           qk_product_instances=counts['load_elements:q'],
                           useful_unique_qk_products=unique_products,
                           qk_recomputation_factor=counts['load_elements:q']/unique_products,
                           unprofiled_hardware_operation_count=None,
                           attention_count_note='Typed-IR schedule expansion, not profiler counts. Scalar PTX preserves lane-private score accumulation; no warp exchange in emitted attention PTX.')
                assert counts['store_elements:matmul_1']==4096
                assert row['qk_recomputation_factor']==32
                assert 'shfl.sync' not in ptx.stdout.decode()
                row['online_exp_instances_vs_unique_score_exp']=counts['exp']/(q[0]*kt[1])
        if name.startswith('a0'):
            row.update(supported_constraints=['32-fold QK product replication in typed schedule and lane-private scalar PTX',f"at most {row['max_simultaneous_sms_from_grid']} simultaneously active SMs from grid"],
                       next_check='Admitted narrowly filtered attention counter/SASS profile; test legal within-warp/subwarp score sharing under identical metadata.')
        elif name in ('r02_big_sum','r03_mean'):
            row.update(supported_constraints=['one prescribed CTA','scalar strided load and carried-sum chain in emitted PTX'],
                       next_check='Same-launch reduction reference and dependency/issue counters; evaluate independent accumulators only with numerical confirmation.')
        elif name=='m01_matmul_512':
            row.update(supported_constraints=['72KiB per-CTA shared launch allocation limits Ada to one resident CTA/SM','two duplicate tail-prefetch stages in supplied IR'],
                       next_check='Profile scheduler/compute/shared/memory behavior before choosing address, pipeline or reuse changes.')
        elif name=='m02_matmul_rect':
            row.update(supported_constraints=['32 split-K partitions','262144 logical atomic partial stores for8192 final outputs'],
                       next_check='Isolated same-launch reference and atomic/issue profiling with prescribed resets and sanitizer.')
        elif case['scope']=='published':
            row.update(supported_constraints=[f"only {row['blocks']} prescribed CTAs",'broad retained A/A comparison noise'],
                       next_check='Validate timer/minimal-control behavior under the same protocol before interpreting small differences.')
        else:
            row.update(supported_constraints=['local generated IR/metadata unavailable'],
                       next_check='Retrieve exact retained case bytes before estimating work or assigning a bottleneck.')
        row.update(bottleneck_status='hardware bottleneck unresolved',
                   causal_confidence='Schedule/code constraints supported where listed; causal latency attribution unconfirmed without counters or controlled reference.',
                   optimized_reference_status='not measured')
        row['derived_metrics'] = metrics.calculate(row)['metrics']
        rows.append(row)
    manifest={'compiler_sha256':source,'source_id':report['source_id'],'suite_id':report['suite_id'],
              'published_gate_id':report['published_gate_id'],'evidence_id':receipt['evidence_id'],
              'device':report['local_device'],'fresh_measurements':False,
              'emmy_local_version':importlib.metadata.version('emmy-ml'),
              'scope':'Offline audit; qualified retained32paired samples per case; no GPU dispatch or new gate issuance.',
              'remote_status':'SSH failed hostname resolution in this side chat; live gates/jobs, profiler availability, observed clocks and durable remote store cannot be revalidated.',
              'clock_note':'Device clock_rate_khz is a nominal limit, not an observed clock trace. MFU uses published theoretical boost ceiling.',
              'unavailable':['fresh timing','Nsight counters','DRAM/L2/shared traffic','physical register/spill counts','live clocks','same-launch optimized reference','empty-kernel control','remote persistence'],
              'artifact_hashes':{p.name:sha(p) for p in [ROOT/'compile.py',EVIDENCE/'source-manifest.json',EVIDENCE/'independent47-timing-receipt.json',EVIDENCE/'fresh47-correctness-receipt.json',EVIDENCE/'independent-per-case.csv']},
              'analysis_code_sha256':sha(Path(__file__)),
              'metrics_calculator_sha256':sha(path),
              'peak_source':'https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf',
              'peak_fp32_tflops_s':82.6,'peak_dram_gb_s':1008,
              'bandwidth_note':'1008GB/s is theoretical reference only; logical/time is never classified as measured DRAM bandwidth.',
              'published_rows':15,'generated_rows':32}
    save('manifest.json',manifest); save('per-case.json',{'manifest':manifest,'cases':rows})
    checks=[]
    for row in rows:
        if not row['case'].startswith('a0'):
            continue
        folder=ROOT/'emmy-hackathon-2026/problems'/row['case'];meta=read(folder/'meta.json');arrays={}
        for arg in meta['kernels'][0]['args']:
            if arg.get('output'):continue
            value=(np.asarray(arg['data'],dtype=np.float32).reshape(arg['shape']) if 'data' in arg
                   else np.random.default_rng(arg['seed']).standard_normal(arg['shape']).astype(np.float32))
            arrays[arg['name']]=value
        score=(arrays['q']@arrays['kt'])/arrays['div_c1'][0]
        numerator=np.exp(score-score.max(axis=1,keepdims=True))
        result=(numerator/numerator.sum(axis=1,keepdims=True))@arrays['v']
        with np.load(folder/'expected.npz',allow_pickle=False) as expected:
            target=expected[meta['graph_outputs'][0]]
        passed=bool(np.allclose(result,target,rtol=meta['rtol'],atol=meta['atol']))
        checks.append({'case':row['case'],'ordinary_numpy_qk_stable_softmax_pv_matches_expected':passed,
                       'max_abs_error':float(np.max(np.abs(result-target))),
                       'rtol':meta['rtol'],'atol':meta['atol'],
                       'scope':'CPU semantic accounting check only; no new GPU correctness evidence.'})
        assert passed
    save('semantic-reference-checks.json',checks)
    save('metrics-input.json',{'cases':[{k:r[k] for k in ['case','latency_us','useful_flops','logical_bytes','work_mode','compute_ceiling'] if k in r} for r in rows]})
    fields=['case','scope','latency_us','p10_us','p90_us','useful_flops','useful_gflops_s','mfu_pct','logical_bytes','logical_effective_bandwidth_gb_s','blocks','threads_per_block','max_simultaneous_sm_coverage_pct','qk_recomputation_factor','aa_relative_noise_threshold','comparison_verdict','dram_bandwidth_gb_s','reference_latency_us']
    with (OUT/'per-case.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in rows:
            v={k:r.get(k) for k in fields};v.update({k:r['derived_metrics'][k] for k in ['mfu_pct','logical_effective_bandwidth_gb_s','dram_bandwidth_gb_s']})
            t=r['derived_metrics']['useful_tflops_s'];v['useful_gflops_s']=t*1000 if t is not None else None
            writer.writerow(v)
    print(json.dumps({'published':15,'generated':32,'all_published_ptx_match_receipt':True,'source':source,'output':str(OUT)}))


if __name__=='__main__':
    main()
