#!/usr/bin/env python3
"""Run the unchanged challenge client against a local runner inside an admitted GPU job."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.published import PUBLISHED_PROBLEMS, published_status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--problem',action='append')
    parser.add_argument('--port',type=int,default=48702)
    parser.add_argument('--timing', action='store_true',
                        help='Opt in to indicative timings after ALL 15 published cases pass')
    args=parser.parse_args()
    compiler=args.compiler.resolve()
    args.out.mkdir(parents=True,exist_ok=False)
    cases=args.problem or list(PUBLISHED_PROBLEMS)
    if len(cases) != len(set(cases)) or any(p not in PUBLISHED_PROBLEMS for p in cases):
        parser.error('select distinct published problem IDs')
    report={'compiler_sha256':hashlib.sha256(compiler.read_bytes()).hexdigest(),'cases':[],
            'benchmark_qualified':False,
            'qualification_note':'Standalone correctness smoke; valid research benchmarks require frozen-source/suite Store gates.'}
    server_log=(args.out/'server.log').open('w')
    server=subprocess.Popen([sys.executable,str(ROOT/'tools/local_runner.py'),
                             '--cache',str(args.out/'cache'),'--port',str(args.port),'--arch','sm_89'],
                            stdout=server_log,stderr=subprocess.STDOUT)
    url=f'http://127.0.0.1:{args.port}'
    def post(path,body):
        request=urllib.request.Request(url+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=40) as response:
            return json.load(response)
    def check_case(problem,label=None):
        command=[sys.executable,str(ROOT/'emmy-hackathon-2026/submit.py'),str(problem),
                 '--compiler',str(compiler),'--server',url]
        started=time.monotonic()
        result=subprocess.run(command,capture_output=True,text=True,timeout=90)
        label=label or Path(problem).name
        (args.out/f'{label}.log').write_text(result.stdout+result.stderr)
        report['cases'].append({'problem':label,'command':command,'exit_code':result.returncode,
                                'wall_seconds':time.monotonic()-started})
        print(label, 'PASS' if result.returncode==0 else 'FAIL',flush=True)
        (args.out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        for _ in range(100):
            if server.poll() is not None: raise RuntimeError('local server exited')
            try:
                with urllib.request.urlopen(url+'/health',timeout=1) as response:
                    report['health']=json.load(response)
                break
            except OSError: time.sleep(.1)
        else: raise RuntimeError('local server did not start')
        for problem in cases: check_case(problem)
        report['published_correctness'] = published_status(report['cases'], source_unchanged=(
            hashlib.sha256(compiler.read_bytes()).hexdigest() == report['compiler_sha256']))
        (args.out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        # Reorder a noncommutative scale/shift ABI without altering its IR or launch.
        source=ROOT/'emmy-hackathon-2026/problems/p02_scale_shift'
        with tempfile.TemporaryDirectory(prefix='abi-',dir=args.out) as temporary:
            import shutil
            fixture=Path(temporary)
            for p in source.iterdir():
                if p.is_file(): shutil.copyfile(p,fixture/p.name)
            meta=json.loads((fixture/'meta.json').read_text())
            for k in meta['kernels']:
                k['arg_order']=list(reversed(k['arg_order']))
                by_name={a['name']:a for a in k['args']}
                k['args']=[by_name[n] for n in k['arg_order']]
            (fixture/'meta.json').write_text(json.dumps(meta))
            check_case(fixture,'reordered_scale_shift_abi')
        report['published_correctness'] = published_status(report['cases'], source_unchanged=(
            hashlib.sha256(compiler.read_bytes()).hexdigest() == report['compiler_sha256']))
        # Partial/debug sweeps cannot authorize performance evaluation.
        if not args.timing or not report['published_correctness']['all15_passed']:
            report['timing_skipped'] = 'not requested' if not args.timing else 'all-15 correctness gate failed'
            (args.out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
            return int(any(row['exit_code'] for row in report['cases']) or
                       (args.timing and not report['published_correctness']['all15_passed']))
        # Warm, repeated indicative timing for correct representative kernels.
        import numpy as np
        report['benchmarks']=[]
        for problem in ('p00_scalar_add','p02_scale_shift','r01_array_sum'):
            folder=ROOT/'emmy-hackathon-2026/problems'/problem
            meta=json.loads((folder/'meta.json').read_text())
            k=meta['kernels'][0]
            compiled=subprocess.run([sys.executable,str(compiler),str(next(folder.glob('*.ir.json')))],
                                     capture_output=True,text=True,check=True,timeout=10)
            c=post('/compile',{'ptx':compiled.stdout})
            assert c.get('ok'),c
            r=post('/run',{**k,'cubin_id':c['cubin_id'],'warmup':3,'iters':10,'max_return':2**22})
            assert r.get('ok'),r
            with np.load(folder/'expected.npz') as expected:
                for name in expected.files:
                    assert not r['outputs'][name]['truncated']
                    got=np.asarray(r['outputs'][name]['values'],dtype=np.float32).reshape(expected[name].shape)
                    np.testing.assert_allclose(got,expected[name],rtol=meta['rtol'],atol=meta['atol'])
            report['benchmarks'].append({'problem':problem,'response':r,
                'claim':'Indicative local latency only; no comparative speedup or 5090 parity claim'})
        (args.out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        return int(any(row['exit_code'] for row in report['cases']))
    finally:
        server.terminate()
        try: server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill();server.wait()
        server_log.close()

if __name__=='__main__': raise SystemExit(main())
