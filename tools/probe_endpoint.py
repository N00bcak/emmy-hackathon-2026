#!/usr/bin/env python3
"""Bounded benign runner contract checks using a locally verified compiler.

No scans, invalid GPU accesses, load tests, or hidden evaluation. Saves exact
requests/responses. Call only intentionally; this spends at most 20 HTTP requests.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import requests

ROOT=Path(__file__).resolve().parents[1]
COUNTER='''.version 8.0
.target sm_89
.address_size 64
.visible .entry k_counter(.param .u64 output) {
.reg .b64 %rd;
.reg .f32 %one;
ld.param.u64 %rd, [output];
mov.f32 %one, 0f3f800000;
red.global.add.f32 [%rd], %one;
ret;
}
'''
SCALAR='''.version 8.0
.target sm_89
.address_size 64
.visible .entry k_scalar(.param .f32 value, .param .u64 output) {
.reg .b64 %rd;
.reg .f32 %value;
ld.param.f32 %value, [value];
ld.param.u64 %rd, [output];
st.global.f32 [%rd], %value;
ret;
}
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--server',default='http://runner.riftstack.ai:48701')
    parser.add_argument('--inspect-schema',action='store_true')
    parser.add_argument('--assemble-only',action='store_true')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    summary={'server':args.server,'requests':0,'checks':[],
             'compiler_sha256':hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
             'limitations':['Behavioral compatibility only; server source, exact timing method and hidden evaluation are not established.']}
    def save(): (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    def check(name,passed,detail=None):
        summary['checks'].append({'name':name,'pass':bool(passed),'detail':detail})
        save()
        if not passed: raise AssertionError(f'{name}: {detail}')
        print(name,'PASS',flush=True)
    def request(path,body=None):
        if summary['requests']>=20: raise RuntimeError('20-request budget exhausted')
        summary['requests']+=1
        identity=f"{summary['requests']:02d}-{path.strip('/')}"
        (args.out/f'{identity}-request.json').write_text(json.dumps(body,indent=2)+'\n')
        started=time.monotonic()
        response=requests.get(args.server+path,timeout=(5,30)) if body is None else requests.post(args.server+path,json=body,timeout=(5,45))
        try: payload=response.json()
        except ValueError: payload={'non_json_body':response.text[:4096]}
        (args.out/f'{identity}-response.json').write_text(json.dumps({'status':response.status_code,'headers':dict(response.headers),
            'wall_seconds':time.monotonic()-started,'body':payload},indent=2)+'\n')
        save()
        response.raise_for_status()
        # Busy is not an experimental failure. Stop without increasing load.
        if payload.get('stage')=='busy': raise RuntimeError('Runner busy: defer and preserve this evidence; do not retry blindly')
        return payload
    def compile_ptx(ptx):
        response=request('/compile',{'ptx':ptx})
        if not response.get('ok'): raise RuntimeError(response)
        return response
    def run(k,cubin,**overrides):
        body={key:copy.deepcopy(k[key]) for key in ('entry','grid','block','smem_bytes','zero_outputs','args')}
        body.update(cubin_id=cubin,warmup=0,iters=1,max_return=2**22)
        body.update(overrides)
        response=request('/run',body)
        if not response.get('ok'): raise RuntimeError(response)
        return response
    try:
        if args.assemble_only:
            for name,ptx in [('counter',COUNTER),('scalar',SCALAR)]:
                source=args.out/f'{name}.ptx'; source.write_text(ptx)
                done=subprocess.run(['ptxas','-arch=sm_89',str(source),'-o',str(args.out/f'{name}.cubin')],
                                     capture_output=True,text=True,timeout=30)
                (args.out/f'{name}.log').write_text(done.stdout+done.stderr)
                check(f'{name}_assembles',done.returncode==0)
            return 0
        health=request('/health'); summary['health']=health
        check('health',health.get('ok') and health.get('can_compile') and health.get('can_run'))
        if args.inspect_schema:
            schema=request('/openapi.json')
            properties=schema['components']['schemas']['RunRequest']['properties']
            summary['schema_defaults']={k:v.get('default') for k,v in properties.items() if 'default' in v}
        compiled={}
        for problem in ('p00_scalar_add','r01_array_sum'):
            folder=ROOT/'emmy-hackathon-2026/problems'/problem
            meta=json.loads((folder/'meta.json').read_text()); k=meta['kernels'][0]
            ptx=subprocess.run([sys.executable,str(args.compiler.resolve()),str(next(folder.glob('*.ir.json')))],
                               capture_output=True,text=True,timeout=10,check=True).stdout
            (args.out/f'{problem}.ptx').write_text(ptx)
            c=compile_ptx(ptx); r=run(k,c['cubin_id'])
            compiled[problem]=(k,c,ptx)
            with np.load(folder/'expected.npz') as expected:
                for name in expected.files:
                    output=r['outputs'][name]
                    got=np.asarray(output['values'],dtype=np.float32).reshape(expected[name].shape)
                    check(f'{problem}_correct',not output['truncated'] and np.allclose(got,expected[name],rtol=meta['rtol'],atol=meta['atol']))
        k,c,ptx=compiled['p00_scalar_add']
        repeated=compile_ptx(ptx)
        check('compile_id_stable',c['cubin_id']==repeated['cubin_id'])
        seeded=copy.deepcopy(k)
        for a in seeded['args']:
            if a['name']=='x0': a['seed']=7
            if a['name']=='x1': a.pop('seed',None);a['data']=[0.0]*int(np.prod(a['shape']))
        r=run(seeded,c['cubin_id'])
        expected=np.random.default_rng(7).standard_normal(k['args'][0]['shape']).astype(np.float32).reshape(-1)
        check('seed_generation_and_exact_data',np.array_equal(np.asarray(r['outputs']['add']['values'],dtype=np.float32),expected))
        r=run(seeded,c['cubin_id'],max_return=3)
        output=r['outputs']['add']
        check('truncation',output['truncated'] and len(output['values'])==3 and np.array_equal(np.asarray(output['values'],dtype=np.float32),expected[:3]))
        summary['sum_observation']={'returned':output['sum'],'float32_sum':float(expected.sum()),'float64_sum':float(expected.astype(np.float64).sum())}
        check('f32_full_sum_when_truncated',output['sum']==float(expected.sum()))
        scalar=compile_ptx(SCALAR)
        kscalar={'entry':'k_scalar','grid':[1,1,1],'block':[1,1,1],'smem_bytes':0,'zero_outputs':[],
                 'args':[{'name':'value','dtype':'f32','scalar':True,'value':3.25},
                         {'name':'output','dtype':'f32','shape':[1],'output':True}]}
        check('scalar_by_value',run(kscalar,scalar['cubin_id'])['outputs']['output']['values']==[3.25])
        counter=compile_ptx(COUNTER)
        kcounter={'entry':'k_counter','grid':[1,1,1],'block':[1,1,1],'smem_bytes':0,'zero_outputs':[],
                  'args':[{'name':'output','dtype':'f32','shape':[1],'output':True}]}
        check('warmup_plus_iterations_accumulate',run(kcounter,counter['cubin_id'],warmup=2,iters=3)['outputs']['output']['values']==[5.0])
        check('zero_before_each_launch',run(kcounter,counter['cubin_id'],warmup=2,iters=3,zero_outputs=['output'])['outputs']['output']['values']==[1.0])
        check('fresh_buffers_each_request',run(kcounter,counter['cubin_id'],warmup=2,iters=3)['outputs']['output']['values']==[5.0])
        body={**kcounter,'cubin_id':counter['cubin_id']}
        response=request('/run',body)
        check('default_warmup5_iters20',response.get('ok') and response.get('iters')==20 and response['outputs']['output']['values']==[25.0])
        return 0
    finally: save()

if __name__=='__main__': raise SystemExit(main())
