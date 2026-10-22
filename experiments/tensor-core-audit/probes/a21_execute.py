import sys,json,subprocess,time,urllib.request
from pathlib import Path
import numpy as np
R=Path('/home/ada/Projects/emmy-hackathon-2026')
sys.path.insert(0,str(R))
from emmy.compiler.graph import Graph
import tensor_core
p=Path('/tmp/a21case')
meta=json.loads((p/'meta.json').read_text()); k=meta['kernels'][0]
g=Graph.from_dict(json.loads((p/'case.ir.json').read_text()))
d=tensor_core.classify(g,g.nodes['C'],k)
ptx=subprocess.run([sys.executable,str(R/'compile.py'),str(p/'case.ir.json')],capture_output=True,text=True,check=True).stdout
# local runner
port=48777
srv=subprocess.Popen([sys.executable,str(R/'tools/local_runner.py'),'--cache','/tmp/a21cache','--port',str(port),'--arch','sm_89'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
def post(path,body):
 req=urllib.request.Request(f'http://127.0.0.1:{port}'+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 return json.load(urllib.request.urlopen(req,timeout=60))
try:
 for _ in range(100):
  try:
   urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=1); break
  except OSError: time.sleep(.1)
 c=post('/compile',{'ptx':ptx,'arch':'sm_89'}); print('compile',c)
 rng=np.random.default_rng(123)
 A=rng.uniform(-1,1,(32,8)).astype(np.float32); B=rng.uniform(-1,1,(8,8)).astype(np.float32); C=np.zeros((32,8),np.float32)
 args=[{'name':'A','dtype':'f32','shape':[32,8],'data':A.reshape(-1).tolist()},
       {'name':'B','dtype':'f32','shape':[8,8],'data':B.reshape(-1).tolist()},
       {'name':'C','dtype':'f32','shape':[32,8],'output':True}]
 payload={'entry':'typed_bad','args':args,'grid':[2,1,1],'block':[32,1,1],'smem_bytes':0,'zero_outputs':[],'cubin_id':c['cubin_id'],'warmup':0,'iters':1,'max_return':100000}
 r=post('/run',payload); print('run ok',r.get('ok'))
 got=np.asarray(r['outputs']['C']['values'],dtype=np.float32).reshape(32,8)
 import math
 def tf32(x):
  x=np.float32(x); b=np.frombuffer(x.tobytes(),dtype=np.uint32).item() if False else 0
  return x
 # independent tf32x3 model via emitter simulate
 from tf32_emitter import simulate_matmul
 ref=simulate_matmul(A.astype(np.float32),B.astype(np.float32))
 # moral expected: CTA0 rows 0..15 uses A rows 1..; CTA1 rows 16..31 uses rows 17..
 moral=np.zeros_like(ref)
 for cta in range(2):
  off=0 if cta==0 else 16
  Ai=A[off+1:off+17]
  if Ai.shape[0] < 16:
    Ai = np.concatenate([Ai, np.zeros((16-Ai.shape[0], 8), dtype=np.float32)], axis=0)
  moral[off:off+16]=simulate_matmul(Ai,B.astype(np.float32))
 print('got vs A@B max',np.abs(got.astype(np.float64)-(A.astype(np.float64)@B.astype(np.float64))).max())
 print('got vs direct global',np.abs(got.astype(np.float64)-ref.astype(np.float64)).max())
 print('got vs unshifted CTA1?',np.abs(got[16:].astype(np.float64)-ref[16:].astype(np.float64)).max())
 print('got vs moral shifted rows',np.abs(got.astype(np.float64)-moral.astype(np.float64)).max())
 print('direct ref vs moral max',np.abs(ref.astype(np.float64)-moral.astype(np.float64)).max())
finally:
 srv.terminate()
