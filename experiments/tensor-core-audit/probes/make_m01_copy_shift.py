import sys,json,dataclasses,hashlib,subprocess
from pathlib import Path
sys.path.insert(0,'/home/ada/Projects/emmy-hackathon-2026')
from emmy.compiler.graph import Graph
from emmy.compiler.ir.expr import BinaryExpr,Literal
import tensor_core
root=Path('/home/ada/Projects/emmy-hackathon-2026')
src=root/'emmy-hackathon-2026/problems/m01_matmul_512'
g=Graph.from_dict(json.loads((src/'k_matmul_2967be.ir.json').read_text()))
meta=json.loads((src/'meta.json').read_text())['kernels'][0]
n=g.nodes['matmul']
def mut(s):
  repl={}
  for f in dataclasses.fields(s):
    if f.name in ('body','else_body','combine_states') and getattr(s,f.name):
      repl[f.name]=tuple(mut(x) for x in getattr(s,f.name))
  if repl: s=dataclasses.replace(s,**repl)
  if type(s).__name__=='CpAsyncCopy' and s.smem=='_b_smem':
    idx=list(s.src_index); idx[1]=BinaryExpr('+',idx[1],Literal(1,'int')); return dataclasses.replace(s,src_index=tuple(idx))
  return s
tile=n.op.body[0]; tile2=dataclasses.replace(tile,body=tuple(mut(x) for x in tile.body))
n2=dataclasses.replace(n,op=dataclasses.replace(n.op,body=(tile2,)))
d=tensor_core.classify(g,n2,meta)
print(d.verdict,d.reason)
out=root/'experiments/tensor-core-audit/probes/m01-b-copy-col-shift'
if out.exists(): import shutil; shutil.rmtree(out)
out.mkdir(parents=True)
raw=g.to_dict(); raw['nodes'][meta['node']]['op_fields']['body']=(tile2,)
(out/'k_matmul_2967be.ir.json').write_bytes(json.dumps(raw,sort_keys=True,separators=(',',':'),default=repr).encode()+b'\n')
(out/'meta.json').write_bytes((src/'meta.json').read_bytes())
r=subprocess.run([sys.executable,str(root/'compile.py'),str(out/'k_matmul_2967be.ir.json')],capture_output=True,text=True)
print('compile rc',r.returncode,'mma',r.stdout.count('mma.sync'),'stderr',r.stderr.strip()[:300])
print('ptxsha',hashlib.sha256(r.stdout.encode()).hexdigest())
open('/tmp/m01mut.ptx','w').write(r.stdout)
