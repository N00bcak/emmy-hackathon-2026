import sys
sys.path.insert(0,'/home/ada/Projects/emmy-hackathon-2026')
import json
from pathlib import Path
from emmy.compiler.graph import Graph
from emmy.compiler.ir.expr import BinaryExpr,Literal,TernaryExpr,Var
import tensor_core

def ev(e,env):
 if isinstance(e,Literal): return e.value
 if isinstance(e,Var): return env[e.name]
 if isinstance(e,BinaryExpr):
  a,b=ev(e.left,env),ev(e.right,env)
  if e.op=='+': return a+b
  if e.op=='-': return a-b
  if e.op=='*': return a*b
  if e.op in ('/','//'): return int(a/b)
  if e.op=='%': return a%b
  if e.op=='<': return int(a<b)
  if e.op=='<=': return int(a<=b)
  if e.op=='>': return int(a>b)
  if e.op=='>=': return int(a>=b)
  if e.op=='==': return int(a==b)
  if e.op=='!=': return int(a!=b)
  raise Exception(e.op)
 if isinstance(e,TernaryExpr): return ev(e.if_true,env) if ev(e.cond,env) else ev(e.if_false,env)
 raise Exception(type(e))

def flat(idx,shape):
 if len(idx)!=len(shape): raise Exception('rank')
 v=0
 for i,s in zip(idx,shape):
  if not (0<=i<s): raise Exception(('oob',i,s))
  v=v*s+i
 return v

def f3(x):
 return f'{float(x):.9g}'

def copy_offsets(copy,sh):
 w=copy.nbytes//4
 return w

def run():
 p=Path('/home/ada/Projects/emmy-hackathon-2026/emmy-hackathon-2026/problems/m01_matmul_512')
 meta=json.loads((p/'meta.json').read_text())['kernels'][0]
 g=Graph.from_dict(json.loads(next(p.glob('*.ir.json')).read_text()))
 t=g.nodes['matmul'].op.body[0];w=tensor_core._walk(t.body)
 print('smem', {k:tuple(int(tensor_core._static(x)) for x in v.extents) for k,v in w.smem.items()})
 for sm,copies in w.copies.items():
  print('\n',sm)
  for cp,ch,seq in copies:
   print('copy seq',seq,'w',cp.nbytes//4,'chain',[(l.axis.name,getattr(l,'start',None),getattr(l,'step',None),getattr(l,'end',None)) for l in ch])
   # group by lane env tid values 0..511 and k values; print unique source/dest forms as strings for first 10
   seen=set(); n=0
   for tid in range(512):
    env,_,_=tensor_core._decode_env(t,{},0,tid)
    env['bid']=0
    vals=[None]
    if ch:
     l=ch[0]; start=ev(l.start,env); step=ev(l.step,env); end=ev(l.end,env) if l.end is not None else int(tensor_core._static(l.axis.extent)); vals=list(range(start,end,step))
    for val in vals:
     if val is not None: env[ch[0].axis.name]=val
     d=tensor_core._copy_dest(cp)
     ds=tuple(ev(x,env) for x in d); ss=tuple(ev(x,env) for x in cp.src_index)
     key=(ds[0],ds[1],ss[0],ss[1],cp.nbytes//4)
     if key not in seen:
      seen.add(key); print(' ',key); n+=1
      if n>30: break
    if n>30: break

run()
