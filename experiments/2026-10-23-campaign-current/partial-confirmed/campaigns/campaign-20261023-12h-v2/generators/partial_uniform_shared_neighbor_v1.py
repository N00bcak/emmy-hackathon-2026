"""Parent-owned partial CTA finite uniform-loop shared-memory oracle v1."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
from emmy.compiler.graph import Graph
from emmy.compiler.tensor import Tensor
from emmy.compiler.dtype import F32
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import FuncCallExpr
from emmy.compiler.ir.kernel.ir import KernelOp,Tile,Load,Write,Loop,Accum,Smem,Sync
from emmy.compiler.ir.stmt import Init
from emmy.compiler.ir.stmt.blocks import StridedLoop
from tools.research_cases import _v,_l,_b,canonical,digest,save_arrays,validate_case
VERSION='partial-uniform-shared-neighbor-oracle-v1'
def generate(root,seed):
 root=Path(root);root.mkdir(parents=True,exist_ok=False);rng=np.random.default_rng(seed%(1<<64));cases=[];checks=[]
 for kind in ['loop','nested-strided-loop']:
  block=int(rng.choice([32,64]));ctas=int(rng.choice([2,3]));tail=int(rng.choice([1,block//2-1]));n=ctas*block-tail;repeats=int(rng.choice([3,5,7]))
  suffix=str(int(rng.integers(100000,999999)));x='x_'+suffix;out='out_'+suffix;qname='q_'+suffix;entry='k_'+suffix
  lane=_v('lane');slot=_b('%',lane,_l(block));base=_b('*',_b('//',lane,_l(block)),_l(block));valid=FuncCallExpr('minimum',(_l(block),_b('-',_l(n),base)));neighbor=_b('%',_b('+',slot,_l(1)),valid)
  consume=(Load('value',x,(lane,),dtype=F32),Write(qname,(slot,),'value',value_dtype=F32),Sync(),Load('neighbor',qname,(neighbor,),dtype=F32),Accum('acc','neighbor','add',F32,axes=('consume' if kind=='nested-strided-loop' else 'repeat',)),Sync())
  if kind=='loop':loop=Loop(Axis('repeat',repeats),consume,seed=False)
  else:loop=StridedLoop(Axis('repeat',2*repeats+1),_l(1),_l(2),(Loop(Axis('consume',1),consume,seed=False),),seed=False)
  body=(Smem(qname,(block,),dtype='float',align=16),Init('acc',0.,dtype=F32),loop,Write(out,(lane,),'acc',value_dtype=F32))
  graph=Graph();graph.add_node(InputOp(),[],Tensor(x,(n,),F32),node_id=x);graph.inputs=[x]
  graph.add_node(KernelOp(body=(Tile((Axis('lane',n),),body,block_threads=block),),name=entry),[x],Tensor(out,(n,),F32),node_id=out);graph.outputs=[out];graph.validate();raw=graph.to_dict();Graph.from_dict(raw).validate()
  values=rng.integers(-8,9,size=n).astype(np.float32);indices=np.arange(n,dtype=np.int64);bases=(indices//block)*block;valids=np.minimum(block,n-bases);neighbors=bases+(indices%block+1)%valids;expected=(values[neighbors].astype(np.int64)*repeats).astype(np.float32)
  scalar=np.asarray([sum(int(values[(i//block)*block+((i%block+1)%min(block,n-(i//block)*block))]) for _ in range(repeats)) for i in range(n)],np.float32);np.testing.assert_array_equal(expected,scalar)
  case=root/kind;case.mkdir();(case/(entry+'.ir.json')).write_bytes(canonical(raw));args=[{'name':out,'dtype':'f32','shape':[n],'output':True},{'name':x,'dtype':'f32','shape':[n]}]
  meta={'id':VERSION+'-'+kind,'graph_inputs':[x],'graph_outputs':[out],'rtol':0.,'atol':0.,'kernels':[{'node':out,'entry':entry,'arg_order':[a['name'] for a in args],'args':args,'grid':[ctas,1,1],'block':[block,1,1],'smem_bytes':block*4,'zero_outputs':[],'final':True}]}
  (case/'meta.json').write_bytes(canonical(meta));save_arrays(case/'inputs.npz',{x:values});save_arrays(case/'expected.npz',{out:expected});files={p.name:digest(p.read_bytes()) for p in sorted(case.iterdir())}
  record={'version':VERSION,'id':digest(canonical(files)),'spec':{'family':'reduction','schedule':kind,'seed':int(seed),'n':n,'block':block,'ctas':ctas,'tail':tail,'repeats':repeats,'types':['partial-cta-finite-uniform-loop-barrier']},'files':files,'ir_paths':[entry+'.ir.json'],'requires_isolated_first':True,'requires_sanitizer':True,'tolerance_justification':'Exact small F32 integer sum of cyclic-next-valid-lane shared slot over finite uniform loop. Last CTA partial, next valid lane wraps within each CTA active subset. WriteQ/barrier/readNeighbor/barrier every iteration tests cross-lane communication and converged physical participation. Integer NumPy gather/repeat independently crosschecked scalar Python sums. All logical outputs written once; padding buffers guarded; output-first ABI; no compiler or renderer in oracle.'}
  (case/'case.json').write_bytes(canonical(record));validate_case(case);cases.append(str(case));checks.append({'kind':kind,'n':n,'block':block,'ctas':ctas,'padding':tail,'repeats':repeats,'neighbor_bounds':bool(np.all((neighbors>=bases)&(neighbors<np.minimum(bases+block,n)))),'integer_reference_equal':True})
 result={'version':VERSION,'seed':int(seed),'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':cases};(root/'case-list.json').write_bytes(canonical(result));(root/'validation.json').write_bytes(canonical(checks));return result
if __name__=='__main__':print(json.dumps(generate(sys.argv[1],int(sys.argv[2]))))
