"""Parent-owned scalar BOOL SSA oracle, F32 buffers only, explicit seedFalse folds."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
from emmy.compiler.graph import Graph
from emmy.compiler.tensor import Tensor
from emmy.compiler.dtype import F32,BOOL
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.kernel.ir import KernelOp,Tile,Load,Write,Assign,Loop,Accum
from emmy.compiler.ir.stmt import Init
from emmy.compiler.ir.stmt.leaves import Let
from emmy.compiler.ir.elementwise import ElementwiseImpl
from tools.research_cases import _v,_l,_b,canonical,digest,save_arrays,validate_case
VERSION='scalar-bool-ssa-independent-oracle-v2'
def generate(root,seed):
 root=Path(root);root.mkdir(parents=True,exist_ok=False);rng=np.random.default_rng(seed%(1<<64));cases=[];checks=[]
 for label in ['scalar-relations','four-folds-seed-false','empty-folds-seed-false']:
  block=int(rng.choice([32,64]));n=block+int(rng.choice([1,7,13]));suffix=str(int(rng.integers(100000,999999)));x='x_'+suffix;y='y_'+suffix;out='out_'+suffix;entry='k_'+suffix;row=_v('row');g=Graph()
  def add_input(name,shape):
   g.add_node(InputOp(),[],Tensor(name,shape,F32),node_id=name);g.inputs.append(name)
  if label=='scalar-relations':
   add_input(x,(n,));add_input(y,(n,));a=rng.integers(-1,2,size=n).astype(np.float32);b=rng.integers(-1,2,size=n).astype(np.float32);a[:4]=[-1,-1,1,1];b[:4]=[-1,1,-1,1];inputs={x:a,y:b}
   body=[Load('a',x,(row,),dtype=F32),Load('b',y,(row,),dtype=F32),Let('left',_b('>',_v('a'),_l(0.)),dtype=BOOL),Let('right',_b('>',_v('b'),_l(0.)),dtype=BOOL),Assign('comparison',ElementwiseImpl('less'),('left','right'),BOOL)]
   relations=[('<','expr_less'),('<=','expr_less_equal'),('>','expr_greater'),('>=','expr_greater_equal')]
   for op,name in relations:body.append(Let(name,_b(op,_v('left'),_v('right')),dtype=BOOL))
   names=['left','right','comparison']+[name for op,name in relations]
   for col,name in enumerate(names):body.extend((Let('float_'+name,_v(name),dtype=F32),Write(out,(row,_l(col)),'float_'+name,value_dtype=F32)))
   left=a>0;right=b>0
   expected=np.stack((left,right,np.logical_and(np.logical_not(left),right),np.less(left,right),np.less_equal(left,right),np.greater(left,right),np.greater_equal(left,right)),axis=1).astype(np.float32)
   scalar=[]
   for i in range(n):
    sa=float(a[i])>0;sb=float(b[i])>0
    scalar.append([float(sa),float(sb),float((not sa)and sb),float(sa<sb),float(sa<=sb),float(sa>sb),float(sa>=sb)])
   scalar=np.asarray(scalar,np.float32);cols=7;trips=None;initial=None
  else:
   trips=7 if label=='four-folds-seed-false' else 0;width=max(trips,1);add_input(x,(n,width));a=rng.integers(-1,2,size=(n,width)).astype(np.float32)
   for i in range(n):
    if i%4==0:a[i].fill(-1)
    elif i%4==1:a[i].fill(-1);a[i,0]=1
    elif i%4==2:a[i].fill(1)
    else:a[i].fill(1);a[i,-1]=-1
   inputs={x:a};names=['add_carrier','multiply_carrier','maximum_carrier','minimum_carrier'];ops=['add','multiply','maximum','minimum'];initial=[False,True,False,True] if trips else [True,False,True,False]
   inner=[Load('a',x,(row,_v('iter')),dtype=F32),Let('value',_b('>',_v('a'),_l(0.)),dtype=BOOL)]
   inner.extend(Accum(name,'value',ElementwiseImpl(op),BOOL,axes=('iter',)) for name,op in zip(names,ops))
   body=[Init(name,float(value),dtype=BOOL) for name,value in zip(names,initial)];body.append(Loop(Axis('iter',trips),tuple(inner),seed=False))
   for col,name in enumerate(names):body.extend((Let('float_'+name,_v(name),dtype=F32),Write(out,(row,_l(col)),'float_'+name,value_dtype=F32)))
   if trips:
    truth=a>0;expected=np.stack((truth.any(axis=1),truth.all(axis=1),truth.any(axis=1),truth.all(axis=1)),axis=1).astype(np.float32)
   else:expected=np.broadcast_to(np.asarray(initial,np.float32),(n,4)).copy()
   scalar=np.empty((n,4),np.float32)
   for i in range(n):
    state=list(initial)
    for k in range(trips):
     v=bool(float(a[i,k])>0);state=[bool(int(state[0])+int(v)),bool(int(state[1])*int(v)),bool(max(int(state[2]),int(v))),bool(min(int(state[3]),int(v)))]
    scalar[i]=state
   cols=4
  np.testing.assert_array_equal(expected,scalar);g.add_node(KernelOp(body=(Tile((Axis('row',n),),tuple(body),block_threads=block),),name=entry),list(g.inputs),Tensor(out,(n,cols),F32),node_id=out);g.outputs=[out];g.validate();raw=g.to_dict();Graph.from_dict(raw).validate()
  case=root/label;case.mkdir();(case/(entry+'.ir.json')).write_bytes(canonical(raw));args=[{'name':out,'dtype':'f32','shape':[n,cols],'output':True}]+[{'name':name,'dtype':'f32','shape':list(value.shape)} for name,value in inputs.items()]
  meta={'id':VERSION+'-'+label,'graph_inputs':list(inputs),'graph_outputs':[out],'rtol':0.,'atol':0.,'kernels':[{'node':out,'entry':entry,'arg_order':[a['name'] for a in args],'args':args,'grid':[2,1,1],'block':[block,1,1],'smem_bytes':0,'zero_outputs':[],'final':True}]}
  (case/'meta.json').write_bytes(canonical(meta));save_arrays(case/'inputs.npz',inputs);save_arrays(case/'expected.npz',{out:expected});files={p.name:digest(p.read_bytes()) for p in sorted(case.iterdir())}
  record={'version':VERSION,'id':digest(canonical(files)),'spec':{'family':'pointwise' if label=='scalar-relations' else 'reduction','schedule':label,'seed':int(seed),'n':n,'block':block,'trips':trips,'initial':initial,'semantic_contract':'scalar-BOOL-relational-Assign-and-BinaryExpr' if label=='scalar-relations' else 'explicit-scalar-BOOL-carrier'},'files':files,'ir_paths':[entry+'.ir.json'],'requires_isolated_first':True,'requires_sanitizer':False,'tolerance_justification':'Public DataType BOOL scalar SSA only. F32 input comparisons make BOOL values, allBOOL Assign less plus fourordered BinaryExpr relations and explicitInit/seedFalse fourcanonicalfolds. BOOL-to-F32 Let snapshots output bits0/1; allbuffersandABI F32. NumPy logicalcomparison/any/all or explicitInit reference independentlycrosschecked Pythoninteger arithmetic+perstepbool casts. EmptyLoop0 preserves explicit nonneutralInit withnonemptydummyinput. PartialfinalCTA noSync; everyoutputwritten. Mixednumeric/default-seed/BOOLbuffersemantics excluded.'}
  (case/'case.json').write_bytes(canonical(record));validate_case(case);cases.append(str(case));checks.append({'label':label,'n':n,'block':block,'trips':trips,'initial':initial,'reference_words':int(expected.size),'bitwise_reference_equal':True,'boolean_outputs':bool(np.all((expected==0)|(expected==1)))})
 result={'version':VERSION,'seed':int(seed),'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':cases};(root/'case-list.json').write_bytes(canonical(result));(root/'validation.json').write_bytes(canonical(checks));return result
if __name__=='__main__':print(json.dumps(generate(sys.argv[1],int(sys.argv[2]))))
