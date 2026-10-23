"""Parent-owned composed softplus F32-tail oracle; no compiler imports/calls."""
from pathlib import Path
import json,sys,math,hashlib
import numpy as np
from emmy.compiler.graph import Graph
from emmy.compiler.tensor import Tensor
from emmy.compiler.dtype import F32
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.kernel.ir import KernelOp,Tile,Load,Write
from tools.research_cases import _v,_l,_assign,canonical,digest,save_arrays,validate_case
VERSION='softplus-f32-negative-tail-composition-v1'
def generate(root,seed):
 root=Path(root);root.mkdir(parents=True,exist_ok=False);rng=np.random.default_rng(seed%(1<<64));n=int(rng.choice([35,65,97]));suffix=str(int(rng.integers(100000,999999)))
 x='x_'+suffix;amp='amp_'+suffix;out='out_'+suffix;entry='k_'+suffix;loaded='value_'+suffix;gain='gain_'+suffix;soft='soft_'+suffix;scaled='scaled_'+suffix;axis='cell_'+suffix
 amplifier=float(np.float32(1e30));values=rng.uniform(-60.,50.,size=n).astype(np.float32)
 special=np.asarray([-20.,-24.,-16.,-15.,-8.,-4.,-3.,-2.772588722239781,-2.75,-1.,-0.,0.,1.,8.,20.,40.,80.,-80.,-16.635532333438686],np.float32);values[:len(special)]=special
 body=(Load(loaded,x,(_v(axis),),dtype=F32),Load(gain,amp,(),dtype=F32),_assign(soft,'softplus',loaded),_assign(scaled,'multiply',soft,gain),Write(out,(_v(axis),_l(0)),soft,value_dtype=F32),Write(out,(_v(axis),_l(1)),scaled,value_dtype=F32))
 graph=Graph();graph.add_node(InputOp(),[],Tensor(x,(n,),F32),node_id=x);graph.add_node(InputOp(),[],Tensor(amp,(),F32),node_id=amp);graph.inputs=[x,amp];graph.add_node(KernelOp(body=(Tile((Axis(axis,n),),body,block_threads=32),),name=entry),[x,amp],Tensor(out,(n,2),F32),node_id=out);graph.outputs=[out];graph.validate();raw=graph.to_dict();Graph.from_dict(raw).validate()
 soft64=np.logaddexp(np.float64(0.),values.astype(np.float64));soft32=soft64.astype(np.float32);other=np.asarray([np.float32(math.log1p(math.exp(float(v))))for v in values],np.float32);assert np.array_equal(soft32.view(np.uint32),other.view(np.uint32))
 amplified=np.multiply(soft32,np.float32(amplifier),dtype=np.float32);expected=np.stack([soft32,amplified],axis=1);assert np.isfinite(expected).all()
 case=root/'softplus-and-amplified-tail';case.mkdir();(case/(entry+'.ir.json')).write_bytes(canonical(raw))
 args=[{'name':amp,'dtype':'f32','shape':[],'scalar':True,'value':amplifier},{'name':out,'dtype':'f32','shape':[n,2],'output':True},{'name':x,'dtype':'f32','shape':[n]}];rotation=int(rng.integers(0,3));args=args[rotation:]+args[:rotation]
 meta={'id':VERSION,'graph_inputs':[x,amp],'graph_outputs':[out],'rtol':1e-5,'atol':1e-7,'kernels':[{'node':out,'entry':entry,'arg_order':[a['name']for a in args],'args':args,'grid':[math.ceil(n/32),1,1],'block':[32,1,1],'smem_bytes':0,'zero_outputs':[],'final':True}]};(case/'meta.json').write_bytes(canonical(meta))
 save_arrays(case/'inputs.npz',{x:values});save_arrays(case/'expected.npz',{out:expected});files={p.name:digest(p.read_bytes())for p in sorted(case.iterdir())}
 spec={'version':VERSION,'family':'pointwise','seed':int(seed),'shape':[n,2],'amplifier':amplifier,'names':{'input':x,'amplifier':amp,'out':out,'soft':soft,'scaled':scaled,'axis':axis},'semantic_contract':'F32softplus per publiclogaddexp(0,x), roundedtoF32beforeordinaryF32multiply withscalaramplifier; finite-negative-tail information mustsurvive beforecomposition. Independent references modeltypedoperationboundary andfixedordinary1e-5/1e-7 tolerances.'}
 record={'version':VERSION,'id':digest(canonical(files)),'spec':spec,'files':files,'ir_paths':[entry+'.ir.json'],'requires_isolated_first':True,'requires_sanitizer':False,'tolerance_justification':'Fixed BEFOREcompilerexecution:rtol1e-5/atol1e-7, finiteF32inputs[-80,80], noNan/Inf/overflow/subnormaloutputreference. F64 NumPylogaddexp→F32cast andindependentPythonlog1p(exp)→F32 exactlyagree; scaled columnroundsF32multiplyafterF32softplus. At-20/-24 mathematicalsoftplus values differ; ordinaryscalar1e30 amplificationmakeslossobservablewithloosefixedrelativeF32tolerance, no arbitrarytightzerotolerance. Positive,zero,transition,tail andpartialTile controls included; everyoutputwritten. ScalarABI explicitrotation. No fixture/compiler/reference-lowering reuse.'}
 (case/'case.json').write_bytes(canonical(record));validate_case(case);result={'version':VERSION,'seed':int(seed),'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':[str(case)]};(root/'case-list.json').write_bytes(canonical(result));return result
if __name__=='__main__':print(json.dumps(generate(sys.argv[1],int(sys.argv[2]))))
