"""CPU-only baseline tests and legal fixtures constructed from typed IR nodes."""
import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import tempfile
import unittest
from emmy.compiler.graph import Graph
from emmy.compiler.dim import Dim

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE/'emmy-hackathon-2026/problems'
spec = importlib.util.spec_from_file_location('baseline_compiler', HERE/'compile.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def fixture(problem, destination, *, tile_extent=None, data_extent=None, arg_reverse=False):
    source = next((ROOT/problem).glob('*.ir.json'))
    graph = Graph.from_dict(json.loads(source.read_text()))
    meta = json.loads(source.with_name('meta.json').read_text())
    kernel = next(n for n in graph.nodes.values() if type(n.op).__name__ == 'KernelOp')
    if data_extent is not None:
        for node in graph.nodes.values():
            node.outputs = tuple(replace(t, shape=(Dim(data_extent),)) if len(t.shape)==1 and mod.static(t.shape[0])>1 else t for t in node.outputs)
        for arg in meta['kernels'][0]['args']:
            if len(arg['shape'])==1 and arg['shape'][0]>1:
                arg['shape']=[data_extent]
    tile = kernel.op.body[0]
    if tile_extent is not None:
        tile = replace(tile, axes=(replace(tile.axes[0], extent=Dim(tile_extent)),))
        meta['kernels'][0]['grid'] = [(tile_extent+meta['kernels'][0]['block'][0]-1)//meta['kernels'][0]['block'][0],1,1]
    if problem.startswith('r') and data_extent is not None:
        loop=tile.body[0]
        loop=replace(loop, axis=replace(loop.axis, extent=Dim(data_extent)))
        tile=replace(tile, body=(loop,)+tile.body[1:])
    kernel.op=replace(kernel.op, body=(tile,))
    if arg_reverse:
        meta['kernels'][0]['arg_order'].reverse()
        byname={a['name']:a for a in meta['kernels'][0]['args']}
        meta['kernels'][0]['args']=[byname[n] for n in meta['kernels'][0]['arg_order']]
    destination.mkdir(exist_ok=True)
    path=destination/source.name
    path.write_text(json.dumps(graph.to_dict(),indent=2,default=repr))
    path.with_name('meta.json').write_text(json.dumps(meta,indent=2))
    return path


def typed_pointwise(body, destination, *, extent=4, out_shape=(4,), block=4):
    """Write a small typed-IR graph whose Tile body is supplied by the test."""
    from emmy.compiler.ir.axis import Axis
    from emmy.compiler.ir.base import InputOp
    from emmy.compiler.ir.kernel.ir import KernelOp, Tile
    from emmy.compiler.tensor import Tensor
    from emmy.compiler.dtype import F32
    graph=Graph()
    graph.add_node(InputOp(),[],Tensor('x',(extent,),F32),node_id='x')
    graph.add_node(KernelOp(body=(Tile((Axis('i',extent),),tuple(body),block_threads=block),),name='k'),
                   ['x'],Tensor('out',tuple(out_shape),F32),node_id='out')
    destination.mkdir(parents=True, exist_ok=True)
    path=destination/'typed.ir.json'
    path.write_text(json.dumps(graph.to_dict(),indent=2,default=repr))
    path.with_name('meta.json').write_text(json.dumps({'graph_inputs':['x'],'graph_outputs':['out'],
        'rtol':0,'atol':0,'kernels':[{'node':'out','entry':'k','arg_order':['x','out'],
        'grid':[1,1,1],'block':[block,1,1],'smem_bytes':0,'zero_outputs':[],
        'args':[{'name':'x','dtype':'f32','shape':[extent]},
                {'name':'out','dtype':'f32','shape':list(out_shape),'output':True}]}]}))
    return path


class CompilerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.out = Path(self.temporary.name)

    def assemble(self,path):
        result=subprocess.run([sys.executable,str(HERE/'compile.py'),str(path)],capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)
        ptx=path.with_suffix('.ptx')
        ptx.write_text(result.stdout)
        assembled=subprocess.run(['ptxas','-arch=sm_89',str(ptx),'-o',str(ptx.with_suffix('.cubin'))],capture_output=True,text=True,timeout=30)
        self.assertEqual(assembled.returncode,0,assembled.stderr)
        return result.stdout

    def test_nonfixture_pointwise_tail_and_reordered_abi(self):
        p=fixture('p00_scalar_add',self.out/'pointwise14',tile_extent=7,data_extent=14,arg_reverse=True)
        out=self.assemble(p)
        self.assertIn('setp.ge.u32',out)
        self.assertIn(', 7;',out)
        self.assertEqual(mod.compile_path(p),out)

    def test_nonfixture_reduction_tail(self):
        p=fixture('r01_array_sum',self.out/'sum257',data_extent=257,arg_reverse=True)
        out=self.assemble(p)
        self.assertIn(', 257;',out)
        self.assertIn('shfl.sync.bfly.b32',out)
        self.assertIn('bar.sync 0',out)

    def test_partial_collective_tile_keeps_all_threads_in_barrier(self):
        p=fixture('r01_array_sum',self.out/'invalid-collective',tile_extent=127)
        out=self.assemble(p)
        # Extra threads stay alive for the CTA-wide barrier, but loads and
        # carried folds must use the complement, the inside-Tile predicate.
        self.assertIn('setp.ge.u32',out)
        self.assertIn('setp.lt.u32',out)
        self.assertRegex(out, r'@%p\d+ ld\.global\.f32')
        self.assertIn('bar.sync 0',out)
        self.assertNotIn('bra kernel_done',out)

    def test_nested_collective_partial_cta_masks_memory(self):
        from emmy.compiler.ir.kernel.ir import Cond
        from emmy.compiler.ir.expr import BinaryExpr, Literal
        path=fixture('r01_array_sum',self.out/'invalid-nested-collective',tile_extent=127)
        graph=Graph.from_dict(json.loads(path.read_text()))
        node=graph.nodes['sum_1']
        tile=node.op.body[0]
        # Nested collectives still need every launched thread alive, and loads
        # must not read outside the active Tile cells.
        cond=Cond(BinaryExpr('==',Literal(0,'int'),Literal(0,'int')),tile.body,())
        node.op=replace(node.op,body=(replace(tile,body=(cond,)),))
        path.write_text(json.dumps(graph.to_dict(),indent=2,default=repr))
        out=self.assemble(path)
        self.assertIn('@!%p',out)
        self.assertIn('bar.sync 0',out)

    def test_reject_tree_indexed_by_lane(self):
        path=fixture('r01_array_sum',self.out/'invalid-tree-mapping')
        graph=Graph.from_dict(json.loads(path.read_text()))
        node=graph.nodes['sum_1']
        tile=node.op.body[0]
        body=tuple(replace(s,tid_var='lane') if type(s).__name__=='TreeHalve' else s for s in tile.body)
        node.op=replace(node.op,body=(replace(tile,body=body),))
        path.write_text(json.dumps(graph.to_dict(),indent=2,default=repr))
        with self.assertRaisesRegex(mod.Unsupported,'warp-indexed TreeHalve'):
            mod.compile_path(path)

    def test_reject_invalid_abi(self):
        p=fixture('p00_scalar_add',self.out/'invalid-abi')
        meta=json.loads(p.with_name('meta.json').read_text())
        meta['kernels'][0]['arg_order']=['x0','x0','add']
        p.with_name('meta.json').write_text(json.dumps(meta))
        with self.assertRaisesRegex(mod.Unsupported,'Duplicate'):
            mod.compile_path(p)

    def test_scalar_by_value_argument_binds_as_ssa(self):
        p=fixture('p00_scalar_add',self.out/'scalar-abi',tile_extent=1,data_extent=1)
        graph=Graph.from_dict(json.loads(p.read_text()))
        node=graph.nodes['add']
        tile=node.op.body[0]
        load0, load1, assign, *_ = tile.body
        # Replace x0's memory load with a by-value metadata scalar. The
        # parameter must be usable directly, not registered as a buffer.
        for statement in tile.body:
            if type(statement).__name__ == 'Assign':
                assign=replace(statement,args=('scale',load1.names[0]))
                tile=replace(tile,body=(load1,assign))
                break
        node.op=replace(node.op,body=(tile,))
        p.write_text(json.dumps(graph.to_dict(),indent=2,default=repr))
        meta=json.loads(p.with_name('meta.json').read_text())
        k=meta['kernels'][0]
        k['arg_order']=[name for name in k['arg_order'] if name!='x0']
        k['args']=[a for a in k['args'] if a['name']!='x0']
        k['args'].insert(0,{'name':'scale','dtype':'f32','scalar':True,'value':2.0})
        k['arg_order'].insert(0,'scale')
        p.with_name('meta.json').write_text(json.dumps(meta,indent=2))
        out=self.assemble(p)
        self.assertIn('.param .f32 param_0',out)
        self.assertIn('ld.param.f32',out)
        self.assertIn('add.f32',out)

    def test_nonunit_block_and_grid_dimensions_flatten(self):
        p=fixture('p00_scalar_add',self.out/'multi-dim-launch',tile_extent=128,data_extent=256)
        meta=json.loads(p.with_name('meta.json').read_text())
        meta['kernels'][0]['block']=[64,2,1]
        meta['kernels'][0]['grid']=[1,2,1]
        p.with_name('meta.json').write_text(json.dumps(meta,indent=2))
        out=self.assemble(p)
        self.assertIn('mad.lo.u32',out)
        self.assertIn('%tid.y',out)
        self.assertIn('%ctaid.y',out)

    def test_partial_cta_non_sum_reduction_masks_fold(self):
        from emmy.compiler.ir.elementwise import ElementwiseImpl
        p=fixture('r01_array_sum',self.out/'partial-max',tile_extent=127,data_extent=127)
        graph=Graph.from_dict(json.loads(p.read_text()))
        node=graph.nodes['sum_1']
        tile=node.op.body[0]
        body=[]
        for s in tile.body:
            if type(s).__name__ == 'StridedLoop':
                inner=tuple(replace(c,op=ElementwiseImpl('maximum')) if type(c).__name__=='Accum' else c
                            for c in s.body)
                s=replace(s,body=inner)
            elif type(s).__name__ in ('WarpShuffle','TreeHalve'):
                s=replace(s,combine_states=tuple(replace(c,op=ElementwiseImpl('maximum')) for c in s.combine_states))
            body.append(s)
        node.op=replace(node.op,body=(replace(tile,body=tuple(body)),))
        p.write_text(json.dumps(graph.to_dict(),indent=2,default=repr))
        out=self.assemble(p)
        self.assertIn('max.f32',out)
        # Inactive lanes must not fold their masked load into the maximum.
        self.assertRegex(out, r'@%p\d+ max\.f32')
        self.assertRegex(out, r'@%p\d+ mov\.f32')

    def test_log_lowering_is_natural_log(self):
        from emmy.compiler.ir.elementwise import ElementwiseImpl
        p=fixture('p00_scalar_add',self.out/'log-natural',tile_extent=1,data_extent=1)
        graph=Graph.from_dict(json.loads(p.read_text()))
        node=graph.nodes['add']
        tile=node.op.body[0]
        load=tile.body[0]
        from emmy.compiler.ir.stmt.leaves import Assign
        logv=Assign('logv',ElementwiseImpl('log'),(load.names[0],),None)
        node.op=replace(node.op,body=(replace(tile,body=(load,logv)),))
        p.write_text(json.dumps(graph.to_dict(),indent=2,default=repr))
        out=self.assemble(p)
        self.assertIn('lg2.approx.f32',out)
        self.assertIn('0f3f317218',out)  # ln(2), not a base-2 result

    def test_reduce_aliases_canonicalize(self):
        from emmy.compiler.ir.elementwise import ElementwiseImpl
        p=fixture('r01_array_sum',self.out/'reduce-alias',tile_extent=127,data_extent=127)
        graph=Graph.from_dict(json.loads(p.read_text()))
        node=graph.nodes['sum_1']
        tile=node.op.body[0]
        body=[]
        for s in tile.body:
            if type(s).__name__ == 'StridedLoop':
                s=replace(s,body=tuple(replace(c,op=ElementwiseImpl('sum'))
                                       if type(c).__name__=='Accum' else c for c in s.body))
            elif type(s).__name__ in ('WarpShuffle','TreeHalve'):
                s=replace(s,combine_states=tuple(replace(c,op=ElementwiseImpl('add')) for c in s.combine_states))
            body.append(s)
        node.op=replace(node.op,body=(replace(tile,body=tuple(body)),))
        p.write_text(json.dumps(graph.to_dict(),indent=2,default=repr))
        out=self.assemble(p)
        self.assertIn('add.f32',out)

    def test_integer_copy_is_a_typed_register_move(self):
        from emmy.compiler.dtype import I32
        from emmy.compiler.ir.axis import Axis
        from emmy.compiler.ir.elementwise import ElementwiseImpl
        from emmy.compiler.ir.expr import Var
        from emmy.compiler.ir.kernel.ir import KernelOp, Load, Tile, Write
        from emmy.compiler.ir.stmt.leaves import Assign
        from emmy.compiler.ir.base import InputOp
        from emmy.compiler.tensor import Tensor

        graph=Graph()
        shape=(4,)
        graph.add_node(InputOp(),[],Tensor('x',shape,I32),node_id='x')
        index=(Var('a0'),)
        body=(Load('value','x',index,dtype=I32),
              Assign('copied',ElementwiseImpl('copy'),('value',),I32),
              Write('out',index,'copied',value_dtype=I32))
        graph.add_node(KernelOp(body=(Tile((Axis('a0',4),),body,block_threads=4),),name='k_copy'),
                       ['x'],Tensor('out',shape,I32),node_id='out')
        path=self.out/'integer-copy.ir.json'
        path.write_text(json.dumps(graph.to_dict(),indent=2,default=repr))
        path.with_name('meta.json').write_text(json.dumps({'kernels':[{
            'node':'out','entry':'k_copy','arg_order':['x','out'],
            'grid':[1,1,1],'block':[4,1,1],'smem_bytes':0,'zero_outputs':[],
            'args':[{'name':'x','dtype':'i32','shape':[4]},
                    {'name':'out','dtype':'i32','shape':[4],'output':True}]}]},indent=2))
        out=self.assemble(path)
        self.assertIn('ld.global.s32',out)
        self.assertIn('mov.s32',out)
        self.assertIn('st.global.s32',out)

    def test_write_uses_output_width_and_converts_value_dtype(self):
        from emmy.compiler.dtype import F32, F64
        from emmy.compiler.ir.axis import Axis
        from emmy.compiler.ir.expr import Var
        from emmy.compiler.ir.kernel.ir import KernelOp, Load, Tile, Write
        from emmy.compiler.ir.stmt.leaves import Assign
        from emmy.compiler.ir.elementwise import ElementwiseImpl
        from emmy.compiler.ir.base import InputOp
        from emmy.compiler.tensor import Tensor
        graph=Graph(); shape=(4,); graph.add_node(InputOp(),[],Tensor('x',shape,F32),node_id='x')
        index=(Var('i'),)
        body=(Load('v','x',index,dtype=F32),
              Assign('wide',ElementwiseImpl('copy'),('v',),F64),
              Write('y',index,'wide',value_dtype=F64))
        graph.add_node(KernelOp(body=(Tile((Axis('i',4),),body,block_threads=4),),name='k_convert'),
                       ['x'],Tensor('y',shape,F64),node_id='y')
        path=self.out/'write-convert.ir.json'; path.write_text(json.dumps(graph.to_dict(),indent=2,default=repr))
        path.with_name('meta.json').write_text(json.dumps({'kernels':[{'node':'y','entry':'k_convert',
            'arg_order':['x','y'],'grid':[1,1,1],'block':[4,1,1],'smem_bytes':0,'zero_outputs':[],
            'args':[{'name':'x','dtype':'f32','shape':[4]},{'name':'y','dtype':'f64','shape':[4],'output':True}]}]}))
        out=self.assemble(path)
        self.assertIn('cvt.f64.f32',out)
        self.assertIn('st.global.f64',out)
        self.assertNotIn('st.global.f32',out)

    def test_vector_copy_has_runtime_fallback_and_scalar_negative(self):
        aligned=fixture('m02_matmul_rect',self.out/'vector-copy-aligned')
        out=self.assemble(aligned)
        self.assertIn('cp.async.cg.shared.global',out)
        self.assertIn('cp.async.commit_group',out)
        self.assertIn('cp.async.wait_group 0',out)
        self.assertIn('copy_scalar_',out)
        # The runtime check is deliberately not an unchecked pointer-alignment
        # assumption; it branches to the scalar path on either address.
        self.assertIn('and.b64',out)
        self.assertIn('setp.eq.u64',out)

    def test_unaligned_copy_index_stays_scalar(self):
        path=fixture('m02_matmul_rect',self.out/'vector-copy-unaligned')
        graph=Graph.from_dict(json.loads(path.read_text()))
        node=next(n for n in graph.nodes.values() if type(n.op).__name__=='KernelOp')
        from emmy.compiler.ir.expr import BinaryExpr, Literal
        def rewrite_copy(statement):
            if type(statement).__name__ != 'CpAsyncCopy':
                return statement
            offset=BinaryExpr('+',statement.src_index[-1],Literal(1,'int'))
            return replace(statement,src_index=(*statement.src_index[:-1],offset))
        def walk(statements):
            found=False
            changed=[]
            for s in statements:
                if type(s).__name__=='CpAsyncCopy':
                    found=True
                    changed.append(rewrite_copy(s))
                    continue
                for field in ('body','else_body'):
                    nested=getattr(s,field,())
                    if nested:
                        rewritten, nested_found=walk(nested)
                        if nested_found:
                            found=True
                            s=replace(s,**{field:rewritten})
                changed.append(s)
            return tuple(changed), found
        tile=node.op.body[0]
        changed, found=walk(tile.body)
        self.assertTrue(found)
        node.op=replace(node.op,body=(replace(tile,body=changed),))
        path.write_text(json.dumps(graph.to_dict(),indent=2,default=repr))
        out=self.assemble(path)
        self.assertNotIn('cp.async.cg.shared.global',out)

    def test_m01_inner_loop_hoists_invariant_index_division(self):
        """m01's loop-invariant index quotient and remainder are not duplicated.

        m01 reuses the staging quotient/remainder through multiple shared-memory
        addresses. The preheader must compute that maximal index once, and the
        inner body must not recompute it; indices that depend on the induction
        variable legitimately remain in the body.
        """
        import re
        source=next((ROOT/'m01_matmul_512').glob('*.ir.json'))
        graph=Graph.from_dict(json.loads(source.read_text()))
        node=next(n for n in graph.nodes.values() if type(n.op).__name__=='KernelOp')
        tile=next(s for s in node.op.body if type(s).__name__=='Tile')
        outer=next(s for s in tile.body if type(s).__name__ in ('Loop','StridedLoop'))
        inner=next(s for s in outer.body if type(s).__name__ in ('Loop','StridedLoop'))

        def uses_division(expr):
            kind=type(expr).__name__
            if kind=='BinaryExpr':
                return (expr.op in ('/','//','%') or uses_division(expr.left)
                        or uses_division(expr.right))
            if kind=='TernaryExpr':
                return (uses_division(expr.cond) or uses_division(expr.if_true)
                        or uses_division(expr.if_false))
            return False

        names=mod.defined_names(inner.body)|{inner.axis.name}
        expressions=[coord for _buffer,index,_offsets in mod.index_addresses(inner.body)
                     for coord in index]
        counts={}
        by_key={}
        for expression in mod.invariant_occurrences(expressions,names):
            key=mod.structural_key(expression)
            counts[key]=counts.get(key,0)+1
            by_key.setdefault(key,expression)
        division_counts={key:count for key,count in counts.items()
                         if uses_division(by_key[key])}
        self.assertEqual(1,sum(count>1 for count in division_counts.values()))
        single_use=sum(count==1 for count in division_counts.values())

        # m01 now takes the tensor-core path; this test guards the preserved
        # SIMT fallback, so invoke that emitter explicitly.
        metadata=json.loads(source.with_name('meta.json').read_text())['kernels'][0]
        lines=mod.Emitter(graph,node,metadata).render().splitlines()
        labels={}
        for index,line in enumerate(lines):
            match=re.fullmatch(r'\s*([A-Za-z_]\w*):\s*',line)
            if match:
                labels[match.group(1)]=index
        backedges=[]
        for index,line in enumerate(lines):
            match=re.fullmatch(r'\s*bra\s+([A-Za-z_]\w*);\s*',line)
            if match and match.group(1) in labels and labels[match.group(1)]<index:
                backedges.append((labels[match.group(1)],index))
        self.assertGreaterEqual(len(backedges),2)
        start,end=min(backedges,key=lambda span:span[1]-span[0])
        preheader_start=max((i for i in labels.values() if i<start),default=-1)+1
        preheader=lines[preheader_start:start]
        body='\n'.join(lines[start+1:end])

        div_re=re.compile(r'\bdiv\.[su]32\s+(%r\d+)\s*,\s*(%r\d+)\s*,\s*(%r\d+|\d+);')
        rem_re=re.compile(r'\brem\.[su]32\s+(%r\d+)\s*,\s*(%r\d+)\s*,\s*(%r\d+|\d+);')
        divisions=[match.groups() for line in preheader
                   if (match:=div_re.search(line))]
        remainders=[match.groups() for line in preheader
                    if (match:=rem_re.search(line))]
        self.assertEqual(1,len(divisions))
        self.assertEqual(1,len(remainders))
        self.assertEqual(divisions[0][1:],remainders[0][1:])
        for register in (divisions[0][0],remainders[0][0]):
            self.assertNotRegex(body,rf'{re.escape(register)}(?!\w)')

        derived={divisions[0][0],remainders[0][0]}
        changed=True
        while changed:
            changed=False
            for line in preheader:
                registers=re.findall(r'%r\d+',line)
                if len(registers)>1 and any(reg in derived for reg in registers[1:]):
                    if registers[0] not in derived:
                        derived.add(registers[0])
                        changed=True
        self.assertTrue(any(len(re.findall(rf'{re.escape(reg)}(?!\w)',body))>=2
                            for reg in derived))
        self.assertLessEqual(sum(bool(re.search(r'\bdiv\.[su]32\b',line))
                                 for line in lines[start+1:end]),single_use)
        self.assertLessEqual(sum(bool(re.search(r'\brem\.[su]32\b',line))
                                 for line in lines[start+1:end]),single_use)

    def test_cond_definitions_participate_in_hoist_analysis(self):
        from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
        from emmy.compiler.ir.kernel.ir import Cond, Loop
        from emmy.compiler.ir.axis import Axis
        from emmy.compiler.ir.stmt.leaves import Let
        branch=Cond(BinaryExpr('==',Var('lane'),Literal(0,'int')),
                    (Let('off',Literal(0,'int')),),())
        loop=Loop(Axis('k',2),(branch,))
        cond_body=dict(mod.provisional_versions((branch,)))
        self.assertIn('off',mod.defined_names((branch,)))
        self.assertGreater(cond_body.get('off',0),0)
        # A nested definition must also be visible to the enclosing loop pass.
        loop_body=dict(mod.provisional_versions((loop,)))
        self.assertGreater(loop_body.get('off',0),0)

    def test_cond_ssa_bindings_do_not_leak_to_join(self):
        import re
        from emmy.compiler.ir.expr import Literal, Var
        from emmy.compiler.ir.kernel.ir import Load, Write
        from emmy.compiler.ir.stmt.leaves import Let
        from emmy.compiler.ir.expr import BinaryExpr
        from emmy.compiler.ir.kernel.ir import Cond
        # Both arms bind a different value for the same name. The join must
        # use the outer binding; neither arm's local value may escape.
        body=(Let('off',Var('i')),
              Cond(BinaryExpr('==',Var('lane'),Literal(0,'int')),
                   (Let('off',Literal(0,'int')),),
                   (Let('off',Literal(1,'int')),)),
              Load('v','x',(Var('off'),)),
              Write('out',(Var('i'),),'v'))
        out=self.assemble(typed_pointwise(body,self.out/'cond-both-arms'))
        lines=out.splitlines()
        labels={m.group(1):i for i,line in enumerate(lines)
                for m in [re.fullmatch(r'\s*([A-Za-z_]\w*):\s*',line)] if m}
        endif=next(i for name,i in labels.items() if name.startswith('endif_'))
        after='\n'.join(lines[endif+1:])
        self.assertRegex(after,r'mad\.lo\.s32\s+%r\d+\s*,\s*%r\d+\s*,\s*1\s*,\s*0;')
        self.assertNotRegex(after,r'mad\.lo\.s32\s+%r\d+\s*,\s*[01]\s*,\s*1\s*,\s*0;')

    def test_cond_false_arm_uses_outer_binding(self):
        import re
        from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
        from emmy.compiler.ir.kernel.ir import Cond, Load, Write
        from emmy.compiler.ir.stmt.leaves import Let
        body=(Let('off',Var('i')),
              Cond(BinaryExpr('==',Var('lane'),Literal(0,'int')),
                   (Let('off',Literal(0,'int')),),
                   (Load('v','x',(Var('off'),)), Write('out',(Var('i'),),'v'))))
        path=typed_pointwise(body,self.out/'cond-scope')
        out=self.assemble(path)
        lines=out.splitlines()
        labels={m.group(1):i for i,line in enumerate(lines)
                for m in [re.fullmatch(r'\s*([A-Za-z_]\w*):\s*',line)] if m}
        start=next(i for name,i in labels.items() if name.startswith('else_'))
        end=next(i for name,i in labels.items() if name.startswith('endif_'))
        else_body='\n'.join(lines[start+1:end])
        # The false arm must use the outer binding, never the true arm's
        # branch-local zero. The outer index is a runtime register.
        self.assertNotRegex(else_body, r'mad\.lo\.s32\s+%r\d+\s*,\s*1\s*,\s*1\s*,\s*0;')
        self.assertRegex(else_body, r'mad\.lo\.s32\s+%r\d+\s*,\s*%r\d+\s*,\s*1\s*,\s*0;')
        self.assertRegex(out,r'@!%p\d+ bra else_')

    def test_let_rebinding_invalidates_cached_address(self):
        import re
        from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
        from emmy.compiler.ir.kernel.ir import Load, Write
        from emmy.compiler.ir.stmt.leaves import Let
        body=(Let('off',Var('i')),
              Load('first','x',(Var('off'),)),
              Let('off',BinaryExpr('+',Var('i'),Literal(1,'int'))),
              Load('second','x',(Var('off'),)),
              Write('out',(Var('i'),),'second'))
        out=self.assemble(typed_pointwise(body,self.out/'let-rebind'))
        add=re.search(r'add\.s32\s+(%r\d+)\s*,\s*(%r\d+)\s*,\s*1;',out)
        self.assertIsNotNone(add)
        loads=re.findall(r'ld\.global\.f32\s+%f\d+,\s*\[(%rd\d+)\];',out)
        self.assertEqual(2,len(loads))
        self.assertNotEqual(loads[0],loads[1])
        self.assertRegex(out,rf'mad\.lo\.s32\s+%r\d+\s*,\s*{re.escape(add.group(1))}\s*,\s*1\s*,\s*0;')

    def test_residue_rejects_non_multiple_product(self):
        from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
        unsound=BinaryExpr('*',Var('x'),Literal(12,'int'))
        sound=BinaryExpr('*',Var('x'),Literal(16,'int'))
        self.assertIsNone(mod.residue(unsound,8))
        self.assertEqual(0,mod.residue(sound,8))

    def test_unsupported_errors_do_not_emit_partial_ptx(self):
        p=fixture('p00_scalar_add',self.out/'unsupported-abi')
        meta=json.loads(p.with_name('meta.json').read_text())
        meta['kernels'][0]['args'][0].pop('shape',None)
        p.with_name('meta.json').write_text(json.dumps(meta))
        r=subprocess.run([sys.executable,str(HERE/'compile.py'),str(p)],capture_output=True,text=True)
        self.assertNotEqual(r.returncode,0)
        self.assertEqual(r.stdout,'')
        self.assertIn('Unsupported',r.stderr)

    def test_numeric_mask_comparison_and_where(self):
        from emmy.compiler.dtype import F32
        from emmy.compiler.ir.elementwise import ElementwiseImpl
        from emmy.compiler.ir.expr import Literal, Var
        from emmy.compiler.ir.kernel.ir import Load, Write
        from emmy.compiler.ir.stmt.leaves import Assign, Let
        body=(Load('value','x',(Var('i'),)),
              Let('zero',Literal(0.0,'float'),F32),
              Assign('positive',ElementwiseImpl('greater'),('value','zero'),F32),
              Assign('result',ElementwiseImpl('where'),('positive','value','zero'),F32),
              Write('out',(Var('i'),),'result'))
        out=self.assemble(typed_pointwise(body,self.out/'mask-where'))
        self.assertIn('setp.gt.f32',out)
        self.assertIn('selp.f32',out)

    def test_integer_arithmetic_and_bitwise_are_typed(self):
        from emmy.compiler.dtype import I32
        from emmy.compiler.ir.axis import Axis
        from emmy.compiler.ir.base import InputOp
        from emmy.compiler.ir.elementwise import ElementwiseImpl
        from emmy.compiler.ir.expr import Var
        from emmy.compiler.ir.kernel.ir import KernelOp, Load, Tile, Write
        from emmy.compiler.ir.stmt.leaves import Assign
        from emmy.compiler.tensor import Tensor
        graph=Graph(); shape=(4,); index=(Var('i'),)
        graph.add_node(InputOp(),[],Tensor('x',shape,I32),node_id='x')
        body=(Load('value','x',index,dtype=I32),
              Assign('quotient',ElementwiseImpl('floor_divide'),('value','three'),I32),
              Assign('masked',ElementwiseImpl('bitwise_and'),('quotient','seven'),I32),
              Write('out',index,'masked',value_dtype=I32))
        graph.add_node(KernelOp(body=(Tile((Axis('i',4),),body,block_threads=4),),name='k_int_math'),
                       ['x'],Tensor('out',shape,I32),node_id='out')
        path=self.out/'integer-math.ir.json'
        path.write_text(json.dumps(graph.to_dict(),indent=2,default=repr))
        path.with_name('meta.json').write_text(json.dumps({'kernels':[{
            'node':'out','entry':'k_int_math','arg_order':['x','out','three','seven'],
            'grid':[1,1,1],'block':[4,1,1],'smem_bytes':0,'zero_outputs':[],
            'args':[{'name':'x','dtype':'i32','shape':[4]},
                    {'name':'out','dtype':'i32','shape':[4],'output':True},
                    {'name':'three','dtype':'i32','shape':[],'scalar':True,'value':3},
                    {'name':'seven','dtype':'i32','shape':[],'scalar':True,'value':7}]}]},indent=2))
        out=self.assemble(path)
        self.assertIn('div.s32',out)
        self.assertIn('and.b32',out)
        self.assertNotIn('cvt.f32',out)

    def test_int64_floor_division_uses_64bit_scratch_registers(self):
        from emmy.compiler.dtype import I64
        from emmy.compiler.ir.axis import Axis
        from emmy.compiler.ir.base import InputOp
        from emmy.compiler.ir.elementwise import ElementwiseImpl
        from emmy.compiler.ir.expr import Literal, Var
        from emmy.compiler.ir.kernel.ir import KernelOp, Load, Tile, Write
        from emmy.compiler.ir.stmt.leaves import Assign, Let
        from emmy.compiler.tensor import Tensor
        graph=Graph(); shape=(4,); index=(Var('i'),)
        graph.add_node(InputOp(),[],Tensor('x',shape,I64),node_id='x')
        body=(Load('value','x',index,dtype=I64),
              Let('minus_three',Literal(-3,'int'),I64),
              Assign('q',ElementwiseImpl('floor_divide'),('value','minus_three'),I64),
              Assign('r',ElementwiseImpl('remainder'),('value','minus_three'),I64),
              Assign('sum',ElementwiseImpl('add'),('q','r'),I64),
              Write('out',index,'sum',value_dtype=I64))
        graph.add_node(KernelOp(body=(Tile((Axis('i',4),),body,block_threads=4),),name='k_int64_math'),
                       ['x'],Tensor('out',shape,I64),node_id='out')
        path=self.out/'integer64-math.ir.json'
        path.write_text(json.dumps(graph.to_dict(),indent=2,default=repr))
        path.with_name('meta.json').write_text(json.dumps({'kernels':[{
            'node':'out','entry':'k_int64_math','arg_order':['x','out'],
            'grid':[1,1,1],'block':[4,1,1],'smem_bytes':0,'zero_outputs':[],
            'args':[{'name':'x','dtype':'i64','shape':[4]},
                    {'name':'out','dtype':'i64','shape':[4],'output':True}]}]},indent=2))
        out=self.assemble(path)
        self.assertIn('div.s64',out)
        self.assertIn('rem.s64',out)
        self.assertIn('add.s64',out)
        self.assertIn('selp.s64',out)
        self.assertNotRegex(out, r'add\.s64 %r\d+')

    def test_bitcast_preserves_bits_across_dtype(self):
        from emmy.compiler.dtype import F32, I32
        from emmy.compiler.ir.axis import Axis
        from emmy.compiler.ir.base import InputOp
        from emmy.compiler.ir.elementwise import ElementwiseImpl
        from emmy.compiler.ir.expr import Var
        from emmy.compiler.ir.kernel.ir import KernelOp, Load, Tile, Write
        from emmy.compiler.ir.stmt.leaves import Assign
        from emmy.compiler.tensor import Tensor
        graph=Graph(); shape=(4,); index=(Var('i'),)
        graph.add_node(InputOp(),[],Tensor('x',shape,F32),node_id='x')
        body=(Load('value','x',index,dtype=F32),
              Assign('bits',ElementwiseImpl('bitcast'),('value',),I32),
              Write('out',index,'bits',value_dtype=I32))
        graph.add_node(KernelOp(body=(Tile((Axis('i',4),),body,block_threads=4),),name='k_bitcast'),
                       ['x'],Tensor('out',shape,I32),node_id='out')
        path=self.out/'bitcast.ir.json'
        path.write_text(json.dumps(graph.to_dict(),indent=2,default=repr))
        path.with_name('meta.json').write_text(json.dumps({'kernels':[{
            'node':'out','entry':'k_bitcast','arg_order':['x','out'],
            'grid':[1,1,1],'block':[4,1,1],'smem_bytes':0,'zero_outputs':[],
            'args':[{'name':'x','dtype':'f32','shape':[4]},
                    {'name':'out','dtype':'i32','shape':[4],'output':True}]}]},indent=2))
        out=self.assemble(path)
        self.assertIn('mov.b32',out)
        self.assertNotIn('cvt.s32.f32',out)

    def test_select_tail_identity_and_warp_barrier(self):
        from emmy.compiler.dtype import F32
        from emmy.compiler.ir.elementwise import ElementwiseImpl
        from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
        from emmy.compiler.ir.kernel.ir import Load, Sync, Write
        from emmy.compiler.ir.stmt.leaves import Assign, Let, Select, SelectBranch
        inside=BinaryExpr('<',Var('i'),Literal(3,'int'))
        mask=BinaryExpr('>=',Var('i'),Literal(3,'int'))
        body=(Load('value','x',(Var('i'),)),
              Let('zero',Literal(0.0,'float'),F32),
              Select('kept',(SelectBranch('value',inside),SelectBranch('zero',mask))),
              Sync(warp=True),
              Write('out',(Var('i'),),'kept'))
        out=self.assemble(typed_pointwise(body,self.out/'select-tail'))
        self.assertIn('setp.lt.s32',out)
        self.assertIn('selp.f32',out)
        self.assertIn('bar.warp.sync',out)

    def test_unsigned_negation_and_absolute_use_valid_ptx(self):
        from emmy.compiler.dtype import I32, U32
        from emmy.compiler.ir.elementwise import ElementwiseImpl
        from emmy.compiler.ir.expr import Var
        from emmy.compiler.ir.kernel.ir import Load, Write
        from emmy.compiler.ir.stmt.leaves import Assign
        path=self.out/'typed-unsigned'
        path.mkdir()
        path=typed_pointwise((Load('v','x',(Var('i'),),dtype=U32),
                              Assign('neg',ElementwiseImpl('negative'),('v',),U32),
                              Assign('abs',ElementwiseImpl('abs'),('v',),U32),
                              Assign('sum',ElementwiseImpl('add'),('neg','abs'),U32),
                              Write('out',(Var('i'),),'sum',value_dtype=U32)),path)
        meta=json.loads(path.with_name('meta.json').read_text())
        meta['kernels'][0]['args'][0]['dtype']='u32'
        meta['kernels'][0]['args'][1]['dtype']='u32'
        meta['kernels'][0]['rtol']=meta['kernels'][0]['atol']=0
        path.with_name('meta.json').write_text(json.dumps(meta))
        graph=json.loads(path.read_text())
        graph['nodes']['x']['outputs'][0]['dtype']='u32'
        graph['nodes']['out']['outputs'][0]['dtype']='u32'
        path.write_text(json.dumps(graph))
        out=self.assemble(path)
        self.assertIn('sub.u32',out)
        self.assertIn('mov.u32',out)
        self.assertNotIn('neg.u32',out)
        self.assertNotIn('abs.u32',out)


    def test_all_published_lower_and_assemble_for_both_targets(self):
        from tools.published import PUBLISHED_PROBLEMS
        for problem in PUBLISHED_PROBLEMS:
            with self.subTest(problem=problem):
                source=next((ROOT/problem).glob('*.ir.json'))
                text=mod.compile_path(source)
                ptx=self.out/(problem+'.ptx')
                ptx.write_text(text,encoding='ascii')
                meta=json.loads(source.with_name('meta.json').read_text())
                for kernel in meta['kernels']:
                    self.assertIn('.entry '+kernel['entry']+'(',text)
                for arch in ('sm_89','sm_120'):
                    result=subprocess.run(['ptxas','-arch='+arch,str(ptx),'-o',str(self.out/(problem+arch+'.cubin'))],
                                          capture_output=True,text=True,timeout=30)
                    self.assertEqual(result.returncode,0,result.stderr)

    def test_f16_storage_uses_b16_carrier_and_promoted_arithmetic(self):
        from emmy.compiler.dtype import F16
        from emmy.compiler.ir.axis import Axis
        from emmy.compiler.ir.base import InputOp
        from emmy.compiler.ir.elementwise import ElementwiseImpl
        from emmy.compiler.ir.expr import Var
        from emmy.compiler.ir.kernel.ir import KernelOp, Load, Smem, Tile, Write
        from emmy.compiler.ir.stmt.leaves import Assign
        from emmy.compiler.tensor import Tensor
        graph=Graph(); shape=(4,); index=(Var('i'),)
        graph.add_node(InputOp(),[],Tensor('x',shape,F16),node_id='x')
        body=(Smem('scratch',(4,),'half',2),Load('v','x',index,dtype=F16),
              Write('scratch',index,'v',value_dtype=F16),
              Load('w','scratch',index,dtype=F16),
              Assign('scaled',ElementwiseImpl('multiply'),('w','two'),F16),
              Assign('mask',ElementwiseImpl('greater_equal'),('w','zero'),F16),
              Assign('pick',ElementwiseImpl('where'),('mask','scaled','w'),F16),
              Write('out',index,'pick',value_dtype=F16))
        graph.add_node(KernelOp(body=(Tile((Axis('i',4),),body,block_threads=4),),name='k_f16'),
                       ['x'],Tensor('out',shape,F16),node_id='out')
        path=self.out/'f16.ir.json'
        path.write_text(json.dumps(graph.to_dict(),indent=2,default=repr))
        path.with_name('meta.json').write_text(json.dumps({'kernels':[{
            'node':'out','entry':'k_f16','arg_order':['x','out','two','zero'],
            'grid':[1,1,1],'block':[4,1,1],'smem_bytes':8,'zero_outputs':[],
            'args':[{'name':'x','dtype':'f16','shape':[4]},
                    {'name':'out','dtype':'f16','shape':[4],'output':True},
                    {'name':'two','dtype':'f16','shape':[],'scalar':True,'value':2.0},
                    {'name':'zero','dtype':'f16','shape':[],'scalar':True,'value':0.0}]}]}))
        out=self.assemble(path)
        self.assertIn('.reg .b16',out)
        self.assertIn('ld.global.b16',out)
        self.assertIn('mul.rn.f16',out)
        self.assertIn('cvt.f32.f16',out)
        self.assertIn('setp.ge.f32',out)
        self.assertIn('st.global.b16',out)
        self.assertIn('.extern .shared .align 16 .b8 __emmy_smem[]',out)
        self.assertIn('st.shared.b16',out)
        self.assertIn('ld.shared.b16',out)

    def test_scalar_math_ops_lower_without_external_calls(self):
        from emmy.compiler.ir.elementwise import ElementwiseImpl
        from emmy.compiler.ir.expr import Var
        from emmy.compiler.ir.kernel.ir import Load, Write
        from emmy.compiler.ir.stmt.leaves import Assign
        path=self.out/'scalar-math'
        path.mkdir()
        path=typed_pointwise((Load('v','x',(Var('i'),)),
                              Assign('s',ElementwiseImpl('sin'),('v',)),
                              Assign('c',ElementwiseImpl('cos'),('v',)),
                              Assign('e',ElementwiseImpl('erf'),('v',)),
                              Assign('g',ElementwiseImpl('gelu'),('v',)),
                              Assign('t',ElementwiseImpl('gelu_tanh'),('v',)),
                              Assign('p',ElementwiseImpl('pow'),('v','two')),
                              Assign('g0',ElementwiseImpl('sign'),('v',)),
                              Write('out',(Var('i'),),'g0')),path)
        meta=json.loads(path.with_name('meta.json').read_text())
        meta['kernels'][0]['arg_order'].append('two')
        meta['kernels'][0]['args'].append(
            {'name':'two','dtype':'f32','shape':[],'scalar':True,'value':2.0})
        path.with_name('meta.json').write_text(json.dumps(meta))
        out=self.assemble(path)
        for instruction in ('sin.approx.f32','cos.approx.f32','ex2.approx.f32',
                            'tanh.approx.f32','setp.gt.f32'):
            self.assertIn(instruction,out)
        self.assertNotIn('.func',out)


def generate_cases(destination):
    destination.mkdir(parents=True, exist_ok=False)
    specifications = [('p00_scalar_add', 'pointwise14', 7, 14),
                      ('r01_array_sum', 'sum257', None, 257)]
    for problem, name, tile, length in specifications:
        path = fixture(problem, destination/name, tile_extent=tile, data_extent=length, arg_reverse=True)
        meta = json.loads(path.with_name('meta.json').read_text())
        inputs = {a['name']: np.random.default_rng(a['seed']).standard_normal(a['shape']).astype(np.float32)
                  for a in meta['kernels'][0]['args'] if 'seed' in a}
        reference = inputs['x0'] + inputs['x1'] if problem.startswith('p') else np.sum(inputs['x'])
        np.savez(path.with_name('expected.npz'), **{meta['graph_outputs'][0]: reference})


if __name__=='__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--generate-fixtures':
        generate_cases(Path(sys.argv[2]))
    else:
        print('compiler_sha256='+hashlib.sha256((HERE/'compile.py').read_bytes()).hexdigest(),flush=True)
        unittest.main()
