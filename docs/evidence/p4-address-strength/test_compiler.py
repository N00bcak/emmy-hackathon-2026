"""CPU-only baseline tests and legal fixtures constructed from typed IR nodes."""
import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
import re
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


def typed_pointwise(body, destination, *, extent=4, out_shape=(4,), block=4, input_shape=None):
    """Write a small typed-IR graph whose Tile body is supplied by the test."""
    from emmy.compiler.ir.axis import Axis
    from emmy.compiler.ir.base import InputOp
    from emmy.compiler.ir.kernel.ir import KernelOp, Tile
    from emmy.compiler.tensor import Tensor
    from emmy.compiler.dtype import F32
    graph=Graph()
    shape=tuple(input_shape or (extent,))
    graph.add_node(InputOp(),[],Tensor('x',shape,F32),node_id='x')
    graph.add_node(KernelOp(body=(Tile((Axis('i',extent),),tuple(body),block_threads=block),),name='k'),
                   ['x'],Tensor('out',tuple(out_shape),F32),node_id='out')
    destination.mkdir(parents=True, exist_ok=True)
    path=destination/'typed.ir.json'
    path.write_text(json.dumps(graph.to_dict(),indent=2,default=repr))
    path.with_name('meta.json').write_text(json.dumps({'graph_inputs':['x'],'graph_outputs':['out'],
        'rtol':0,'atol':0,'kernels':[{'node':'out','entry':'k','arg_order':['x','out'],
        'grid':[1,1,1],'block':[block,1,1],'smem_bytes':0,'zero_outputs':[],
        'args':[{'name':'x','dtype':'f32','shape':list(shape)},
                {'name':'out','dtype':'f32','shape':list(out_shape),'output':True}]}]}))
    return path


def emitter_fixture():
    """A complete emitter state for unit tests of one lowering helper.

    Keeping this in one place avoids the stale partial ``__new__`` fixtures
    that silently miss fields added to ``Emitter.emit`` or its helpers.
    """
    emitter = mod.Emitter.__new__(mod.Emitter)
    emitter.counts = {'r': 0, 'rh': 0, 'rd': 0, 'f': 0, 'fd': 0, 'p': 0}
    emitter.lines = []
    emitter.unary_cache = {}
    emitter.label_count = 0
    emitter.dynamic_shared = False
    emitter.cp_async_base = '__emmy_smem'
    return emitter


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
        for arch in ('sm_89', 'sm_120'):
            # Target-only adaptation checks lowering legality on both devices.
            # Assembly is not a GPU correctness or performance receipt.
            target_ptx=ptx.with_suffix('.'+arch+'.ptx')
            target_ptx.write_text(re.sub(r'^\.target\s+sm_\d+', '.target '+arch,
                                       re.sub(r'^\.version\s+[\d.]+', '.version 8.7', result.stdout, flags=re.M), flags=re.M))
            assembled=subprocess.run(['ptxas','-arch='+arch,str(target_ptx),'-o',str(target_ptx.with_suffix('.cubin'))],capture_output=True,text=True,timeout=30)
            self.assertEqual(assembled.returncode,0,arch+': '+assembled.stderr)
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

    def test_single_variable_affine_address_scales_directly(self):
        from emmy.compiler.dtype import F32
        from emmy.compiler.ir.axis import Axis
        from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
        from emmy.compiler.ir.kernel.ir import KernelOp, Load, Tile, Write
        from emmy.compiler.ir.base import InputOp
        from emmy.compiler.tensor import Tensor

        graph=Graph(); shape=(16,)
        graph.add_node(InputOp(),[],Tensor('x',shape,F32),node_id='x')
        # The index is affine but not written as a single multiplication, so
        # the direct scale must be derived from the typed expression tree.
        index=(BinaryExpr('+',BinaryExpr('*',Var('i'),Literal(4,'int')),Literal(0,'int')),)
        body=(Load('v','x',index,dtype=F32),Write('y',index,'v',value_dtype=F32))
        graph.add_node(KernelOp(body=(Tile((Axis('i',4),),body,block_threads=4),),name='k'),
                       ['x'],Tensor('y',shape,F32),node_id='y')
        path=self.out/'direct-affine.ir.json'
        path.write_text(json.dumps(graph.to_dict(),indent=2,default=repr))
        path.with_name('meta.json').write_text(json.dumps({'kernels':[{'node':'y','entry':'k',
            'arg_order':['x','y'],'grid':[1,1,1],'block':[4,1,1],'smem_bytes':0,'zero_outputs':[],
            'args':[{'name':'x','dtype':'f32','shape':[16]},
                    {'name':'y','dtype':'f32','shape':[16],'output':True}]}]}))
        out=self.assemble(path)
        self.assertRegex(out,r'mul\.wide\.s32 %rd\d+, %r\d+, 16;')

    def test_affine_address_falls_back_on_unproven_structure(self):
        from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
        cases={
            'offset':((BinaryExpr('+',BinaryExpr('*',Var('i'),Literal(4,'int')),Literal(2,'int')),),0),
            'sum':((BinaryExpr('+',BinaryExpr('*',Var('i'),Literal(4,'int')),Var('j')),),0),
            'negative-scale':((BinaryExpr('*',Var('i'),Literal(-4,'int')),),0),
        }
        for label,(index,_) in cases.items():
            with self.subTest(label=label):
                self.assertIsNone(mod.affine_byte_scale((16,),'f32',index))
        self.assertEqual(('i',4,16),mod.affine_byte_scale((16,),'f32',
                         (BinaryExpr('*',Var('i'),Literal(4,'int')),)))

        emitter=emitter_fixture(); emitter.buffers={'x':((16,),'f32')}
        emitter.shared=set(); emitter.ptrs={'x':'%base'}
        emitter.values={'i':('%i','i32')}; emitter.versions={'i':0}
        emitter.address_cache={}; emitter.expr_cache={}; emitter.int_ranges={}
        emitter.var_residue={}; emitter.axis_facts={}
        emitter.loop_invariants=[]; emitter.loop_expressions=[]
        emitter.address('x',(BinaryExpr('*',Var('i'),Literal(4,'int')),),1)
        # An element offset is part of the flattened i32 offset and must use
        # the generic add-before-widen chain.
        self.assertIn('add.s32',emitter.lines[-3])
        self.assertIn('mul.wide.s32',emitter.lines[-2])

    def test_affine_address_requires_current_i32_binding(self):
        from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
        emitter=emitter_fixture(); emitter.buffers={'x':((16,),'f32')}
        emitter.shared=set(); emitter.ptrs={'x':'%base'}; emitter.values={'i':('%i','f64')}
        emitter.address_cache={}; emitter.expr_cache={}
        emitter.versions={'i':0}; emitter.int_ranges={}; emitter.var_residue={}
        emitter.axis_facts={}; emitter.loop_invariants=[]; emitter.loop_expressions=[]
        index=(BinaryExpr('*',Var('i'),Literal(4,'int')),)
        # The direct path declines a non-i32 binding; the generic lowering
        # then fails closed instead of coercing the index.
        with self.assertRaises(mod.Unsupported):
            emitter.address('x',index)

        emitter.values={'i':('%i','i32')}
        address,_=emitter.address('x',index)
        # No range proof means the existing flatten-then-scale chain remains.
        self.assertIn('mad.lo.s32',emitter.lines[-3])
        self.assertIn('mul.wide.s32',emitter.lines[-2])
        self.assertTrue(address.startswith('%rd'))

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
        self.assertRegex(out,r'max(?:\.NaN)?\.f32')
        # Inactive lanes must not fold their masked load into the maximum.
        self.assertRegex(out, r'@%p\d+ max(?:\.NaN)?\.f32')
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
        # Evaluate emitted address arithmetic independently of register names.
        # The shared-state destination must be base+offset, including when the
        # assembler assigns a nonzero dynamic slab position (SM120 regression).
        import re
        for dynamic in (False, True):
            for base, offset in ((0, 48), (1024, 48), (6144, 256)):
                with self.subTest(dynamic=dynamic, base=base, offset=offset):
                    emitter=emitter_fixture()
                    emitter.dynamic_shared=dynamic
                    emitter.cp_async_base='dynamic_slab'
                    emitter._dynamic_shared_base='%old_base'
                    slab='dynamic_slab' if dynamic else 'static_slab'
                    registers={slab:base, '%destination':base+offset, '%old_base':base}
                    emitter.emit_cp_async('%source','%destination',16,base_name='static_slab')
                    for instruction in emitter.lines[:-1]:
                        op, operands=instruction.strip().rstrip(';').split(None,1)
                        dst,*args=[x.strip() for x in operands.split(',')]
                        values=[registers[x] for x in args]
                        if op.startswith(('mov.','cvt.')): value=values[0]
                        elif op.startswith('sub.'): value=values[0]-values[1]
                        elif op.startswith('add.'): value=sum(values)
                        else: self.fail('Unhandled address instruction: '+op)
                        bits=int(re.search(r'(32|64)$',op).group(1))
                        registers[dst]=value % (1 << bits)
                    destination=re.search(r'\[([^]]+)\]',emitter.lines[-1]).group(1)
                    self.assertEqual(base+offset,registers[destination])

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

    def test_pointwise_tuple_accesses_vectorize_with_runtime_fallback(self):
        for problem in ('p02_scale_shift', 'p03_relu_mul'):
            with self.subTest(problem=problem):
                source=next((ROOT/problem).glob('*.ir.json'))
                out=mod.compile_path(source)
                # Every affine four-element tuple is one 128-bit transaction,
                # but the pointer is only known aligned at runtime. Each vector
                # access keeps a scalar branch for a misaligned allocation.
                self.assertGreaterEqual(out.count('ld.global.v4.f32'), 1)
                self.assertRegex(out, r'st\.global\.v4\.f32')
                self.assertGreaterEqual(out.count('and.b64'), 2)
                self.assertIn('setp.eq.u64',out)
                self.assertRegex(out, r'@!%p\d+ bra access_scalar_\d+;')
                self.assertRegex(out, r'ld\.global\.f32 %f\d+, \[%rd\d+\+4\];')

    def test_noncontiguous_and_unaligned_groups_stay_scalar(self):
        from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
        from emmy.compiler.ir.kernel.ir import Load, Write
        def index(offset, step):
            base=BinaryExpr('*',Var('i'),Literal(4,'int'))
            start=base if not offset else BinaryExpr('+',base,Literal(offset,'int'))
            return start if not step else BinaryExpr('+',start,Literal(step,'int'))
        def copy(step, offset):
            names=tuple(f'v{k}' for k in range(4))
            return (*(Load(name,'x',(index(offset,k*step),)) for k,name in enumerate(names)),
                    Write('out',(index(offset,0),),values=names))
        for label, step, offset in (('noncontiguous',2,0),('offset-one',1,1)):
            with self.subTest(label=label):
                path=typed_pointwise(copy(step,offset),self.out/label,extent=4,block=1,
                                     input_shape=(64,))
                meta=json.loads(path.with_name('meta.json').read_text())
                meta['kernels'][0]['grid']=[4,1,1]
                path.with_name('meta.json').write_text(json.dumps(meta))
                out=self.assemble(path)
                # Offset one has a contiguous four-element run, but it starts
                # unaligned. A strided source has no contiguous load run; its
                # destination still does, so only the store may combine.
                self.assertNotIn('ld.global.v4.f32',out)
                self.assertEqual(4,out.count('ld.global.f32'))
                if step > 1:
                    self.assertIn('st.global.v4.f32',out)
                    self.assertEqual(4,out.count('st.global.f32'))
                else:
                    self.assertNotIn('st.global.v4.f32',out)
                    self.assertEqual(4,out.count('st.global.f32'))

    def test_destination_shadowing_prevents_vector_load_group(self):
        import importlib.util
        from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
        from emmy.compiler.ir.kernel.ir import Load
        from emmy.compiler.ir.stmt.leaves import Let
        spec=importlib.util.spec_from_file_location('access_compiler',HERE/'compile.py')
        access=importlib.util.module_from_spec(spec);spec.loader.exec_module(access)
        def index(k):
            base=BinaryExpr('*',Var('i'),Literal(4,'int'))
            return base if not k else BinaryExpr('+',base,Literal(k,'int'))
        clean=(Load('a','x',(index(0),)),Load('b','x',(index(1),)),
               Load('c','x',(index(2),)),Load('d','x',(index(3),)))
        shadowed=(Load('a','x',(index(0),)),Load('b','x',(index(1),)),
                  Let('c',Literal(9,'int')),Load('c','x',(index(2),)),
                  Load('d','x',(index(3),)))
        buffers={'x':((64,),'f32')}
        proven=access.access_descriptors(clean,buffers,{})
        rejected=access.access_descriptors(shadowed,buffers,{})
        self.assertEqual(1,len(proven.loads))
        self.assertEqual((1,1,1,1),proven.loads[next(iter(proven.loads))].reads)
        self.assertEqual({},rejected.loads)

    def test_shared_aligned_group_has_no_runtime_fallback(self):
        from emmy.compiler.ir.kernel.ir import Load, Smem, Sync, Write
        from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
        def index(k):
            base=BinaryExpr('*',Var('i'),Literal(4,'int'))
            return base if not k else BinaryExpr('+',base,Literal(k,'int'))
        names=tuple(f'v{k}' for k in range(4))
        body=(Smem('scratch',(16,),'float',16),
              *(Load(name,'x',(index(k),)) for k,name in enumerate(names)),
              Write('scratch',(index(0),),values=names),Sync(0,None,False),
              *(Load(f's{k}','scratch',(index(k),)) for k in range(4)),
              Write('out',(index(0),),values=tuple(f's{k}' for k in range(4))))
        path=typed_pointwise(body,self.out/'shared-vector',extent=16,block=1,
                             input_shape=(16,),out_shape=(16,))
        meta=json.loads(path.with_name('meta.json').read_text())
        meta['kernels'][0]['grid']=[16,1,1]
        meta['kernels'][0]['arg_order'].append('scratch')
        meta['kernels'][0]['args'].append(
            {'name':'scratch','dtype':'f32','shape':[16],'output':True})
        path.with_name('meta.json').write_text(json.dumps(meta))
        out=self.assemble(path)
        self.assertIn('st.shared.v4.f32',out)
        self.assertIn('ld.global.v4.f32',out)
        # Shared offsets are statically aligned, so those groups do not pay for
        # a per-access alignment predicate; the global source still does.
        self.assertNotRegex(out, r'@!%p\d+ bra [^\n]*\n\s*ld\.shared\.v4')

    def test_mixed_typed_grouped_store_stays_scalar(self):
        import importlib.util
        from emmy.compiler.dtype import F32, BOOL
        from emmy.compiler.ir.expr import Literal
        from emmy.compiler.ir.kernel.ir import Write
        from emmy.compiler.ir.stmt.leaves import Let
        spec=importlib.util.spec_from_file_location('typed_store_compiler',HERE/'compile.py')
        typed=importlib.util.module_from_spec(spec);spec.loader.exec_module(typed)
        body=(Let('a',Literal(0,'int'),F32),Let('b',Literal(True,'bool'),BOOL),
              Let('c',Literal(2,'int'),F32),Let('d',Literal(False,'bool'),BOOL),
              Write('out',(Literal(0,'int'),),values=('a','b','c','d')))
        plan=typed.access_descriptors(body,{'out':((4,),'f32')},{})
        self.assertEqual({},plan.stores)

    def test_integer_grouped_store_converts_like_scalar(self):
        import importlib.util
        from emmy.compiler.ir.expr import Literal
        from emmy.compiler.ir.kernel.ir import Write
        from emmy.compiler.ir.stmt.leaves import Let
        from emmy.compiler.dtype import I32
        spec=importlib.util.spec_from_file_location('integer_store_compiler',HERE/'compile.py')
        typed=importlib.util.module_from_spec(spec);spec.loader.exec_module(typed)
        body=(Let('a',Literal(0,'int'),I32),Let('b',Literal(1,'int'),I32),
              Let('c',Literal(2,'int'),I32),Let('d',Literal(3,'int'),I32),
              Write('out',(Literal(0,'int'),),values=('a','b','c','d')))
        plan=typed.access_descriptors(body,{'out':((4,),'f32')},{})
        self.assertEqual(1,len(plan.stores))

    def test_partial_cta_vector_load_has_three_way_mask(self):
        from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
        from emmy.compiler.ir.kernel.ir import Load, Sync, Write
        def index(k):
            base=BinaryExpr('*',Var('i'),Literal(4,'int'))
            return base if not k else BinaryExpr('+',base,Literal(k,'int'))
        names=tuple(f'v{k}' for k in range(4))
        body=(*(Load(name,'x',(index(k),)) for k,name in enumerate(names)),
              Write('out',(index(0),),values=names),Sync(0,None,False))
        path=typed_pointwise(body,self.out/'partial-vector',extent=7,block=4,
                             input_shape=(28,),out_shape=(28,))
        meta=json.loads(path.with_name('meta.json').read_text())
        meta['kernels'][0]['grid']=[2,1,1]
        path.with_name('meta.json').write_text(json.dumps(meta))
        out=self.assemble(path)
        self.assertRegex(out, r'@!%p\d+ bra access_masked_zero_\d+;')
        self.assertRegex(out, r'@!%p\d+ bra access_masked_scalar_\d+;')
        self.assertIn('ld.global.v4.f32',out)
        self.assertRegex(out, r'ld\.global\.f32 %f\d+, \[%rd\d+\+12\];')
        self.assertRegex(out, r'@%p\d+ st\.global\.v4\.f32')

    def test_m01_inner_loop_hoists_invariant_index_division(self):
        """m01's selected staged pipeline has its documented invariants."""
        import re
        source=next((ROOT/'m01_matmul_512').glob('*.ir.json'))
        lines=mod.compile_path(source).splitlines()
        rendered='\n'.join(lines)
        # plan_simt_ptx addresses the dynamic slab with a 32-bit shared-window
        # base; the generic path's 64-bit form is not part of this route.
        self.assertRegex(rendered, r'mov\.u32 %r\d+, __emmy_smem;')
        self.assertIn('cp.async.cg.shared.global', rendered)
        self.assertIn('cp.async.wait_group', rendered)
        self.assertIn('bar.sync', rendered)
        self.assertIn('fma.rn.f32', rendered)
        self.assertNotRegex(rendered, r'\bbra\s+(?:loop|loop_done)_\d+;')
        self.assertRegex(rendered, r'ld\.shared\.v4\.f32')
        self.assertRegex(rendered, r'st\.global\.v4\.f32')

    def test_generic_loop_hoists_invariant_index_division(self):
        """The generic loop analysis hoists a repeated invariant quotient."""
        from emmy.compiler.ir.axis import Axis
        from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
        from emmy.compiler.ir.kernel.ir import Load, Loop, Write
        from emmy.compiler.ir.elementwise import ElementwiseImpl
        from emmy.compiler.ir.stmt.leaves import Assign
        quotient=BinaryExpr('//',BinaryExpr('*',Var('i'),Literal(4,'int')),
                            Literal(2,'int'))
        loop=Loop(Axis('k',4),(Load('a','x',(quotient,)),
                               Load('b','x',(quotient,)),
                               Assign('v',ElementwiseImpl('add'),('a','b')),
                               Write('out',(Var('i'),),'v')),seed=False)
        out=self.assemble(typed_pointwise((loop,),self.out/'generic-hoist',extent=4))
        match=re.search(r'\nloop_\d+:',out)
        self.assertIsNotNone(match)
        preheader, body=out[:match.start()],out[match.start():]
        # The quotient is invariant in k, appears in both loads and is used
        # twice, so its strength-reduced form is computed once before the
        # backedge and neither product nor shift is repeated inside the loop.
        self.assertEqual(1,preheader.count('shr.s32'))
        self.assertNotIn('shr.s32',body)
        self.assertNotIn('div.',body)
        self.assertNotIn('rem.',body)
        self.assertEqual(2,body.count('ld.global.f32'))

    def test_prefetch_scalar_loads_replaces_the_map_read(
            self):
        """f03 issues its map input beside the reduction and reuses it."""
        source=next((ROOT/'f03_l2norm').glob('*.ir.json'))
        out=mod.compile_path(source)
        lines=out.splitlines()
        vector_start=next(i for i,line in enumerate(lines)
                          if 'ld.global.v4.f32' in line)
        map_start=next(i for i,line in enumerate(lines)
                       if 'row_map_fallback_' in line)
        map_end=next(i for i,line in enumerate(lines[map_start+1:],map_start+1)
                     if 'row_map_done_' in line)
        vector_region='\n'.join(lines[vector_start:map_start])
        map_region='\n'.join(lines[map_start:map_end])
        # The reduction vector loads x; the map consumes those same registers
        # rather than issuing a second vector read for the same row.
        loaded=set(re.findall(r'ld\.global\.v4\.f32 \{([^}]*)\}',vector_region)[0].split(', '))
        used=set(re.findall(r'mul\.f32 %f\d+, (%f\d+),',map_region))
        self.assertTrue(loaded & used)
        self.assertNotIn('ld.global.v4.f32',map_region)

    def test_row_cache_and_cta_total_mechanisms_are_populated(self):
        """f01/f03/f04's merged reductions leave a cache and CTA total."""
        for problem in ('f01_rmsnorm','f03_l2norm','r04_row_sum'):
            with self.subTest(problem=problem):
                source=next((ROOT/problem).glob('*.ir.json'))
                graph=Graph.from_dict(json.loads(source.read_text()))
                meta=json.loads(source.with_name('meta.json').read_text())['kernels'][0]
                node=graph.nodes[meta['node']]
                emitter=mod.Emitter(graph,node,meta)
                emitter.render()
                self.assertEqual(1,len(emitter.row_caches))
                self.assertEqual(1,len(emitter.cta_total_trees))
                _source,_va,_trips,loads,registers=next(iter(emitter.row_caches.values()))
                self.assertTrue(loads)
                self.assertTrue(all(registers.values()))

    def test_row_maps_use_the_shared_strided_access_proof(self):
        """f01/f03 keep row reuse, and a rejected proof fails closed."""
        from emmy.compiler.dtype import F32
        from emmy.compiler.ir.axis import Axis
        from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
        from emmy.compiler.ir.kernel.ir import Load, StridedLoop, Write
        map_axis = Axis('lane', 512)
        row = (Var('row'), Var('lane'))
        map_loop = StridedLoop(
            map_axis, Var('lane_co'), Literal(128, 'int'),
            (Load('value', 'x', row, dtype=F32),
             Write('out', row, 'value', value_dtype=F32)))
        source = StridedLoop(
            Axis('lane', 512), Var('lane_co'), Literal(128, 'int'),
            (Load('source_value', 'x', (Var('row'), Var('lane')), dtype=F32),))
        buffers = {'x': ((64, 512), 'f32'), 'out': ((64, 512), 'f32')}
        uniform = {'row', 'lane_co'}.__contains__
        proven = mod.strided_access_groups(
            map_loop, source, tuple(source.body), buffers, {'x', 'out'}, set(),
            {'row': 64, 'lane_co': 128}, uniform)
        self.assertEqual(1, len(proven.loads))
        self.assertEqual(1, len(proven.stores))
        self.assertIs(proven.loads[map_loop.body[0]].cached, source.body[0])
        # Every field that contributes to vector legality is checked here.
        rejected = {
            'axis-coefficient': ((Var('row'), Literal(2, 'int')), buffers, uniform),
            'not-last-coordinate': ((Var('lane'), Var('row')), buffers, uniform),
            'nonuniform-row': ((Var('tid'), Var('lane')), buffers, uniform),
            'mixed-dtype': (row, {**buffers, 'out': ((64, 512), 'f64')}, uniform),
            'unaligned-row-term': (
                row, {'x': ((64, 513), 'f32'), 'out': ((64, 513), 'f32')}, uniform),
        }
        for label, (index, table, is_uniform) in rejected.items():
            with self.subTest(label=label):
                body = (Load('value', 'x', index, dtype=F32),
                        Write('out', index, 'value', value_dtype=F32))
                loop = replace(map_loop, body=body)
                self.assertIsNone(mod.strided_access_groups(
                    loop, source, tuple(source.body), table, {'x', 'out'}, set(),
                    {'row': 64, 'lane_co': 128}, is_uniform))
        # A legal map whose source row is not the cached load still vectorizes,
        # but without claiming the reduction's registers.
        moved = replace(source.body[0], index=(Var('row'), BinaryExpr('+', Var('lane'), Literal(4, 'int'))))
        uncached = mod.strided_access_groups(
            map_loop, replace(source, body=(moved,)), (moved,), buffers,
            {'x', 'out'}, set(), {'row': 64, 'lane_co': 128}, uniform)
        self.assertIsNone(uncached.loads[map_loop.body[0]].cached)

        for problem in ('f01_rmsnorm', 'f03_l2norm'):
            with self.subTest(problem=problem):
                source = next((ROOT/problem).glob('*.ir.json'))
                out = mod.compile_path(source)
                self.assertIn('row_map_fallback_', out)
                self.assertRegex(out, r'st\.global\.v4\.f32')
                original = mod.strided_access_groups
                mod.strided_access_groups = lambda *args, **kwargs: None
                try:
                    rejected_out = mod.compile_path(source)
                finally:
                    mod.strided_access_groups = original
                self.assertNotIn('row_map_fallback_', rejected_out)

    def test_collective_tree_reuses_its_root_slot(self):
        """A CTA total is read from shared slot zero once, not re-read."""
        source=next((ROOT/'r04_row_sum').glob('*.ir.json'))
        lines=mod.compile_path(source).splitlines()
        tree_skips=[i for i,line in enumerate(lines) if re.fullmatch(r'tree_skip_\d+:',line)]
        self.assertTrue(tree_skips)
        # The final barrier publishes the tree root; the consumer then loads
        # slot zero exactly once and reuses that register in the epilogue.
        root_start=tree_skips[-1]
        root_lines=lines[root_start:]
        shared_loads=[line.strip() for line in root_lines if 'ld.shared.f32' in line]
        self.assertEqual(1,len(shared_loads))
        self.assertIn('[%rd',shared_loads[0])
        load_at=next(i for i,line in enumerate(root_lines)
                     if 'ld.shared.f32' in line)
        epilogue='\n'.join(root_lines[load_at+1:])
        self.assertNotIn('ld.shared.f32',epilogue)
        self.assertIn('st.global.f32',epilogue)

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
        # Let captures its value in a fresh register. Follow those captures
        # when checking that the second address uses the rebound index.
        indices={add.group(1)}
        for line in out.splitlines():
            move=re.search(r'mov\.b32\s+(%r\d+)\s*,\s*(%r\d+);',line)
            if move and move.group(2) in indices:
                indices.add(move.group(1))
        self.assertTrue(any(re.search(rf'mad\.lo\.s32\s+%r\d+\s*,\s*{re.escape(index)}\s*,\s*1\s*,\s*0;',out)
                            for index in indices))

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
                              Assign('g0',ElementwiseImpl('sign'),('v',)),
                              Write('out',(Var('i'),),'g0')),path)
        meta=json.loads(path.with_name('meta.json').read_text())
        meta['kernels'][0]['arg_order'].append('two')
        meta['kernels'][0]['args'].append(
            {'name':'two','dtype':'f32','shape':[],'scalar':True,'value':2.0})
        path.with_name('meta.json').write_text(json.dumps(meta))
        out=self.assemble(path)
        self.assertRegex(out, r'ex2\.approx(?:\.ftz)?\.f32')
        for instruction in ('sin.approx.f32','cos.approx.f32',
                            'tanh.approx.f32','setp.gt.f32'):
            self.assertIn(instruction,out)
        self.assertNotIn('.func',out)


    def test_pow_support_or_clean_rejection(self):
        from emmy.compiler.ir.elementwise import ElementwiseImpl
        from emmy.compiler.ir.expr import Var
        from emmy.compiler.ir.kernel.ir import Load, Write
        from emmy.compiler.ir.stmt.leaves import Assign
        path=typed_pointwise((Load('v','x',(Var('i'),)),
                              Assign('p',ElementwiseImpl('pow'),('v','v')),
                              Write('out',(Var('i'),),'p')),self.out/'unsupported-pow')
        try:
            mod.compile_path(path)
        except mod.Unsupported:
            pass
        else:
            self.assemble(path)
            return
        result=subprocess.run([sys.executable,str(HERE/'compile.py'),str(path)],capture_output=True,text=True,timeout=15)
        self.assertNotEqual(result.returncode,0)
        self.assertEqual(result.stdout,'')
        self.assertIn('Unsupported',result.stderr)


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
