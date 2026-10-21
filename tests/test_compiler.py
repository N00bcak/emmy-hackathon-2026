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

    def test_reject_partial_collective_tile(self):
        p=fixture('r01_array_sum',self.out/'invalid-collective',tile_extent=127)
        with self.assertRaisesRegex(mod.Unsupported,'partial final CTA'):
            mod.compile_path(p)

    def test_reject_nested_collective_partial_cta(self):
        from emmy.compiler.ir.kernel.ir import Cond
        from emmy.compiler.ir.expr import BinaryExpr, Literal
        path=fixture('r01_array_sum',self.out/'invalid-nested-collective',tile_extent=127)
        graph=Graph.from_dict(json.loads(path.read_text()))
        node=graph.nodes['sum_1']
        tile=node.op.body[0]
        # Put the whole reduction inside a conditional, hiding all collectives
        # from a shallow Tile-body inspection. This fixture must fail pre-emission.
        cond=Cond(BinaryExpr('==',Literal(0,'int'),Literal(0,'int')),tile.body,())
        node.op=replace(node.op,body=(replace(tile,body=(cond,)),))
        path.write_text(json.dumps(graph.to_dict(),indent=2,default=repr))
        with self.assertRaisesRegex(mod.Unsupported,'partial final CTA'):
            mod.compile_path(path)

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

    def test_unsupported_errors_do_not_emit_partial_ptx(self):
        p=fixture('p00_scalar_add',self.out/'unsupported-scalar-abi')
        meta=json.loads(p.with_name('meta.json').read_text())
        meta['kernels'][0]['args'][0]['scalar']=True
        meta['kernels'][0]['args'][0]['value']=1.0
        p.with_name('meta.json').write_text(json.dumps(meta))
        r=subprocess.run([sys.executable,str(HERE/'compile.py'),str(p)],capture_output=True,text=True)
        self.assertNotEqual(r.returncode,0)
        self.assertEqual(r.stdout,'')
        self.assertIn('Unsupported',r.stderr)


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
