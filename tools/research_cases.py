"""Versioned, typed research problems and independent mathematical references.

This module constructs IR; it never renders CUDA or lowers a candidate compiler.
Published cooperative/async schedules are retained in the regression suite;
fresh serial schedules exercise independently varying dimensions and operations.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np
from emmy.compiler.graph import Graph
from emmy.compiler.tensor import Tensor
from emmy.compiler.dtype import F32
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.kernel.ir import KernelOp, Tile, Load, Write, Assign, Loop, StridedLoop, Accum, Cond
from emmy.compiler.ir.stmt.leaves import Let
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Var, Literal, BinaryExpr
from emmy.compiler.ir.elementwise import ElementwiseImpl

VERSION = 'emmy-research-cases-v1'
TOLERANCES = {
    'pointwise': (1e-6, 1e-6), 'reduction': (1e-4, 1e-4),
    'rmsnorm': (1e-3, 1e-4), 'l2norm': (1e-3, 1e-4),
    'matmul': (1e-3, 1e-3), 'atomic_matmul': (1e-3, 1e-3),
    'softmax': (1e-3, 1e-5), 'attention': (1e-3, 1e-4),
}
DOMAINS = {'normal', 'zeros', 'impulse', 'cancellation', 'bounded'}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), default=repr).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def save_arrays(path, arrays):
    """Deterministic NPZ: no ZIP timestamps, object arrays, or pickle."""
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, array in sorted(arrays.items()):
            data = io.BytesIO()
            np.lib.format.write_array(data, np.ascontiguousarray(array), allow_pickle=False)
            item = zipfile.ZipInfo(name + '.npy', date_time=(1980, 1, 1, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(item, data.getvalue())


def _inputs(rng, shape, pattern):
    if pattern not in DOMAINS:
        raise ValueError('unknown input domain')
    x = np.clip(rng.normal(0, .5, shape), -4, 4).astype('float32')
    if pattern == 'zeros': x.fill(0)
    elif pattern == 'bounded': x = rng.uniform(-2, 2, shape).astype('float32')
    elif pattern == 'impulse':
        x.fill(0); x.reshape(-1)[int(rng.integers(x.size))] = 1
    elif pattern == 'cancellation':
        flat = x.reshape(-1)
        flat[::2] = .5
        flat[1::2] = -.5
        flat[-1] += np.float32(.001)
    return x


def reference(spec, inputs):
    """Plain mathematical oracles. No online recurrence or lowering helpers."""
    names = spec['names']
    get = lambda name: inputs[names[name]].astype(np.float64)
    family = spec['family']
    if family == 'pointwise':
        x, y = inputs[names['x']], inputs[names['y']]
        op = spec['operation']
        if op == 'add': value = np.add(x, y)
        elif op == 'subtract': value = np.subtract(x, y)
        elif op == 'multiply': value = np.multiply(x, y)
        elif op == 'relu_mul': value = np.maximum(x, np.float32(0)) * y
        elif op == 'scale_shift': value = x.astype('float64') * get('scale')[0] + get('shift')[0]
        else: raise ValueError('unknown pointwise operation')
    elif family == 'reduction': value = np.array([get('x').sum(dtype=np.float64)])
    elif family in ('matmul', 'atomic_matmul'): value = get('x') @ get('y')
    elif family == 'softmax':
        x = get('x'); e = np.exp(x - np.max(x, axis=1, keepdims=True))
        value = e / np.sum(e, axis=1, keepdims=True)
    elif family == 'attention':
        scores = (get('x') @ get('y')) / float(get('scale')[0])
        e = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        value = (e / np.sum(e, axis=1, keepdims=True)) @ get('v')
    elif family == 'rmsnorm':
        x = get('x'); value = x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + spec['epsilon'])
    elif family == 'l2norm':
        x = get('x'); value = x / np.sqrt(np.sum(x*x, axis=-1, keepdims=True) + spec['epsilon'])
    else: raise ValueError('unknown family')
    return {names['out']: np.asarray(value, dtype=np.float32)}


def _v(name): return Var(name)
def _l(value): return Literal(value, 'int' if isinstance(value, int) else 'float')
def _b(op, a, b): return BinaryExpr(op, a, b)
def _assign(name, op, *args): return Assign(name, ElementwiseImpl(op), tuple(args), F32)
def _fold(name, value, op='add', axes=()):
    return Accum(name, value, ElementwiseImpl(op), F32, axes=axes)


class Builder:
    def __init__(self, spec, rng):
        self.graph, self.kernels, self.shapes = Graph(), [], {}
        self.spec, self.rng = spec, rng
        self.block = spec['block']

    def input(self, name, shape):
        self.graph.add_node(InputOp(), [], Tensor(name, tuple(shape)), node_id=name)
        self.graph.inputs.append(name)
        self.shapes[name] = list(shape)

    def kernel(self, out, shape, inputs, tile_axes, body, *, atomic=False):
        entry = 'k_' + out
        self.graph.add_node(KernelOp(body=(Tile(tuple(tile_axes), tuple(body), block_threads=self.block),), name=entry),
                            inputs, Tensor(out, tuple(shape)), node_id=out)
        self.shapes[out] = list(shape)
        order = list(inputs) + [out]
        self.rng.shuffle(order)
        args = [{'name': n, 'dtype': 'f32', 'shape': self.shapes[n], **({'output': True} if n == out else {})} for n in order]
        self.kernels.append({'node': out, 'entry': entry, 'arg_order': order, 'args': args,
                             'grid': [math.ceil(math.prod(a.extent.as_static() for a in tile_axes)/self.block), 1, 1],
                             'block': [self.block, 1, 1], 'smem_bytes': 0,
                             'zero_outputs': [out] if atomic else [], 'final': False})

    def matmul(self, x, y, out, m, n, k, *, atomic=False, scale=None):
        # A split-reduction dimension produces atomic partial sums when requested.
        splits = 2 if atomic else 1
        axes = [Axis('row', m), Axis('col', n)] + ([Axis('part', splits)] if atomic else [])
        coord = _b('+', _b('*', _v('kk'), _l(splits)), _v('part')) if atomic else _v('kk')
        valid = _b('<', coord, _l(k))
        loads = (Load('left', x, (_v('row'), coord), dtype=F32), Load('right', y, (coord, _v('col')), dtype=F32),
                 _assign('product', 'multiply', 'left', 'right'), _fold('total', 'product', axes=('kk',)))
        loop = Loop(Axis('kk', math.ceil(k/splits)), (Cond(valid, loads),) if atomic else loads)
        body = [loop]
        value = 'total'
        inputs = [x, y]
        if scale:
            inputs.append(scale)
            body += [Load('divisor', scale, (_l(0),), dtype=F32), _assign('scaled', 'divide', value, 'divisor')]
            value = 'scaled'
        body.append(Write(out, (_v('row'), _v('col')), value, value_dtype=F32, atomic=atomic))
        self.kernel(out, (m, n), inputs, axes, body, atomic=atomic)

    def softmax(self, x, out, rows, cols):
        # Three ordinary serial passes; oracle independently uses vectorized NumPy.
        maxloop = Loop(Axis('mxcol', cols), (Load('mxval', x, (_v('row'), _v('mxcol')), dtype=F32),
                       _fold('rowmax', 'mxval', 'maximum', ('mxcol',))))
        sumloop = Loop(Axis('sumcol', cols), (Load('sval', x, (_v('row'), _v('sumcol')), dtype=F32),
                       _assign('delta', 'subtract', 'sval', 'rowmax'), _assign('ev', 'exp', 'delta'),
                       _fold('denom', 'ev', axes=('sumcol',))))
        writeloop = Loop(Axis('outcol', cols), (Load('oval', x, (_v('row'), _v('outcol')), dtype=F32),
                         _assign('od', 'subtract', 'oval', 'rowmax'), _assign('oe', 'exp', 'od'),
                         _assign('prob', 'divide', 'oe', 'denom'),
                         Write(out, (_v('row'), _v('outcol')), 'prob', value_dtype=F32)))
        self.kernel(out, (rows, cols), [x], [Axis('row', rows)], [maxloop, sumloop, writeloop])


def _build(spec, rng):
    b = Builder(spec, rng); names, shape = spec['names'], spec['shape']
    x, y, out = (names[z] for z in ('x', 'y', 'out'))
    family = spec['family']
    if family == 'pointwise':
        n, = shape; width = spec['vector_width']; op = spec['operation']
        b.input(x, shape); b.input(y, shape)
        inputs = [x, y]; body = []; values = []
        if op == 'scale_shift':
            for key in ('scale', 'shift'):
                b.input(names[key], [1]); inputs.append(names[key])
                body.append(Load(key+'_val', names[key], (_l(0),), dtype=F32))
        base = _b('*', _v('cell'), _l(width))
        for lane in range(width):
            index = _b('+', base, _l(lane)); a, c, v = 'a'+str(lane), 'b'+str(lane), 'z'+str(lane)
            stmt = [Load(a, x, (index,), dtype=F32), Load(c, y, (index,), dtype=F32)]
            if op == 'relu_mul':
                stmt += [Let('zero'+str(lane), _l(0.0), F32), _assign('rel'+str(lane), 'maximum', a, 'zero'+str(lane)),
                         _assign(v, 'multiply', 'rel'+str(lane), c)]
            elif op == 'scale_shift':
                stmt += [_assign('scaled'+str(lane), 'multiply', a, 'scale_val'), _assign(v, 'add', 'scaled'+str(lane), 'shift_val')]
            else: stmt.append(_assign(v, op, a, c))
            # Full vectors are grouped. A remainder uses scalar guarded lanes.
            if n % width:
                stmt.append(Write(out, (index,), v, value_dtype=F32))
                body.append(Cond(_b('<', index, _l(n)), tuple(stmt)))
            else: body.extend(stmt); values.append(v)
        if not n % width: body.append(Write(out, (base,), values=tuple(values), value_dtype=F32))
        b.kernel(out, shape, inputs, [Axis('cell', math.ceil(n/width))], body)
    elif family == 'reduction':
        # Cooperative template is typed-loaded, then only its proven extent field
        # and matching graph/ABI shape change. All collective participation stays128.
        source = Path(__file__).resolve().parents[1]/'emmy-hackathon-2026/problems/r01_array_sum'
        g = Graph.from_dict(json.loads(next(source.glob('*.ir.json')).read_text()))
        node = next(v for v in g.nodes.values() if isinstance(v.op, KernelOp))
        from emmy.compiler.dim import Dim
        n, = shape
        for v in g.nodes.values():
            v.outputs = tuple(replace(t, shape=(Dim(n),)) if t.name == 'x' else t for t in v.outputs)
        tile = node.op.body[0]; loop = tile.body[0]
        node.op = replace(node.op, body=(replace(tile, body=(replace(loop, axis=replace(loop.axis, extent=Dim(n))),)+tile.body[1:]),))
        b.graph = g
        names.update(x='x', out=node.id)
        meta = json.loads((source/'meta.json').read_text())
        b.kernels = meta['kernels']
        for a in b.kernels[0]['args']:
            if a['name'] == 'x': a['shape'] = [n]
        spec['block'] = b.kernels[0]['block'][0]
        spec['schedule'] = 'cooperative-template-128-full-participation'
    elif family in ('matmul', 'atomic_matmul'):
        m, n, k = shape; b.input(x, (m,k)); b.input(y, (k,n))
        b.matmul(x, y, out, m, n, k, atomic=family=='atomic_matmul')
    elif family == 'softmax' and spec['schedule']=='cooperative-online-softmax':
        # Preserve the upstream coupled-state schedule, changing only documented
        # row/loop extents and tensor shapes. All 128 CTA lanes still participate.
        source=Path(__file__).resolve().parents[1]/'emmy-hackathon-2026/problems/f02_row_softmax'
        graph=Graph.from_dict(json.loads(next(source.glob('*.ir.json')).read_bytes()))
        node=next(v for v in graph.nodes.values() if isinstance(v.op,KernelOp))
        m,n=shape
        def change(statement):
            if isinstance(statement,Tile):
                return replace(statement,axes=tuple(replace(a,extent=m) if a.name=='a0' else a for a in statement.axes),
                               body=tuple(change(s) for s in statement.body))
            if isinstance(statement,StridedLoop):
                if statement.axis.name not in ('a1','a3') or statement.end is not None:
                    raise ValueError('upstream online template changed; independently review a new generator version')
                return replace(statement,axis=replace(statement.axis,extent=n),body=tuple(change(s) for s in statement.body))
            return statement
        for v in graph.nodes.values(): v.outputs=tuple(replace(t,shape=(m,n)) for t in v.outputs)
        node.op=replace(node.op,body=tuple(change(s) for s in node.op.body),name=node.op.name+'_'+format(int(rng.integers(0,2**63)), 'x'))
        b.graph=graph; b.kernels=json.loads((source/'meta.json').read_bytes())['kernels']
        k=b.kernels[0]; k['entry']=node.op.name; k['grid']=[m,1,1]
        for a in k['args']: a['shape']=[m,n]
        b.rng.shuffle(k['args']); k['arg_order']=[a['name'] for a in k['args']]
        names.update(x='x',out=node.id); spec['block']=128
    elif family == 'softmax':
        m, n = shape; b.input(x, shape); b.softmax(x, out, m, n)
    elif family == 'attention':
        m, n, d, dv = shape
        b.input(x, (m,d)); b.input(y, (d,n)); b.input(names['v'], (n,dv)); b.input(names['scale'], (1,))
        scores, probs = names['scores'], names['probs']
        b.matmul(x,y,scores,m,n,d,scale=names['scale'])
        b.softmax(scores,probs,m,n)
        b.matmul(probs,names['v'],out,m,dv,n)
        spec['schedule'] = 'ordered-three-kernel-attention'
    elif family in ('rmsnorm','l2norm'):
        m,n = shape; b.input(x,shape)
        body = [Loop(Axis('red',n),(Load('val',x,(_v('row'),_v('red')),dtype=F32),
                  _assign('square','multiply','val','val'),_fold('squares','square',axes=('red',))))]
        if family=='rmsnorm':
            body += [Let('count',_l(float(n)),F32),_assign('sumval','divide','squares','count')]
            total='sumval'
        else: total='squares'
        body += [Let('epsilon',_l(spec['epsilon']),F32),_assign('plus_eps','add',total,'epsilon'),
                 _assign('denom','sqrt','plus_eps'),Loop(Axis('col',n),(Load('value',x,(_v('row'),_v('col')),dtype=F32),
                 _assign('normal','divide','value','denom'),Write(out,(_v('row'),_v('col')),'normal',value_dtype=F32)))]
        b.kernel(out,shape,[x],[Axis('row',m)],body)
    else: raise ValueError('unknown family')
    b.graph.outputs = [names['out']]
    b.kernels[-1]['final'] = True
    return b


def generate_case(destination, family, seed, *, shape=None, pattern='normal', operation=None,
                  vector_width=None, block=None, schedule='serial'):
    if family not in TOLERANCES: raise ValueError('unknown family')
    rng = np.random.default_rng(seed)
    rank = {'pointwise':1,'reduction':1,'matmul':3,'atomic_matmul':3,'softmax':2,'attention':4,'rmsnorm':2,'l2norm':2}[family]
    defaults = [int(rng.choice([1,7,31,32,33,127,128,129,257,1024,4096]))] if rank==1 else [int(rng.choice([1,3,7,16,31,32,33])) for _ in range(rank)]
    shape = list(shape or defaults)
    if len(shape)!=rank or any(type(v) is not int or v<1 or v>16384 for v in shape): raise ValueError('invalid shape')
    if math.prod(shape)>2**24: raise ValueError('case too large')
    block = block or int(rng.choice([64,128,256]))
    if block not in (32,64,128,256): raise ValueError('invalid block')
    width = vector_width or int(rng.choice([1,2,4]))
    if width not in (1,2,4): raise ValueError('invalid vector width')
    operation = operation or str(rng.choice(['add','subtract','multiply','relu_mul','scale_shift']))
    if operation not in ('add','subtract','multiply','relu_mul','scale_shift'): raise ValueError('invalid operation')
    names = {k:k+'_'+str(int(rng.integers(100000,999999))) for k in ['x','y','out','v','scale','shift','scores','probs']}
    if schedule not in ('serial','cooperative') or (schedule=='cooperative' and family!='softmax'):
        raise ValueError('cooperative schedule is supported only for online softmax')
    spec = {'version':VERSION,'family':family,'seed':int(seed),'shape':shape,'pattern':pattern,'names':names,
            'operation':operation,'block':block,'vector_width':width,'epsilon':1e-5,
            'schedule':'cooperative-online-softmax' if schedule=='cooperative' else 'serial-per-cell',
            'reference':'independent-numpy-f64/plain-stable-softmax'}
    b = _build(spec,rng)
    input_rng=np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    inputs = {n:_inputs(input_rng, tuple(d.as_static() for d in node.output.shape),pattern) for n,node in b.graph.nodes.items() if isinstance(node.op,InputOp)}
    if names['scale'] in inputs: inputs[names['scale']] = np.array([math.sqrt(shape[2])*float(input_rng.choice([.5,1.,2.])) if family=='attention' else float(input_rng.choice([.25,.5,.75,1.25]))],dtype='float32')
    if names['shift'] in inputs: inputs[names['shift']] = np.array([float(input_rng.choice([-2.,-.25,.5,1.]))],dtype='float32')
    # Input data stays in inputs.npz, outside the compiler's minimal directory.
    rtol,atol = TOLERANCES[family]
    meta = {'id':family+'-'+str(seed),'graph_inputs':list(inputs),'graph_outputs':[names['out']],
            'rtol':rtol,'atol':atol,'kernels':b.kernels}
    destination = Path(destination); destination.mkdir(parents=True,exist_ok=False)
    ir_paths=[]
    for k in b.kernels:
        path=destination/(k['entry']+'.ir.json')
        path.write_bytes(canonical(b.graph.to_dict())); ir_paths.append(path.name)
        # Verify serialization through the actual required deserializer.
        Graph.from_dict(json.loads(path.read_bytes()))
    (destination/'meta.json').write_bytes(canonical(meta))
    save_arrays(destination/'inputs.npz',inputs)
    save_arrays(destination/'expected.npz',reference(spec,inputs))
    files={p.name:digest(p.read_bytes()) for p in sorted(destination.iterdir())}
    record={'version':VERSION,'spec':spec,'id':digest(canonical(files)),'files':files,'ir_paths':ir_paths,
            'tolerance_justification':'Fixed published-family policy; bounded finite f32 inputs, independent f64 oracle. No candidate-specific changes.',
            'requires_sanitizer':family=='atomic_matmul','requires_isolated_first':family=='atomic_matmul'}
    (destination/'case.json').write_bytes(canonical(record))
    validate_case(destination)
    return destination


def validate_case(directory):
    directory=Path(directory); record=json.loads((directory/'case.json').read_bytes())
    for name,expected in record['files'].items():
        if Path(name).name!=name or digest((directory/name).read_bytes())!=expected: raise ValueError('case artifact changed: '+name)
    meta=json.loads((directory/'meta.json').read_bytes())
    with np.load(directory/'inputs.npz',allow_pickle=False) as f: inputs={n:f[n] for n in f.files}
    with np.load(directory/'expected.npz',allow_pickle=False) as f: outputs={n:f[n] for n in f.files}
    if not all(np.isfinite(a).all() for a in [*inputs.values(),*outputs.values()]): raise ValueError('nonfinite case')
    known=set(inputs)
    for k,path in zip(meta['kernels'],record['ir_paths'],strict=True):
        graph=Graph.from_dict(json.loads((directory/path).read_bytes()))
        if not isinstance(graph.nodes[k['node']].op,KernelOp): raise ValueError('not a kernel')
        if graph.nodes[k['node']].op.name!=k['entry']: raise ValueError('wrong entry')
        names=[a['name'] for a in k['args']]
        if len(set(names))!=len(names) or len(k['arg_order'])!=len(names) or set(k['arg_order'])!=set(names): raise ValueError('invalid ABI')
        if math.prod(k['block'])>1024 or math.prod(k['grid'])<1: raise ValueError('invalid launch')
        for arg in k['args']:
            if arg.get('scalar'):
                if arg['shape'] or 'value' not in arg or not math.isfinite(arg['value']): raise ValueError('invalid scalar')
            elif not arg.get('output') and arg['name'] not in known: raise ValueError('missing input')
            if arg['name'] in inputs and list(inputs[arg['name']].shape)!=arg['shape']: raise ValueError('shape mismatch')
            if arg.get('output'): known.add(arg['name'])
    if set(meta['graph_outputs'])!=set(outputs): raise ValueError('wrong expected outputs')
    return {'valid':True,'id':record['id'],'family':record.get('spec',{}).get('family'),'kernels':len(meta['kernels'])}


def import_problem(source,destination):
    """Copy unchanged published IR/metadata/expectations plus realized inputs."""
    source,destination=Path(source),Path(destination); destination.mkdir(parents=True,exist_ok=False)
    meta=json.loads((source/'meta.json').read_bytes()); inputs={}
    for k in meta['kernels']:
        for a in k['args']:
            if a.get('scalar') or a.get('output'): continue
            if 'data' in a: inputs[a['name']]=np.asarray(a['data'],dtype=a['dtype'].replace('f','float').replace('i','int')).reshape(a['shape'])
            elif 'seed' in a: inputs[a['name']]=np.random.default_rng(a['seed']).standard_normal(a['shape']).astype(a['dtype'].replace('f','float').replace('i','int'))
    for p in [source/'meta.json',source/'expected.npz',*sorted(source.glob('*.ir.json'))]: (destination/p.name).write_bytes(p.read_bytes())
    save_arrays(destination/'inputs.npz',inputs)
    paths=[]
    for k in meta['kernels']:
        matches=[p.name for p in source.glob('*.ir.json') if k['node'] in Graph.from_dict(json.loads(p.read_bytes())).nodes]
        # Prefer exact entry filename if a graph includes several kernels.
        exact=[p for p in matches if p==k['entry']+'.ir.json']; paths.append((exact or matches)[0])
    files={p.name:digest(p.read_bytes()) for p in sorted(destination.iterdir())}
    record={'version':VERSION,'id':digest(canonical(files)),'files':files,'ir_paths':paths,
            'spec':{'family':'published','problem':meta['id']},
            'requires_sanitizer':bool(any(k.get('zero_outputs') for k in meta['kernels'])),
            'requires_isolated_first':bool(any(k.get('zero_outputs') for k in meta['kernels']))}
    (destination/'case.json').write_bytes(canonical(record)); validate_case(destination)
    return destination


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('destination',type=Path); p.add_argument('--family',choices=sorted(TOLERANCES),default='pointwise'); p.add_argument('--seed',type=int,required=True)
    p.add_argument('--shape',type=int,nargs='+'); p.add_argument('--pattern',choices=sorted(DOMAINS),default='normal')
    args=p.parse_args(); print(generate_case(args.destination,args.family,args.seed,shape=args.shape,pattern=args.pattern))
