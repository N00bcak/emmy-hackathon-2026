import ast,collections,gzip,hashlib,json
from pathlib import Path
def stable_dump(node):
    def encode(value):
        if isinstance(value,ast.AST):
            return [type(value).__name__,sorted((name,encode(child)) for name,child in ast.iter_fields(value) if child is not None and not (isinstance(child,list) and not child))]
        if isinstance(value,list):return [encode(child) for child in value]
        return value
    return json.dumps(encode(node),sort_keys=True,separators=(',',':'))

BASE=Path('experiments/2026-10-23-campaign-current/coverage-audit-v4')
rows=json.load(open(BASE/'campaign-rows.json')); old=json.load(open('experiments/2026-10-23-campaign-current/test-consolidation-v3/policy.json')); chosen={x['id']:dict(x) for x in old['cases']};byid={x['record']['id']:x for x in rows}
def v(x):return x['record']['spec'].get('version',x['record']['version'])
def pick(version,predicate,reason):
 x=next(x for x in rows if v(x)==version and predicate(x['record']['spec']))
 r=x['record'];chosen[r['id']]={'id':r['id'],'record_sha256':hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'family':r['spec']['family'],'selector':{'version':version},'reason':reason}
 return r['id']
additions=[]
def add(*args):
 cid=pick(*args);additions.append(cid);return cid
for family in ['rmsnorm','l2norm','atomic_matmul','softmax','attention']:
 add('emmy-research-cases-scalar-flatindex-v3',lambda s,f=family:s['family']==f and (f!='softmax' or s.get('schedule')=='serial-per-cell'),f'Independent generated {family}; '+('ordered three-kernel intermediate/ABI execution' if family=='attention' else 'preserve distinct generated lowering schedule'))
# Left shift has one width-specific b64 emitter for both signednesses; keep both right-shift opcodes.
add('native64-shift-oracle-v1',lambda s:s['ir_dtype']=='i64' and s['op']=='left_shift','64-bit bitwise left-shift and u32 shift-count lowering; signed/unsigned left shift share shl.b64')
add('native64-shift-oracle-v1',lambda s:s['ir_dtype']=='u64' and s['op']=='right_shift','Unsigned 64-bit logical right shift must not become signed shr.s64')
for schedule in ['canonical-tail','short-ring','nondivisible-counter','observable-padding']:
 add('async-ring-independent-sum-oracle-v1',lambda s,k=schedule:s['schedule']==k,'Async ring lifetime: '+schedule)
add('async-ring-independent-sum-oracle-v1',lambda s:s['schedule']=='short-ring' and s['extent']==1,'Async ring shorter than prefill lead; initial invalid prefill differs from extent == lead')
for length,live in [(2,False),(4,False),(8,False),(4,True)]:
 add('exact-shared-tree-f32-order-v1',lambda s,n=length,l=live:s['length']==n and s['scratch_liveout']==l,f'Exact F32 shared-tree order at length {length}; '+('scratch continuation is observable' if live else 'immutable pre-tree snapshot/root broadcast'))
for extent in [8,9,12,17]:
 add('shared-pair-per-lane-serial-dot-v1',lambda s,n=extent:s['extent']==n,f'Shared pair serial dots extent {extent}: preserve alignment, explicit/implicit seed and tail boundary')
add('mixed-f16-result-compute-boundary-v1',lambda s:True,'Explicit F16 result with F32 operands computes before one rounding; native all-F16 control')
for tail in range(4):
 add('strided-per-lane-serial-add-oracle-v1',lambda s,t=tail:s['tail_mod4']==t,f'Strided per-lane strict F32 addition tail mod4={tail}; seed and explicit-init order')
for version,schedules in [
 ('fused-online-cooperative-coverage-v3',['full-segment16','full-segment32','partial-fallback']),
 ('fused-online-cooperative-coverage-v4',['full-segment16','full-segment32']),
 ('fused-online-cooperative-coverage-v5-full-tail',['full-segment16','full-segment32']),
 ('distinct-query-normalization-partition-v7',['full-segment16','full-segment32']),
 ('fused-online-seeded-key-coefficients-v8',['full-segment16','full-segment32','tail32','partial-fallback']),
 ('duplicate-normalization-attention-v6',['full-segment16','full-segment32'])]:
 for schedule in schedules:
  add(version,lambda s,k=schedule:s['schedule'].endswith(k),f'{version}: {schedule}; preserve seed, numerical domain and normalization dataflow')
add('scalar-bool-ssa-independent-oracle-v2',lambda s:s['schedule']=='empty-folds-seed-false','Zero-trip BOOL folds preserve initial carriers; a nonempty fold does not cover this path')
add('nested-key-owned-score-scope-v1',lambda s:s['width']==8,'Optimized KEY nested score scope at subgroup width8 as well as width16')
# Replace the single-iteration sample with an existing multi-iteration witness, not a new generator.
remove=next(c['id'] for c in old['cases'] if c.get('selector',{}).get('version')=='nested-seeded-fold-lexical-scope-v1')
chosen.pop(remove)
replacement=pick('nested-seeded-fold-lexical-scope-v1',lambda s:s['nested_steps']>1 and s['stride_inner']>1,'Nested seeded fold scope, multi-trip ordinary/strided bodies and seedFalse live updates; replaces single-trip witness')
print('old',len(old['cases']),'added',len(additions),'replaced',remove[:10],replacement[:10],'new',len(chosen),'generated',len(chosen)-15)
new=dict(old);new['version']='gpu-regression-compact-v4';new['cases']=list(chosen.values());new['scope']='Audited remote campaign behavior witnesses, with explicit per-case representation map; input samples are not claimed equivalent.'
(BASE/'policy.json').write_text(json.dumps(new,sort_keys=True,separators=(',',':'))+'\n')
# Preserve actual parsed AST features, including constants and all statement/operation names.
def features(x):
 calls=collections.Counter();constants=collections.Counter();flags=collections.Counter();bodies=[]
 for name,doc in x['documents'].items():
  if name=='meta.json':continue
  for node in doc.get('nodes',{}).values():
   if node.get('op')!='KernelOp':continue
   for b in node['op_fields']['body']:
    tree=ast.parse(b,mode='eval');bodies.append(stable_dump(tree))
    for n in ast.walk(tree):
     if isinstance(n,ast.Call):
      calls[ast.unparse(n.func)]+=1
      if isinstance(n.func,ast.Name) and n.func.id in ['ElementwiseImpl','DataType'] and n.args and isinstance(n.args[0],ast.Constant):calls[n.func.id+':'+str(n.args[0].value)]+=1
     if isinstance(n,ast.keyword) and n.arg in ['seed','atomic','unroll'] and isinstance(n.value,ast.Constant):flags[n.arg+':'+str(n.value.value)]+=1
     if isinstance(n,ast.Constant):constants[repr(n.value)]+=1
 return {'calls':dict(calls),'flags':dict(flags),'constants':dict(constants),'ast_sha256':hashlib.sha256('\n'.join(bodies).encode()).hexdigest()}
(BASE/'ir-features.json').write_text(json.dumps({x['record']['id']:features(x) for x in rows},indent=2))
# Extract operation/dtype/control interactions from the actual expression ASTs.
def surface(x):
 result=set()
 for name,doc in x['documents'].items():
  if name=='meta.json':continue
  for node in doc.get('nodes',{}).values():
   if node.get('op')!='KernelOp':continue
   for body in node['op_fields']['body']:
    tree=ast.parse(body,mode='eval')
    for call in (n for n in ast.walk(tree) if isinstance(n,ast.Call)):
     fn=ast.unparse(call.func);result.add('node:'+fn)
     for kw in call.keywords:
      if kw.arg in ('op','dtype','func','seed','atomic','unroll') and isinstance(kw.value,ast.Constant): result.add(fn+':'+kw.arg+'='+repr(kw.value.value))
      if fn=='DataType' and kw.arg=='name' and isinstance(kw.value,ast.Constant):result.add('dtype:'+str(kw.value.value))
     if fn=='ElementwiseImpl' and call.args and isinstance(call.args[0],ast.Constant):result.add('operation:'+str(call.args[0].value))
 return result
surfaces={x['record']['id']:surface(x) for x in rows};all_surface=set().union(*surfaces.values());new_surface=set().union(*(surfaces[cid] for cid in chosen))
print('surface tokens all/retained',len(all_surface),len(new_surface),'missing',sorted(all_surface-new_surface))
(BASE/'surface.json').write_text(json.dumps({k:sorted(v) for k,v in surfaces.items()},indent=2))
# Keep boundary cases even where operation-token coverage alone would appear sufficient.
add('emmy-research-cases-scalar-flatindex-v3',lambda s:s['family']=='softmax' and s.get('schedule')=='cooperative-online-softmax' and s['shape'][1]==127,'Cooperative softmax single column slice vs existing 257-column multi-slice boundary')
add('partial-uniform-shared-neighbor-oracle-v1',lambda s:s['schedule']=='loop','Partial CTA direct uniform Loop barrier vs nested StridedLoop barrier entry')
new['cases']=list(chosen.values());(BASE/'policy.json').write_text(json.dumps(new,sort_keys=True,separators=(',',':'))+'\n')
print('Boundary-inclusive selection:',len(chosen),'total,',len(chosen)-15,'generated; no new cases generated')
# Restore distinct AST programs discovered by comparing remote bodies, not labels.
add('async-ring-independent-sum-oracle-v1',lambda s:s['slots']==4,'Four-slot/lead3 ring has an additional prefill stage; three-slot/lead2 IR is not its structural replacement')
add('native64-shift-oracle-v1',lambda s:s['ir_dtype']=='u64' and s['op']=='left_shift','Unsigned left-shift typed program retained explicitly, avoiding reliance on common opcode reasoning')
add('emmy-research-cases-scalar-flatindex-v3',lambda s:s['family']=='pointwise' and s['vector_width']==1 and s['pattern']=='bounded','Independent scalar F32 add with partial Tile and generated ABI names')
add('emmy-research-cases-scalar-flatindex-v3',lambda s:s['family']=='pointwise' and s['vector_width']==2,'Independent grouped width2 F32 add with tail and cancellation input domain')
add('emmy-research-cases-scalar-flatindex-v3',lambda s:s['family']=='matmul' and s['shape'][1]>1,'Standalone serial contraction program; ordered multi-kernel attention is not an identical graph')
new['cases']=list(chosen.values());(BASE/'policy.json').write_text(json.dumps(new,sort_keys=True,separators=(',',':'))+'\n')
policy_ids=set(chosen)
def witnesses(x):
 cid=x['record']['id'];s=x['record']['spec'];version=v(x)
 if cid in policy_ids:return [cid],'retained exact case'
 def find(version,predicate=lambda s:True):
  matches=[y['record']['id'] for y in rows if y['record']['id'] in policy_ids and v(y)==version and predicate(y['record']['spec'])]
  assert matches,(cid,version,s)
  return matches
 if version=='emmy-research-cases-scalar-flatindex-v3':
  f=s['family']
  if f=='pointwise':return find(version,lambda z:z['family']==f and z['vector_width']==s['vector_width']),'Same generated scalar/grouped width and add program; repeated numerical samples and full/tail shapes remain archived'
  if f=='reduction':return [y['record']['id'] for y in rows if y['record']['spec'].get('problem')=='r01_array_sum']+find('partial-reductions-primary-identities-v1'),'Cooperative warp/shared sum plus partial-lane fold identities; input preprocessing and samples differ'
  if f=='matmul':return find(version,lambda z:z['family']==f),'Same standalone serial contraction program; dimension and input samples remain archived'
  return find(version,lambda z:z['family']==f and z.get('schedule')==s.get('schedule') and (f!='softmax' or z['shape'][1]//128==s['shape'][1]//128 or s.get('schedule')=='serial-per-cell')),'Same generated family/schedule; repeated input domains, shape and ABI-name samples remain archived'
 if version.startswith('fused-online-cooperative-coverage-v'):
  if s['schedule'].endswith('partial-fallback'):
   return find('fused-online-cooperative-coverage-v3',lambda z:z['schedule'].endswith('partial-fallback')),'Same explicit-init seedFalse partial KEY fallback IR; version4 partial cases retain bounded domain'
  return find(version,lambda z:z['schedule']==s['schedule']),'Same KEY seed mode, segment width and numerical domain; full-tail version kept separately'
 if version in ['distinct-query-normalization-partition-v7','duplicate-normalization-attention-v6','fused-online-seeded-key-coefficients-v8']:
  return find(version,lambda z:z['schedule']==s['schedule']),'Same normalization dependency graph, KEY seed and segment/fallback class; repeated draws archived'
 if version=='native64-shift-oracle-v1':
  if s['op']=='left_shift':return find(version,lambda z:z['ir_dtype']==s['ir_dtype'] and z['op']==s['op']),'Same signedness-specific typed left-shift program and 32-bit shift count; repeated raw-bit sample archived'
  return find(version,lambda z:z['ir_dtype']==s['ir_dtype'] and z['op']==s['op']),'Same signedness-specific right-shift opcode and 32-bit shift count; raw-bit input sample archived'
 if version=='async-ring-independent-sum-oracle-v1':
  return find(version,lambda z:z['schedule']==s['schedule'] and z['slots']==s['slots'] and (s['schedule']!='short-ring' or (z['extent']<z['lead'])==(s['extent']<s['lead']))),'Same ring semantic class; short-ring prefill-invalid and prefill-full boundaries separately retained'
 if version=='exact-shared-tree-f32-order-v1':return find(version,lambda z:z['length']==s['length'] and z['scratch_liveout']==s['scratch_liveout']),'Same exact tree length/order, snapshot and scratch observability; cancellation coefficients vary'
 if version=='shared-pair-per-lane-serial-dot-v1':return find(version,lambda z:z['extent']==s['extent'] and z['row_stride']==s['row_stride'] and z['seeded']==s['seeded']),'Same alignment/stride, seed mode, serial dot order and full/tail boundary'
 if version=='strided-per-lane-serial-add-oracle-v1':return find(version,lambda z:z['tail_mod4']==s['tail_mod4'] and z['seeded']==s['seeded']),'Same per-lane strict serial addition, seed mode and tail residue; explicit initial sample differs'
 if version.startswith('scalar-bool-ssa-independent-oracle-v'):
  schedule='scalar-relations' if s['schedule']=='assign-less' else s['schedule']
  return find('scalar-bool-ssa-independent-oracle-v2',lambda z:z['schedule']==schedule),'Version2 retains version1 Assign-less inside broader scalar relations; zero/nonzero fold paths kept separately'
 if version=='lexical-carrier-shadow-oracle-v1':return find('lexical-carrier-shadow-oracle-v2'),'Version2 combined program contains Let, grouped Load, scalar Load shadow continuations plus Assign shadow'
 if version=='partial-uniform-shared-neighbor-oracle-v1':return find(version,lambda z:z['schedule']==s['schedule']),'Same direct/nested uniform barrier participation and partial CTA class; trip/tail sample archived'
 if version=='nested-key-owned-score-scope-v1':return find(version,lambda z:z['width']==s['width']),'Same optimized KEY frame/score ownership and subgroup width; ancestor constant sample differs'
 if version=='nested-seeded-fold-lexical-scope-v1':return find(version),'Multi-trip witness retains identical AST statement/control topology and all eight observable scope outputs; single-trip numeric sample archived'
 fields={'lexical-load-oracle-v1':['mode'],'opaque-child-address-oracle-v1':['type']}
 if version in fields:return find(version,lambda z:all(z[k]==s[k] for k in fields[version])),'Same expression node or declaration mode and observable continuation; numeric sample differs'
 if version in ['nested-tile-init-oracle-v1','select-lexical-shadow-snapshot-oracle-v1','softplus-f32-negative-tail-composition-v1','erf-f32-normal-small-composition-v1','lexical-carrier-shadow-oracle-v2']:return find(version),'Same combined observable program/semantic contract; repeated seed and constants archived'
 raise AssertionError(('UNMAPPED',version,cid,s))
audit=[]
for x in rows:
 cid=x['record']['id'];ws,reason=witnesses(x)
 assert ws and set(ws)<=policy_ids
 audit.append({'case_id':cid,'version':v(x),'selected':cid in policy_ids,'witness_ids':ws,'reason':reason,'same_input_bytes':any(x['record']['files'].get('inputs.npz')==byid[w]['record']['files'].get('inputs.npz') for w in ws),'ir_ast_sha256':features(x)['ast_sha256']})
coverage={'version':'remote-campaign-coverage-v1','source_id':'549b6047f91818ea876deb76adfc88cabc36f554c05ab2bc51b2c961aa9143f2','receipt_id':'a33674b0f2a33a317518c55b7d363fceb1b7e0d31d0c23a9ad16fd55f3e62eba','campaign_cases':278,'selected_ids':list(chosen),'surface_tokens':sorted(all_surface),'retained_surface_tokens':sorted(new_surface),'limitations':'Manual mechanism representation audit, not semantic equivalence or proof over unseen IRs. Input samples, dimensions and constants are intentionally sampled; all frozen cases remain for targeted replay. No new GPU execution.','cases':audit}
assert len(audit)==len({r['case_id'] for r in audit})==278
assert len(chosen)==122 and len(chosen)-15==107
out=json.dumps(coverage,sort_keys=True,separators=(',',':'))+'\n';(BASE/'gpu-regression-coverage.json').write_text(out)
new['coverage_manifest']={'filename':'gpu-regression-coverage.json','sha256':hashlib.sha256(out.encode()).hexdigest()}
(BASE/'policy.json').write_text(json.dumps(new,sort_keys=True,separators=(',',':'))+'\n')
print('Mapped',len(audit),'campaign cases; archived',sum(not r['selected'] for r in audit),'selected',sum(r['selected'] for r in audit),'surface missing',len(all_surface-new_surface))
import re
class SampleNormalizer(ast.NodeTransformer):
 def visit_Constant(self,n):
  if isinstance(n.value,bool) or n.value is None:return n
  if isinstance(n.value,(int,float)):return ast.copy_location(ast.Constant(value='NUM'),n)
  if isinstance(n.value,str):return ast.copy_location(ast.Constant(value=re.sub(r'_[0-9]+\b','_SAMPLE',n.value)),n)
  return n
 def visit_UnaryOp(self,n):
  if isinstance(n.operand,ast.Constant) and isinstance(n.operand.value,(int,float)) and not isinstance(n.operand.value,bool):return ast.copy_location(ast.Constant(value='NUM'),n)
  return self.generic_visit(n)
def topology(x):
 docs=[]
 for name,d in x['documents'].items():
  if name=='meta.json':continue
  bodies=[]
  for node in d.get('nodes',{}).values():
   if node.get('op')=='KernelOp':bodies.extend(stable_dump(SampleNormalizer().visit(ast.parse(b,mode='eval'))) for b in node['op_fields']['body'])
  docs.append(sorted(bodies))
 return hashlib.sha256(json.dumps(sorted(docs)).encode()).hexdigest()
top={x['record']['id']:topology(x) for x in rows}
for a in coverage['cases']:
 a['same_sample_normalized_ir_topology']=any(top[a['case_id']]==top[w] for w in a['witness_ids'])
 a['sample_normalized_ir_sha256']=top[a['case_id']]
# Normalization is diagnostic only: numeric boundary/seed/width classes are manually retained above.
coverage['normalization']='Diagnostic AST topology only: tensor-name suffixes and numeric literals normalized; Boolean constants, variable names, op names and keyword structure retained. Not used to declare semantic equivalence.'
out=json.dumps(coverage,sort_keys=True,separators=(',',':'))+'\n';(BASE/'gpu-regression-coverage.json').write_text(out)
new['coverage_manifest']['sha256']=hashlib.sha256(out.encode()).hexdigest();(BASE/'policy.json').write_text(json.dumps(new,sort_keys=True,separators=(',',':'))+'\n')
mismatches=collections.Counter(a['version'] for a in coverage['cases'] if not a['selected'] and not a['same_sample_normalized_ir_topology'])
print('Nonidentical normalized topologies (require explicit broader-witness rationale):',dict(mismatches))
def tile_statements(x):
 roots=[]
 for name,d in x['documents'].items():
  if name=='meta.json':continue
  for node in d.get('nodes',{}).values():
   if node.get('op')=='KernelOp':
    for body in node['op_fields']['body']:
     tree=SampleNormalizer().visit(ast.parse(body,mode='eval'))
     for kw in tree.body.keywords:
      if kw.arg=='body':roots.append([stable_dump(n) for n in kw.value.elts])
 return roots
compound=[]
for a in coverage['cases']:
 if a['selected'] or a['same_sample_normalized_ir_topology']:continue
 x=byid[a['case_id']];oldbody=tile_statements(x)[0]
 checks=[]
 for cid in a['witness_ids']:
  newbody=tile_statements(byid[cid])[0];pos=0
  for statement in newbody:
   if pos<len(oldbody) and statement==oldbody[pos]:pos+=1
  checks.append({'witness_id':cid,'old_statements':len(oldbody),'matched_ordered_statements':pos,'ordered_subsequence':pos==len(oldbody)})
 compound.append({'case_id':a['case_id'],'checks':checks})
print('Compound-program checks:',compound)
(BASE/'compound-checks.json').write_text(json.dumps(compound,indent=2))
coverage['compound_checks']=compound
assert all(any(c['ordered_subsequence'] for c in x['checks']) for x in compound)
coverage['summary']={'prior_selected':73,'selected':122,'published':15,'generated':107,'excluded':156,'excluded_same_normalized_topology':154,'excluded_ordered_compound_subsequence':2,'surface_tokens':94,'new_generated_types':0}
out=json.dumps(coverage,sort_keys=True,separators=(',',':'))+'\n';(BASE/'gpu-regression-coverage.json').write_text(out)
new['coverage_manifest']['sha256']=hashlib.sha256(out.encode()).hexdigest();(BASE/'policy.json').write_text(json.dumps(new,sort_keys=True,separators=(',',':'))+'\n')
