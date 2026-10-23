"""Replay the remote coverage audit against unchanged frozen cases; CPU only."""
import ast,hashlib,json,re,sys,importlib.util
from pathlib import Path
def stable_dump(node):
    def encode(value):
        if isinstance(value,ast.AST):
            return [type(value).__name__,sorted((name,encode(child)) for name,child in ast.iter_fields(value) if child is not None and not (isinstance(child,list) and not child))]
        if isinstance(value,list):return [encode(child) for child in value]
        return value
    return json.dumps(encode(node),sort_keys=True,separators=(',',':'))

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

root=Path(sys.argv[1]);folder=Path(sys.argv[2]);catalog=Path(sys.argv[3])
spec=importlib.util.spec_from_file_location('audited_selector',folder/'regression_suite.py');selector=importlib.util.module_from_spec(spec);spec.loader.exec_module(selector)
policy=folder/'policy.json';audit=selector.read(folder/'gpu-regression-coverage.json')
selection=selector.select([catalog],policy)
receipt=selector.read(root/'objects'/audit['receipt_id']/'manifest.json')['payload']['report']
assert receipt['source_id']==audit['source_id'] and receipt['sanitizer_passed']
checks={r['case_id']:r for r in receipt['cases']};assert set(checks)=={r['case_id'] for r in audit['cases']}
index={}
for path in selector.catalog_paths(catalog):
 r=selector.read(Path(path)/'case.json');index.setdefault(r['id'],Path(path))
rows={}
for row in audit['cases']:
 cid=row['case_id'];path=index[cid];rec=selector.verify_case(path,{'record_sha256':row['record_sha256']})
 check=checks[cid];assert check['status']=='pass' and check['guards_ok'] and check['inputs_ok'] and check['isolated']
 docs={n:selector.read(path/n) for n in rec['files'] if n.endswith('.ir.json') or n=='meta.json'}
 rows[cid]={'record':rec,'documents':docs}
 assert topology(rows[cid])==row['sample_normalized_ir_sha256']
all_surface=set().union(*(surface(x) for x in rows.values()));selected=set(selection['case_ids'])
retained_surface=set().union(*(surface(rows[cid]) for cid in selected))
assert all_surface==retained_surface==set(audit['surface_tokens'])
for row in audit['cases']:
 if row['selected']:continue
 same=any(topology(rows[row['case_id']])==topology(rows[w]) for w in row['witness_ids'])
 assert same==row['same_sample_normalized_ir_topology']
 if not same:
  old=tile_statements(rows[row['case_id']])[0];matched=False
  for w in row['witness_ids']:
   pos=0
   for statement in tile_statements(rows[w])[0]:
    if pos<len(old) and statement==old[pos]:pos+=1
   matched |= pos==len(old)
  assert matched
out=folder/'case-list.json'
with out.open('x') as f:json.dump(selection,f,sort_keys=True);f.write('\n')
result={'source_id':audit['source_id'],'compiler_sha256':selector.current(root)['compiler_sha256'],'receipt_id':audit['receipt_id'],'verified_campaign_cases':len(rows),'selected_cases':len(selected),'generated_cases':selection['generated_cases'],'excluded_cases':len(rows)-len(selected),'surface_tokens':len(all_surface),'missing_surface_tokens':[],'all_case_files_and_records_verified':True,'existing_isolated_guarded_receipt_passed':True,'fresh_gpu_execution':False,'compound_programs_rechecked':2,'policy_sha256':selector.sha(policy.read_bytes())}
with (folder/'verified-source-evidence.json').open('x') as f:json.dump(result,f,sort_keys=True);f.write('\n')
value=selector.install(root,selection,out,[catalog],policy)
assert value['source_id']==audit['source_id']
from tools.research_store import atomic_json
value['verified_source_evidence']=str(folder/'verified-source-evidence.json');value['coverage_audit']=str(folder/'gpu-regression-coverage.json');atomic_json(root/'regression-suites/current.json',value)
print(json.dumps(result,indent=2))
