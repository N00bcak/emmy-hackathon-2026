"""Parent-only extra synccheck for the four independently validated partial CTA cases."""
import os,sys,time,json,uuid,subprocess
from pathlib import Path
from tools.research_store import Store,read_json,atomic_json
from tools import research_pipeline as p
C=Path(__file__).resolve().parent;P=C.parent/'campaign-20261022-12h-v1'
s=Store('/root/nfs/emmy/research','/root/bcy/emmy/research-workspaces')
exp='correctness-partial-uniform-loop-v3';m=s._manifest(exp);out=C/'partial-v3-synccheck'
assert os.environ.get('EMMY_JOB_DIR') and not out.exists(),'one immutable attempt; inspect before any continuation'
out.mkdir();owner='sync-'+uuid.uuid4().hex[:8];job=s.claim_gpu_auxiliary(owner,exp);assert job
s.gpu_job_ids(job['job_id'],owner,[os.getpid(),os.environ['EMMY_JOB_DIR']]);rows=[]
try:
 for label in ['partial-uniform-oracle-validation-667788','partial-uniform-independent-v3']:
  root=P/'diagnostics'/exp/label;assert read_json(root/'identity.json')['source_id']==m['source_id']
  for spec in p._module_specs(read_json(root/'preparation/prepared.json'),exp):
   d=out/spec['id'];d.mkdir();resultpath=d/'result.json';jobpath=d/'job.json'
   atomic_json(jobpath,{'mode':'correctness','cases':[spec],'width':1,'repetitions':3,'out':str(resultpath)})
   command=['compute-sanitizer','--tool','synccheck','--error-exitcode','99',sys.executable,'-m','tools.research_gpu',str(jobpath)]
   r=subprocess.run(command,capture_output=True,timeout=45);(d/'stdout').write_bytes(r.stdout);(d/'stderr').write_bytes(r.stderr)
   result=read_json(resultpath) if resultpath.exists() else {}
   ok=r.returncode==0 and b'ERROR SUMMARY: 0 errors' in r.stdout+r.stderr and p._checks_ok(result)
   row={'case_id':spec['id'],'source_id':m['source_id'],'ok':ok,'exit_code':r.returncode,'command':command,'host_job_dir':os.environ['EMMY_JOB_DIR'],'result':result}
   aid=s.put_artifact('partial-cta-synccheck',row,p.artifacts_from_tree(d));rows.append({'artifact_id':aid,**{k:row[k] for k in ['case_id','source_id','ok','exit_code','host_job_dir']}});atomic_json(out/'progress.json',rows)
   assert ok,'preserved synccheck failure; no retry/promotion'
 atomic_json(out/'summary.json',{'source_id':m['source_id'],'reservation':m['confirmation'],'rows':rows,'all_ok':all(r['ok'] for r in rows),'host_job_dir':os.environ['EMMY_JOB_DIR']})
 print(json.dumps({'source_id':m['source_id'],'cases':len(rows),'all_ok':True,'root':str(out)}))
finally:s.finish_gpu(job['job_id'],owner,{})
