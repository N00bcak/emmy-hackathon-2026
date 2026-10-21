"""CPU-only candidate isolation, assembly cache, and evidence-driven static gate."""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from tools.research_cases import canonical, digest, validate_case


def _restrict():
    """Linux compiler boundary: no privilege escalation, network, or evaluator UID."""
    # Resolving the SONAME may invoke ldconfig. Do this before NPROC/AS limits:
    # rootless host-UID process accounting can otherwise make lookup return None.
    library=ctypes.util.find_library('seccomp')
    if not library: raise RuntimeError('libseccomp is required for compiler isolation')
    sec=ctypes.CDLL(library,use_errno=True)
    resource.setrlimit(resource.RLIMIT_CORE, (0,0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (2_100_000,2_100_000))
    resource.setrlimit(resource.RLIMIT_AS, (2*1024**3,2*1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (15,15))
    resource.setrlimit(resource.RLIMIT_NPROC, (64,64))
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(38,1,0,0,0): raise OSError('PR_SET_NO_NEW_PRIVS failed')
    sec.seccomp_init.argtypes=[ctypes.c_uint32]; sec.seccomp_init.restype=ctypes.c_void_p
    sec.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p]; sec.seccomp_syscall_resolve_name.restype=ctypes.c_int
    sec.seccomp_rule_add.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint]
    sec.seccomp_load.argtypes=[ctypes.c_void_p]; sec.seccomp_release.argtypes=[ctypes.c_void_p]
    ctx=sec.seccomp_init(0x7fff0000)
    if not ctx: raise RuntimeError('seccomp init failed')
    try:
        for name in ('socket','socketpair','connect','ptrace','process_vm_readv','process_vm_writev',
                     'setsid','setpgid','kill','tkill','tgkill'):
            nr=sec.seccomp_syscall_resolve_name(name.encode())
            if nr>=0 and sec.seccomp_rule_add(ctx,0x00050000|1,nr,0): raise RuntimeError('seccomp rule failed')
        if sec.seccomp_load(ctx): raise RuntimeError('seccomp load failed')
    finally: sec.seccomp_release(ctx)
    _restrict_ioctl(libc, sec.seccomp_syscall_resolve_name(b'ioctl'))
    os.setgroups([]); os.setgid(65534); os.setuid(65534)
    # Credential changes clear PDEATHSIG, so install it after dropping identity.
    expected_parent=int(os.environ['EMMY_COMPILER_PARENT_PID'])
    if libc.prctl(1,signal.SIGKILL,0,0,0) or os.getppid()!=expected_parent:
        raise RuntimeError('compiler parent disappeared before restricted execution')


def _restrict_ioctl(libc, syscall_number):
    """Deny device ioctls while retaining CPython's close-on-exec/TTY probes.

    FIOCLEX is used when opening ordinary files; denying every ioctl prevents
    Python from opening its script. No NVIDIA/DRM/UVM request is whitelisted.
    The preceding libseccomp filter already validates the native ABI.
    """
    if syscall_number < 0: raise RuntimeError('ioctl syscall is unavailable')
    class Instruction(ctypes.Structure):
        _fields_=[('code',ctypes.c_ushort),('jt',ctypes.c_ubyte),('jf',ctypes.c_ubyte),('k',ctypes.c_uint32)]
    class Program(ctypes.Structure):
        _fields_=[('length',ctypes.c_ushort),('instructions',ctypes.POINTER(Instruction))]
    allowed=(0x5451,0x5450,0x5401,0x5413)  # FIOCLEX, FIONCLEX, TCGETS, TIOCGWINSZ
    n=len(allowed)
    rules=[(0x20,0,0,0),(0x15,0,n+2,syscall_number),(0x20,0,0,24)]
    rules += [(0x15,n-i,0,request) for i,request in enumerate(allowed)]
    rules += [(0x06,0,0,0x00050000|errno.EPERM),(0x06,0,0,0x7fff0000)]
    instructions=(Instruction*len(rules))(*(Instruction(*r) for r in rules))
    program=Program(len(rules),instructions)
    if libc.prctl(22,2,ctypes.byref(program),0,0):
        raise OSError(ctypes.get_errno(),'ioctl seccomp filter failed')


def _kill_group(pid):
    try:
        os.killpg(pid,signal.SIGKILL)
    except ProcessLookupError:
        pass
    # compile_ir is a subreaper: detached parents cannot leave zombies under a
    # non-reaping container PID1. Wait only for children in this owned group.
    deadline=time.monotonic()+1
    while time.monotonic()<deadline:
        try:
            reaped,_=os.waitpid(-pid,os.WNOHANG)
            if not reaped: time.sleep(.01)
        except ChildProcessError:
            break


def compile_ir(source, case_dir, ir_name, out, *, timeout=20):
    if sys.platform!='linux' or os.geteuid()!=0:
        raise RuntimeError('compile boundary requires container root on Linux; no insecure fallback')
    if ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0):
        raise OSError('cannot install compiler descendant subreaper')
    source,case_dir,out=Path(source),Path(case_dir),Path(out)
    out.mkdir(parents=True,exist_ok=True)
    validate_case(case_dir)
    record=json.loads((case_dir/'case.json').read_bytes())
    if ir_name not in record['ir_paths']: raise ValueError('IR outside validated case')
    # Minimal compiler view excludes references, runtime samples, and source sidecars.
    sandbox=Path(tempfile.mkdtemp(prefix='emmy-compiler-'))
    try:
        # Reject before copytree: following a link would expose its target first.
        if source.is_symlink() or not source.is_dir():
            raise ValueError('compiler snapshot must be a directory without symlinks')
        for p in source.rglob('*'):
            if p.is_symlink() or not (p.is_file() or p.is_dir()):
                raise ValueError('compiler snapshot must not contain symlinks or special files')
        src=sandbox/'source'; shutil.copytree(source,src,symlinks=False)
        for p in src.rglob('*'):
            if p.is_symlink(): raise ValueError('compiler snapshot must not contain symlinks')
            p.chmod(0o555 if p.is_dir() else 0o444)
        src.chmod(0o555)
        inputs=sandbox/'input'; inputs.mkdir()
        (inputs/ir_name).write_bytes((case_dir/ir_name).read_bytes())
        meta=json.loads((case_dir/'meta.json').read_bytes())
        for k in meta['kernels']:
            for arg in k['args']:
                arg.pop('seed',None); arg.pop('data',None)
        for key in ('code','id'): meta.pop(key,None)
        (inputs/'meta.json').write_bytes(canonical(meta))
        for p in inputs.iterdir(): p.chmod(0o444)
        inputs.chmod(0o555); sandbox.chmod(0o755)
        command=[sys.executable,'-B','-m','tools.research_prepare','--compiler-child',str(src/'compile.py'),str(inputs/ir_name)]
        started=time.monotonic()
        with (out/'stdout.ptx').open('wb') as stdout, (out/'stderr.txt').open('wb') as stderr:
            child=subprocess.Popen(command,cwd=Path(__file__).resolve().parents[1],stdout=stdout,stderr=stderr,
                    env={'PATH':os.environ['PATH'],'HOME':'/nonexistent','PYTHONHASHSEED':'0',
                         'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1',
                         'EMMY_COMPILER_PARENT_PID':str(os.getpid())},
                    start_new_session=True)
            try:
                try: code=child.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    _kill_group(child.pid); child.wait(); code=124
            finally:
                # Successful compilers cannot leave background children running.
                _kill_group(child.pid)
        result={'command':command,'exit_code':code,'seconds':time.monotonic()-started,
                'isolation':'uid65534/no-new-privs/seccomp-network-and-ioctl-deny/no-group-escape/readonly-source-and-semantic-input',
                'source_ir':ir_name,'ptx_sha256':digest((out/'stdout.ptx').read_bytes())}
        (out/'compile.json').write_bytes(canonical(result))
        ptx=(out/'stdout.ptx').read_bytes()
        if code or not ptx or len(ptx)>2_000_000 or not ptx.isascii():
            raise ValueError(f'compiler rejected/invalid PTX ({code}): '+(out/'stderr.txt').read_text(errors='replace')[-2000:])
        return result
    finally:
        for folder,dirs,files in os.walk(sandbox):
            Path(folder).chmod(0o755)
        shutil.rmtree(sandbox)


def normalize_sass(text):
    # Comments contain offsets and encodings. Operands may contain semantic
    # hexadecimal immediates and must remain part of the instruction identity.
    lines=[]
    for line in text.splitlines():
        line=re.sub(r'/\*.*?\*/','',line)
        if ';' not in line: continue
        line=line.split(';')[0].strip()
        if line: lines.append(line)
    return '\n'.join(lines)


def resource_envelope(target, launch):
    """Optimistic occupancy bound, not a latency predictor or exact occupancy.

    Architectural limits for 8.9 and 12.x: CUDA Programming Guide, tables 30/31.
    Register allocation granules, reserved shared memory, barriers and compiler
    details can reduce occupancy further; query occupancy on the execution GPU.
    """
    if target['arch'] not in ('sm_89','sm_120'): raise ValueError('unknown resource target')
    threads=math.prod(launch['block'])
    registers=max(target.get('registers') or [0])
    static=max([int(n) for n in re.findall(r'SHARED:(\d+)',target.get('resources',''))] or [0])
    shared=static+launch.get('smem_bytes',0)
    warps=math.ceil(threads/32)
    limits=[24,48//warps]
    if registers: limits.append(65536//(registers*warps*32))
    if shared: limits.append((100*1024)//shared)
    blocks=max(0,min(limits))
    impossible=threads>1024 or registers>255 or registers*warps*32>65536 or shared>99*1024
    return {'arch':target['arch'],'threads':threads,'registers_per_thread':registers,
            'static_shared_bytes':static,'dynamic_shared_bytes':launch.get('smem_bytes',0),
            'blocks_per_sm_upper_bound':blocks,'warp_occupancy_upper_bound':min(1,blocks*warps/48),
            'architecturally_impossible':impossible,
            'source':'https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html',
            'limitation':'Optimistic bound, ignores allocation granules and reservations; resources are not speed predictions.'}


def assemble(ptx, cache, arch):
    ptx,cache=Path(ptx),Path(cache)
    tools={name:subprocess.check_output([name,'--version'],text=True) for name in ('ptxas','cuobjdump','nvdisasm')}
    key=digest(canonical({'ptx':digest(ptx.read_bytes()),'arch':arch,'flags':['-v'], 'tools':tools}))
    target=cache/key
    if (target/'assembly.json').exists():
        result=json.loads((target/'assembly.json').read_bytes())
        unsigned={k:v for k,v in result.items() if k!='metadata_sha256'}
        if result.get('metadata_sha256')!=digest(canonical(unsigned)) or result.get('key')!=key or result.get('arch')!=arch:
            raise ValueError('assembly cache metadata corrupted')
        for name,expected in result['hashes'].items():
            if Path(name).name!=name or digest((target/name).read_bytes())!=expected: raise ValueError('assembly cache corrupted')
        return {**result,'folder':str(target),'cache_hit':True}
    cache.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix='.assembly-',dir=cache))
    try:
        (temp/'kernel.ptx').write_bytes(ptx.read_bytes())
        command=['ptxas','-v','-arch='+arch,str(temp/'kernel.ptx'),'-o',str(temp/'module.cubin')]
        (temp/'attempt.json').write_bytes(canonical({'command':command,'arch':arch,'tools':tools,'ptx_sha256':digest(ptx.read_bytes())}))
        start=time.monotonic(); process=subprocess.run(command,capture_output=True,timeout=30)
        (temp/'ptxas.stdout').write_bytes(process.stdout); (temp/'ptxas.stderr').write_bytes(process.stderr)
        if process.returncode: raise ValueError('assembly failed: '+process.stderr.decode(errors='replace'))
        sass=subprocess.check_output(['cuobjdump','--dump-sass',str(temp/'module.cubin')],text=True,timeout=20)
        disasm=subprocess.check_output(['nvdisasm',str(temp/'module.cubin')],text=True,timeout=20)
        resources=subprocess.check_output(['cuobjdump','--dump-resource-usage',str(temp/'module.cubin')],text=True,timeout=20)
        (temp/'sass.txt').write_text(sass); (temp/'nvdisasm.txt').write_text(disasm)
        (temp/'sass.normalized.txt').write_text(normalize_sass(sass)); (temp/'resources.txt').write_text(resources)
        registers=[int(v) for v in re.findall(r'Used (\d+) registers',process.stderr.decode())]
        spills=re.findall(r'(\d+) bytes spill stores, (\d+) bytes spill loads',process.stderr.decode())
        result={'key':key,'arch':arch,'command':command,'exit_code':0,'seconds':time.monotonic()-start,
                'tools':tools,'registers':registers,'spills':spills,'resources':resources,
                'instruction_mix':{op:len(re.findall(r'\b'+op+r'\b',sass)) for op in ('LDG','STG','LDS','STS','SHFL','BAR','FFMA','FADD','IMAD','RED','ATOM')},
                'sass_sha256':digest(normalize_sass(sass).encode()),
                'hashes':{p.name:digest(p.read_bytes()) for p in temp.iterdir()}}
        result['metadata_sha256']=digest(canonical(result))
        (temp/'assembly.json').write_bytes(canonical(result))
        try: temp.rename(target)
        except OSError as error:
            if error.errno not in (errno.EEXIST,errno.ENOTEMPTY): raise
            shutil.rmtree(temp)
            return assemble(ptx,cache,arch)
        return {**result,'folder':str(target),'cache_hit':False}
    except Exception as error:
        # Keep the attempted command and scratch location even on assembler timeout.
        (temp/'failure.json').write_bytes(canonical({'error':str(error),'folder':str(temp)}))
        raise ValueError(f'assembly/disassembly failed; diagnostics retained at {temp}: {error}') from error


def prepare(source, case_dirs, out, cache, mechanism=None, baseline=None):
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for ci,case in enumerate(case_dirs):
        case=Path(case); validate_case(case)
        record=json.loads((case/'case.json').read_bytes()); modules=[]
        for ki,ir in enumerate(record['ir_paths']):
            folder=out/f'case-{ci}-kernel-{ki}'
            compiled=compile_ir(source,case,ir,folder)
            targets={arch:assemble(folder/'stdout.ptx',cache,arch) for arch in ('sm_89','sm_120')}
            launch=json.loads((case/'meta.json').read_bytes())['kernels'][ki]
            envelopes={arch:resource_envelope(target,launch) for arch,target in targets.items()}
            modules.append({'ir':ir,'compile':compiled,'targets':targets,'resource_envelopes':envelopes})
        rows.append({'case_dir':str(case.resolve()),'case_id':record['id'],'kernels':modules,
                     'launch_contract':json.loads((case/'meta.json').read_bytes())['kernels'],
                     'requires_sanitizer':record.get('requires_sanitizer',False),
                     'requires_isolated_first':record.get('requires_isolated_first',False)})
    activation=[]
    for rule in (mechanism or {}).get('required_sass',[]):
        arch=rule.get('arch','sm_89'); pattern=rule['pattern']
        if arch not in ('sm_89','sm_120'): raise ValueError('unknown activation target')
        case_ids=rule.get('case_ids'); entries=rule.get('entries')
        for selector,known in ((case_ids,{r['case_id'] for r in rows}),
                               (entries,{k['entry'] for r in rows for k in r['launch_contract']})):
            if selector is not None and (not isinstance(selector,list) or not selector or not set(selector)<=known):
                raise ValueError('unknown or empty activation scope')
        selected=[]
        for row in rows:
            for kernel,launch in zip(row['kernels'],row['launch_contract'],strict=True):
                if (case_ids is None or row['case_id'] in case_ids) and (entries is None or launch['entry'] in entries):
                    text=(Path(kernel['targets'][arch]['folder'])/'sass.normalized.txt').read_text()
                    selected.append({'case_id':row['case_id'],'entry':launch['entry'],
                                     'present':bool(re.search(pattern,text))})
        if not selected: raise ValueError('activation scope matches no kernels')
        mode=rule.get('match','all')
        if mode not in ('all','any'): raise ValueError('unknown activation match policy')
        present=all(x['present'] for x in selected) if mode=='all' else any(x['present'] for x in selected)
        activation.append({'arch':arch,'pattern':pattern,'present':present,'required':rule.get('required',True),
                           'match':mode,'kernels':selected})
    inactive=any(x['required'] and not x['present'] for x in activation)
    impossible=any(e['architecturally_impossible'] for r in rows for k in r['kernels'] for e in k.get('resource_envelopes',{}).values())
    identical=False
    if baseline:
        def runtime_identity(row):
            return (row['case_id'],row.get('launch_contract'),
                    [(k['ir'],k['targets']['sm_89']['sass_sha256'],k['targets']['sm_120']['sass_sha256']) for k in row['kernels']])
        identical=all('launch_contract' in r for r in baseline['cases']) and [runtime_identity(r) for r in rows]==[
                  runtime_identity(r) for r in baseline['cases']]
    result={'cases':rows,'activation':activation,'gate':'invalid' if impossible else ('inactive' if inactive else 'pass'),
            'runtime_identical_to_baseline':identical,
            'limitation':'registers/spills/resource deltas are evidence, not a performance verdict; launch-resource feasibility is checked on device'}
    (out/'prepared.json').write_bytes(canonical(result))
    return result


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--compiler-child':
        script, ir = sys.argv[2:4]
        os.chdir(Path(script).parent)
        _restrict()
        os.execv(sys.executable,[sys.executable,'-B',script,ir])
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--source',type=Path,required=True)
    p.add_argument('--cases',type=Path,nargs='+',required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True); args=p.parse_args()
    print(json.dumps(prepare(args.source,args.cases,args.out,args.cache),indent=2))
