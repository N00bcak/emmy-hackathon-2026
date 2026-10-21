#!/usr/bin/env python3
"""Exercise the local API inside one admitted GPU job; never calls the public runner."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--synthetic-dir',type=Path)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    log=(args.out/'server.log').open('w')
    server=subprocess.Popen([sys.executable,str(ROOT/'tools/local_runner.py'),'--cache',str(args.out/'cache'),'--port','48703'],stdout=log,stderr=subprocess.STDOUT)
    base='http://127.0.0.1:48703'
    results=[]
    try:
        for _ in range(100):
            if server.poll() is not None: raise RuntimeError('local server exited')
            try:
                with urllib.request.urlopen(base+'/health',timeout=1): pass
                break
            except OSError: time.sleep(.1)
        else: raise RuntimeError('local server startup timed out')
        command=[sys.executable,str(ROOT/'tools/probe_endpoint.py'),'--server',base,'--compiler',str(args.compiler),'--out',str(args.out/'contract')]
        result=subprocess.run(command,timeout=180)
        results.append({'command':command,'exit_code':result.returncode})
        if args.synthetic_dir:
            for folder in sorted(args.synthetic_dir.iterdir()):
                if not (folder/'expected.npz').is_file(): continue
                command=[sys.executable,str(ROOT/'emmy-hackathon-2026/submit.py'),str(folder),'--compiler',str(args.compiler),'--server',base]
                result=subprocess.run(command,capture_output=True,text=True,timeout=60)
                (args.out/f'{folder.name}.log').write_text(result.stdout+result.stderr)
                results.append({'command':command,'exit_code':result.returncode})
                print(folder.name,'PASS' if result.returncode==0 else 'FAIL',flush=True)
        (args.out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        return int(any(result['exit_code'] for result in results))
    finally:
        server.terminate()
        try: server.wait(timeout=5)
        except subprocess.TimeoutExpired: server.kill();server.wait()
        log.close()
if __name__=='__main__': raise SystemExit(main())
