from pathlib import Path
import subprocess,sys,json,time,os
R=Path(__file__).resolve().parents[1];seed=sys.argv[1];start=time.time()
p=subprocess.Popen([sys.executable,str(R/'code/probe.py'),seed],cwd=R)
rc=p.wait()
(R/'receipts'/f'exit_s{seed}.json').write_text(json.dumps(dict(pid=p.pid,wrapper_pid=os.getpid(),exit_code=rc,start=start,end=time.time()),indent=2))
sys.exit(rc)
