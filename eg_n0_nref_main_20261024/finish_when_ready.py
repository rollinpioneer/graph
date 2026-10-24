import sys,time,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent;start=time.time()
while len(list((R/'dense').glob('*.json')))<48:
 if time.time()-start>1800:raise TimeoutError('dense audit incomplete')
 time.sleep(5)
subprocess.run([sys.executable,str(R/'analyze.py')],check=True,cwd=R)
subprocess.run([sys.executable,str(R/'finalize.py')],check=True,cwd=R)
