from core import *
import subprocess
while True:
 p=R/'results/pipeline_status.json'
 if p.exists():
  status=json.loads(p.read_text())
  if status['status']=='COMPLETED_AT_S1_STOP':break
  if status['status'] in ['TECHNICAL_STOP','READY_FOR_LOCKED_CONFIRMATION']:raise RuntimeError(status)
 time.sleep(15)
for name in ['action_diagnostic.py','finish_outputs.py','final_report.py']:
 subprocess.run([sys.executable,str(R/'code'/name)],check=True)
write(R/'results/postprocess.json',dict(status='PASS',scripts=['action_diagnostic.py','finish_outputs.py','final_report.py']))
print('POSTPROCESS_PASS',flush=True)
