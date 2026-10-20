from common import *
import subprocess
while not (R/'results/evaluation_complete.json').exists():
 p=R/'results/pipeline_status.json'
 if p.exists() and json.load(open(p))['stage']=='TECHNICAL_STOP':raise RuntimeError('evaluation technical stop; preserve partial results')
 time.sleep(10)
subprocess.run([sys.executable,str(R/'code/report.py')],check=True)
write(R/'results/report_complete.json',dict(status='PASS',report_sha256=sha(R/'FINAL_REPORT.md'),C4_opened=False,next_experiment_started=False,Git_final_archive_pending=True))
