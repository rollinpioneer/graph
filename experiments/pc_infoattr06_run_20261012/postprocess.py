"""Sequential final CPU processing after all four frozen workers exit."""
import os,pathlib,subprocess,time,json,resource
r=pathlib.Path(os.environ['R']);py=os.environ['PY'];out=[]
for name in ['chain_details.py','mechanism_review.py','audit.py','figures.py','censoring.py']:
 begin=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN)
 with (r/'logs'/('final_'+name+'.log')).open('w') as log:p=subprocess.run([py,'-u',str(r/name)],stdout=log,stderr=subprocess.STDOUT)
 after=resource.getrusage(resource.RUSAGE_CHILDREN);out.append({'script':name,'wall_seconds':time.monotonic()-begin,'child_user_CPU_s':after.ru_utime-before.ru_utime,'child_system_CPU_s':after.ru_stime-before.ru_stime,'exit_code':p.returncode})
 (r/'receipts/final_CPU_processing.json').write_text(json.dumps({'phases':out,'prior_interim_CPU_passes':'initial analysis; discovery chain reruns fixed calibration; discovery figures; interim mechanism check; no GPU calls. These phase clocks were not cumulatively collected. Final clocks measured here.'},indent=2))
 print('CPU_PHASE',name,p.returncode,flush=True)
 if p.returncode:raise SystemExit(p.returncode)
with (r/'logs/final_finalize.log').open('w') as log:subprocess.run([py,'-u',str(r/'finalize.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
print('POSTPROCESS_COMPLETE',flush=True)
