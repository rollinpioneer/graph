"""External supervisor: only terminates its own process group; GPU time includes setup."""
import argparse, pathlib, json, time, subprocess, signal, os, xml.etree.ElementTree as ET, datetime
import shutil,hashlib
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seconds',type=float,required=True);p.add_argument('--gpu',required=True);p.add_argument('cmd',nargs=argparse.REMAINDER);a=p.parse_args()
out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=False)
root=pathlib.Path(__file__).resolve().parent
for name in ['evaluate.py','guard.py','run_job.sh','launch.py','RESOURCE_GATE.md','CORE_CARD.md']:
 if (root/name).exists():shutil.copy2(root/name,out/name)
used=sum(json.loads(f.read_text()).get('charged_gpu_seconds',0) for f in (root/'runs').glob('*/guard_result.json')) if (root/'runs').exists() else 0
reserved=sum(json.loads(f.read_text()).get('max_seconds',0) for f in (root/'runs').glob('*/launch.json') if not (f.parent/'guard_result.json').exists())
remaining=7000-used-reserved
assert remaining>0,('GPU budget exhausted',used)
limit=min(a.seconds,remaining)
begin=time.monotonic();utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
env=os.environ.copy();env.pop('CUDA_VISIBLE_DEVICES',None);env['CUDA_DEVICE_ORDER']='PCI_BUS_ID';env['STAGE5_GPU_INDEX']=a.gpu
cmd=a.cmd[1:] if a.cmd and a.cmd[0]=='--' else a.cmd
(out/'launch.json').write_text(json.dumps({'utc':utc,'cmd':cmd,'gpu':a.gpu,'max_seconds':limit,'prior_gpu_seconds':used,'memory_stop_mib':11264},indent=2))
log=(out/'process.log').open('w');child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
(out/'active.json').write_text(json.dumps({'pgid':child.pid}))
peak=0;charge=0;last=begin;reason=None;telemetry_errors=0
def members():
 ans=set()
 for x in pathlib.Path('/proc').iterdir():
  if x.name.isdigit():
   try:
    if os.getpgid(int(x.name))==child.pid:ans.add(int(x.name))
   except ProcessLookupError:pass
 return ans
def all_members():
 groups={child.pid}
 for f in (root/'runs').glob('*/active.json'):
  if not (f.parent/'guard_result.json').exists():
   groups.add(json.loads(f.read_text())['pgid'])
 ans=set()
 for x in pathlib.Path('/proc').iterdir():
  if x.name.isdigit():
   try:
    if os.getpgid(int(x.name)) in groups:ans.add(int(x.name))
   except (ProcessLookupError,PermissionError):pass
 return ans
try:
 while child.poll() is None:
  now=time.monotonic();pids=members();allpids=all_members();active=set();mem=0;aggregate=0;entries=[]
  try:
   raw=subprocess.check_output(['nvidia-smi','-q','-x'],text=True,timeout=5)
   for gpu in ET.fromstring(raw).findall('gpu'):
    uuid=gpu.findtext('uuid');total=gpu.findtext('fb_memory_usage/used');util=gpu.findtext('utilization/gpu_util')
    entries.append({'uuid':uuid,'all_memory':total,'all_utilization':util})
    for q in gpu.findall('processes/process_info'):
     val=q.findtext('used_memory','0 MiB').split()[0]
     if int(q.findtext('pid')) in allpids and val.isdigit():aggregate+=int(val)
     if int(q.findtext('pid')) in pids:
      val=q.findtext('used_memory','0 MiB').split()[0]
      if val.isdigit():mem+=int(val)
      active.add(uuid)
   telemetry_errors=0
  except Exception as e:
   telemetry_errors+=1;entries=[{'error':str(e)}]
  charge+=(now-last)*max(1,len(active));last=now;peak=max(peak,mem)
  row={'seconds':now-begin,'task_memory_mib':mem,'aggregate_task_memory_mib':aggregate,'charged_gpu_seconds':charge,'pids':sorted(pids),'active_gpu_uuids':sorted(active),'background':entries}
  with (out/'gpu_memory.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  if aggregate>=11264:reason='memory_safety_stop'
  if charge>=limit-5:reason='time_safety_stop'
  if telemetry_errors>=3:reason='telemetry_unavailable'
  if reason:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=3)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
   break
  time.sleep(.5)
finally:
 if child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=3)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
 child.wait();now=time.monotonic();charge+=now-last
 log.close()
 result={'start_utc':utc,'end_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'wall_seconds':now-begin,'charged_gpu_seconds':charge,'total_gpu_seconds':used+charge,'task_peak_mib':peak,'exit_code':child.returncode,'stop_reason':reason,'gpu':a.gpu,'pid':child.pid,'accounting':'conservative full child-process wall time, multiplied by GPUs on which task memory appears'}
 (out/'guard_result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
