"""Replace only this round's monitor; preserve worker, samples, and phase/global caps."""
import pathlib,sys,os,signal,time,json,datetime,subprocess,xml.etree.ElementTree as ET,hashlib
ROOT=pathlib.Path(__file__).resolve().parent;name=sys.argv[1];parent=int(sys.argv[2]);out=ROOT/'runs'/name
assert name in ['supplement3','supplement6']
assert str(ROOT/'guard.py').encode() in pathlib.Path('/proc/%d/cmdline'%parent).read_bytes()
pgid=json.loads((out/'active.json').read_text())['pgid']
assert str(ROOT/'rollout.py').encode() in pathlib.Path('/proc/%d/cmdline'%pgid).read_bytes()
assert not (out/'guard_result.json').exists()
os.kill(parent,signal.SIGSTOP)
oldrows=[json.loads(x) for x in (out/'gpu_memory.jsonl').read_text().splitlines()]
lastrow=oldrows[-1];launch=json.loads((out/'launch.json').read_text());started=datetime.datetime.fromisoformat(launch['utc'])
now=time.monotonic();elapsed=(datetime.datetime.now(datetime.timezone.utc)-started).total_seconds();begin=now-elapsed
charge=lastrow['charged_gpu_seconds']+max(0,elapsed-lastrow['seconds'])*max(1,len(lastrow['active_gpu_uuids']));last=now
peak=max(x['task_memory_mib'] for x in oldrows);reason=None;errors=0;exitcode=None
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_supervisor_pid':parent,'worker_pgid':pgid,
         'original_per_job_reserved_seconds':800,'phase_A_hard_seconds_including_margins':2200,'global_hard_gpu_seconds':7200,
         'reason':'Cost-only amendment of suggested phase split: slower concurrent per-step recording; preserve two frozen 200-step episodes under hard7200 GPU seconds. No task worker restart. No worker restart, source/scene change, or outcome-based selection.',
         'new_supervisor_source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'previous_charge':charge}
(out/'monitor_takeover.json').write_text(json.dumps(receipt,indent=2))
def status(pid):
    try:
        x=pathlib.Path('/proc/%d/stat'%pid).read_text();v=x[x.rfind(')')+2:].split()
        return v[0],int(v[49]) if len(v)>49 else 0
    except FileNotFoundError:return 'gone',0
def members(groups):
    ans=set()
    for p in pathlib.Path('/proc').iterdir():
        if p.name.isdigit():
            try:
                if os.getpgid(int(p.name)) in groups and status(int(p.name))[0]!='Z':ans.add(int(p.name))
            except (ProcessLookupError,PermissionError):pass
    return ans
def phase_charge():
    total=0
    for p in (ROOT/'runs').glob('*'):
        if not (p.name.startswith('supplement') or p.name.startswith('record_')):continue
        if (p/'guard_result.json').exists():total+=json.loads((p/'guard_result.json').read_text())['charged_gpu_seconds']+2
        elif (p/'gpu_memory.jsonl').exists():
            lines=(p/'gpu_memory.jsonl').read_text().splitlines()
            if lines:total+=json.loads(lines[-1])['charged_gpu_seconds']+2
    return total
try:
    while True:
        st,word=status(pgid)
        if st in ['Z','gone']:
            exitcode=-(word&127) if word&127 else word>>8;break
        groups={json.loads(p.read_text())['pgid'] for p in (ROOT/'runs').glob('*/active.json') if not (p.parent/'guard_result.json').exists()}
        own=members({pgid});allpids=members(groups);active=set();mem=0;aggregate=0
        try:
            xml=ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],text=True,timeout=5))
            for g in xml.findall('gpu'):
                for p in g.findall('processes/process_info'):
                    pid=int(p.findtext('pid'));v=p.findtext('used_memory','0 MiB').split()[0]
                    if v.isdigit() and pid in allpids:aggregate+=int(v)
                    if pid in own:
                        if v.isdigit():mem+=int(v)
                        active.add(g.findtext('uuid'))
            errors=0
        except Exception:errors+=1
        now=time.monotonic();charge+=(now-last)*max(1,len(active));last=now;peak=max(peak,mem)
        row={'seconds':now-begin,'task_memory_mib':mem,'aggregate_task_memory_mib':aggregate,'charged_gpu_seconds':charge,
             'pids':sorted(own),'active_gpu_uuids':sorted(active),'monitor':'takeover_guard'}
        with (out/'gpu_memory.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if aggregate>=11264:reason='memory_safety_stop'
        if phase_charge()>=2185:reason='supplement_phase_time_safety_stop'
        global_used=0
        for folder in (ROOT/'runs').iterdir():
            f=folder/'guard_result.json';log=folder/'gpu_memory.jsonl'
            if f.exists():global_used+=json.loads(f.read_text())['charged_gpu_seconds']+2
            elif log.exists():
                lines=log.read_text().splitlines()
                if lines:global_used+=json.loads(lines[-1])['charged_gpu_seconds']+2
        if global_used>=6985:reason='global_time_safety_stop'
        if errors>=3:reason='telemetry_unavailable'
        if reason:
            os.killpg(pgid,signal.SIGTERM);time.sleep(2)
            if status(pgid)[0] not in ['Z','gone']:os.killpg(pgid,signal.SIGKILL)
        time.sleep(.5)
finally:
    if status(pgid)[0] not in ['Z','gone']:
        os.killpg(pgid,signal.SIGTERM);time.sleep(2)
        if status(pgid)[0] not in ['Z','gone']:os.killpg(pgid,signal.SIGKILL)
        reason=reason or 'monitor_exception'
    st,word=status(pgid)
    if exitcode is None:exitcode=-(word&127) if word&127 else word>>8
    # The stopped old monitor owns the exited worker. Terminate only that monitor;
    # init then reaps the worker. Never signal another experiment or user.
    os.kill(parent,signal.SIGKILL)
    now=time.monotonic();charge+=(now-last)*max(1,len(lastrow['active_gpu_uuids']))
    result={'start_utc':launch['utc'],'end_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'wall_seconds':now-begin,
            'charged_gpu_seconds':charge,'task_peak_mib':peak,'exit_code':exitcode,'stop_reason':reason,'gpu':launch['gpu'],'pid':pgid,
            'accounting':'Continuous pre/post-monitor charge, full wall time multiplied by task-resident GPUs; 2s per-job margin added by final analyzer.',
            'monitor_takeover':receipt}
    (out/'guard_result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
