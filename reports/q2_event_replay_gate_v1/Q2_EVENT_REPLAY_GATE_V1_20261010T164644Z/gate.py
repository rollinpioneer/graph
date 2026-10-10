"""Authorized G1/G2 only. Controlled CPU/FP32/one-state scoring; no intervention."""
import copy
import gzip
import hashlib
import heapq
import io
import json
import math
import os
import pickle
import resource
import struct
import sys
import time
from pathlib import Path

ROOT = Path(sys.argv[1])
REPO = Path('/home/xushijie3/work/graph_cp_disr')
REG = json.loads((ROOT / 'registration.json').read_text())
START = time.process_time()
WALL = time.perf_counter()
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO / 'src'))
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
torch.set_float32_matmul_precision('highest')
from cp_disr.pddl import depots as DP, task as T
from cp_disr.pddl.compact import cli_ext as CE
from cp_disr.pddl.search_match.budget import Budget
from cp_disr.pddl.search_match.engine import SuccessorIndex

COUNTS = {'model_state_evaluations': 0, 'search_prefix_and_restore_calls': 0, 'failed_model_calls': 0, 'repairs': 0, 'GPU_device_seconds': 0}
TASKS, INDEX, EV = {}, {}, {}
PROTOCOL = REG['scoring_protocol']['id']
CANDIDATES, SELECTED, G2 = [], [], []
SEARCH_STOP = None
(ROOT / 'snapshots').mkdir(exist_ok=True)
(ROOT / 'decisions').mkdir(exist_ok=True)
LEDGER = (ROOT / 'resource_ledger.jsonl').open('a', buffering=1)
RAW = (ROOT / 'score_bytes.bin').open('ab', buffering=0)
CHECKS = Budget(7000, 10**9, 8*2**30, 20)
EVAL_BUDGET = 8192


class Stop(Exception):
    pass


def emit(kind, **data):
    LEDGER.write(json.dumps({'kind': kind, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **data}) + '\n')


def save(name, obj):
    (ROOT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def guard(n=0):
    if COUNTS['model_state_evaluations'] + n > EVAL_BUDGET:
        raise Stop('MODEL_STATE_EVALUATION_LIMIT')
    if time.process_time()-START+REG['CPU_ancillary_reserve_seconds'] >= 7200:
        raise Stop('CPU_LIMIT')
    if time.perf_counter()-WALL > 7000:
        raise Stop('WALL_GUARD')


def bits(value):
    return struct.pack('<f', value).hex()


def value_from_bits(h):
    return struct.unpack('<f', bytes.fromhex(h))[0]


def load_task(cid):
    if cid not in TASKS:
        info = REG['tasks'][cid]
        assert hashlib.sha256(Path(info['file']).read_bytes()).hexdigest() == info['sha256']
        domain = DP.DOMAIN_IPC if cid.startswith('ipc') else DP.DOMAIN_TYPED
        task = T.depots_task(domain, info['file'])
        TASKS[cid], INDEX[cid] = task, SuccessorIndex(task)
    return TASKS[cid], INDEX[cid]


def evaluator(cid, model):
    if (cid, model) not in EV:
        guard()
        emit('MODEL_LOAD_START', task=cid, model=model)
        v2 = REPO / 'runs/final_master/c1_route_b/compact_diagnosis_v2/20261009T165825Z'
        ev = CE.make_evaluator(str(v2), model, 'cpu')
        ev.model.eval()
        for p in ev.model.parameters():
            assert p.dtype == torch.float32 and p.device.type == 'cpu'
            p.requires_grad_(False)
        task, _ = load_task(cid)
        ev.prepare(task, {'domain': str(DP.DOMAIN_IPC if cid.startswith('ipc') else DP.DOMAIN_TYPED), 'problem': REG['tasks'][cid]['file']}, CHECKS)
        EV[cid, model] = ev
        emit('MODEL_LOAD_END', task=cid, model=model)
    return EV[cid, model]


def raw_score(cid, model, s, reason):
    guard(1)
    ev = evaluator(cid, model)
    COUNTS['model_state_evaluations'] += 1
    seq = COUNTS['model_state_evaluations']
    emit('MODEL_STATE_START', sequence=seq, task=cid, state=format(s, 'x'), model=model, reason=reason, input_graphs=1)
    try:
        with torch.inference_mode():
            v = ev.evaluate([s], CHECKS)[0]
        assert math.isfinite(v) and value_from_bits(bits(v)) == v
        RAW.write(struct.pack('<f', v))
        emit('MODEL_STATE_END', sequence=seq, status='OK', value_bits_le=bits(v), byte_offset=(seq-1)*4)
        return v
    except Exception as err:
        COUNTS['failed_model_calls'] += 1
        RAW.write(bytes(4))
        emit('MODEL_STATE_END', sequence=seq, status='FAILED', error=repr(err))
        raise


def search_score(st, states):
    vals = []
    for s in states:
        key = format(s, 'x')
        if key in st['score_cache']:
            st['score_cache_hits'] += 1
            emit('CACHE_HIT', task=st['task'], model='D0@820', state=key, protocol=PROTOCOL)
        else:
            st['score_cache_misses'] += 1
            emit('CACHE_MISS', task=st['task'], model='D0@820', state=key, protocol=PROTOCOL)
            st['score_cache'][key] = bits(raw_score(st['task'], 'D0@820', s, 'search_cache_miss'))
        vals.append(value_from_bits(st['score_cache'][key]))
    st['calls'] += 1
    return vals


def blank(cid):
    task, _ = load_task(cid)
    return {'task': cid, 'task_sha256': REG['tasks'][cid]['sha256'], 'domain_sha256': REG['tasks'][cid]['domain_sha256'], 'scoring_protocol': PROTOCOL, 'model_sha256': REG['models']['D0@820']['sha256_actual'], 'heap': [], 'closed': set(), 'node': {task.init_mask:(0,None,None)}, 'serial':0, 'expanded':0, 'generated':0, 'duplicates':0, 'path_updates':0, 'calls':0, 'best_h':None, 'h_root':None, 'score_cache':{}, 'score_cache_hits':0, 'score_cache_misses':0, 'pending_scores':[], 'status':'RUNNING', 'plan':None}


def normalized(st):
    obj = copy.deepcopy(st)
    obj['heap'] = [[bits(h), serial, format(s,'x'), s not in st['closed']] for h,serial,s in st['heap']]
    obj['closed'] = [format(s,'x') for s in sorted(st['closed'])]
    obj['node'] = [[format(s,'x'), g, None if p is None else format(p,'x'), a] for s,(g,p,a) in sorted(st['node'].items())]
    return obj


def restored(obj):
    st = copy.deepcopy(obj)
    st['heap'] = [(value_from_bits(h), ser, int(s,16)) for h,ser,s,valid in obj['heap']]
    st['closed'] = {int(s,16) for s in obj['closed']}
    st['node'] = {int(s,16):(g,None if p is None else int(p,16),a) for s,g,p,a in obj['node']}
    assert normalized(st) == obj
    assert not st['pending_scores']
    return st


def digest(st):
    return hashlib.sha256(json.dumps(normalized(st), sort_keys=True, separators=(',',':')).encode()).hexdigest()


def step(st):
    task, index = load_task(st['task'])
    while st['heap']:
        h, serial, s = heapq.heappop(st['heap'])
        if s not in st['closed']:
            break
    else:
        st['status'] = 'OPEN_EXHAUSTED'
        return None
    st['closed'].add(s)
    st['expanded'] += 1
    g = st['node'][s][0]+1
    pending = []
    decision = {'expanded':st['expanded'], 'state':format(s,'x'), 'score_bits':bits(h), 'serial':serial}
    for a in index.applicable(s):
        t=(s & ~a.del_mask)|a.add_mask
        st['generated'] += 1
        if task.solvable_by_relaxation and t & task.goal_mask == task.goal_mask:
            st['node'][t]=(g,s,a.index)
            acts=[]
            p=t
            while st['node'][p][1] is not None:
                _,p0,act=st['node'][p]
                acts.append(act); p=p0
            st['plan']=list(reversed(acts))
            st['status']='SOLVED'
            return decision
        if t in st['closed']:
            st['duplicates']+=1
            continue
        old=st['node'].get(t)
        if old is None:
            st['node'][t]=(g,s,a.index);pending.append(t)
        elif g<old[0]:
            st['node'][t]=(g,s,a.index);st['path_updates']+=1
        else:
            st['duplicates']+=1
    if pending:
        vals=search_score(st,pending)
        for t,v in zip(pending,vals):
            heapq.heappush(st['heap'],(v,st['serial'],t));st['serial']+=1
            if v<st['best_h']:st['best_h']=v
    return decision


def choose(entries, values):
    ordered=sorted((v,e[1],e[2]) for e,v in zip(entries,values))
    first=ordered[0]
    return {'state':format(first[2],'x'), 'serial':first[1], 'score_bits':bits(first[0]), 'minimum_ties':sum(v==first[0] for v,_,_ in ordered), 'top_second_gap':ordered[1][0]-first[0] if len(ordered)>1 else None}


DIAG_CACHE = {}


def screen(st, origin, force_three=False):
    cid=st['task'];entries=sorted(e for e in st['heap'] if e[2] not in st['closed'])
    if len(entries)<2:return None
    proposal=[]
    for model in REG['model_order']:
        vals=[]
        for _,ser,s in entries:
            key=(cid,model,s)
            if model=='D0@820' and format(s,'x') in st['score_cache']:
                v=value_from_bits(st['score_cache'][format(s,'x')])
            elif key in DIAG_CACHE:v=DIAG_CACHE[key]
            elif force_three:
                v=None
            else:
                v=raw_score(cid,model,s,'full_OPEN_prescreen');DIAG_CACHE[key]=v
            vals.append(v)
        if force_three:proposal.append(None)
        else:proposal.append(choose(entries,vals))
    main = force_three or (proposal[0]['state']==proposal[1]['state']!=proposal[2]['state'])
    row={'task':cid,'k':st['expanded'],'origin':origin,'valid_OPEN':len(entries),'prescreen':proposal,'status':'NO_MAIN_DISAGREEMENT'}
    CANDIDATES.append(row)
    if not main:return None
    if COUNTS['model_state_evaluations']+9*len(entries)>8192:
        row['status']='SKIPPED_COMPLETE_OPEN_REPEAT_BUDGET';return None
    repetitions=[]
    for rep in range(3):
        order=list(reversed(entries)) if rep==2 else entries
        choices={}
        for model in REG['model_order']:
            vals=[raw_score(cid,model,e[2], 'independent_OPEN_repeat_%d'%rep) for e in order]
            choices[model]=choose(order,vals)
        repetitions.append(choices)
    row['independent_repetitions']=repetitions
    states=[[r[m]['state'] for m in REG['model_order']] for r in repetitions]
    stable=states[0]==states[1]==states[2]
    relation=all(s[0]==s[1]!=s[2] for s in states)
    ties=any(r[m]['minimum_ties']!=1 for r in repetitions for m in REG['model_order'])
    row['status']='TIE_DEPENDENT_DISAGREEMENT' if stable and relation and ties else ('STABLE_MAIN_EVENT' if stable and relation else 'PROTOCOL_SENSITIVE_OR_UNSTABLE_EVENT')
    if row['status']!='STABLE_MAIN_EVENT':return None
    obj=normalized(st)
    name='%s_k%03d_%s'%(cid,st['expanded'],origin)
    save('snapshots/'+name+'.json',obj)
    row['snapshot']='snapshots/'+name+'.json';row['snapshot_sha256']=digest(st)
    row['snapshot_unchanged_by_observer']=True
    return {'row':row, 'snapshot':obj}


def prefix(cid, limit):
    guard()
    if COUNTS['search_prefix_and_restore_calls']>=6:raise Stop('PREFIX_RESTORE_CALL_LIMIT')
    COUNTS['search_prefix_and_restore_calls']+=1
    emit('SEARCH_START', call=COUNTS['search_prefix_and_restore_calls'], task=cid, role='qualification_prefix', expansion_cap=limit, intervention=False)
    st=blank(cid);task,_=load_task(cid)
    h=search_score(st,[task.init_mask])[0];st['best_h']=st['h_root']=h
    heapq.heappush(st['heap'],(h,0,task.init_mask));st['serial']=1
    candidate=None;records=[]
    path=ROOT/'decisions'/('%s_prefix.jsonl.gz'%cid)
    with gzip.open(path,'wt') as f:
        while st['status']=='RUNNING' and st['heap'] and st['expanded']<limit:
            guard()
            if candidate is None:
                got=screen(st,'fresh_prefix')
                if got:candidate=got
            before=digest(st);decision=step(st)
            after=normalized(st)
            rec={'before_sha256':before,'decision':decision,'after':after}
            records.append(rec);f.write(json.dumps(rec)+'\n')
    if st['status']=='RUNNING':st['status']='PREFIX_NODE_LIMIT'
    emit('SEARCH_END', task=cid, status=st['status'], expanded=st['expanded'], generated=st['generated'])
    return candidate,records,st


def replay(candidate, baseline=None):
    if COUNTS['search_prefix_and_restore_calls']>=6:raise Stop('PREFIX_RESTORE_CALL_LIMIT')
    row=candidate['row'];obj=candidate['snapshot'];cid=row['task']
    st=restored(obj)
    COUNTS['search_prefix_and_restore_calls']+=1
    emit('SEARCH_START', call=COUNTS['search_prefix_and_restore_calls'], task=cid, role='no_intervention_restore', expansion_cap=64 if cid.startswith('ipc') else 32, intervention=False)
    records=[];limit=64 if cid.startswith('ipc') else 32
    suffix = None if baseline is None else [r for r in baseline if r['decision'] and r['decision']['expanded']>obj['expanded']]
    if suffix is not None:limit=min(limit,len(suffix))
    for _ in range(limit):
        if st['status']!='RUNNING' or not st['heap']:break
        guard();before=digest(st);decision=step(st);records.append({'before_sha256':before,'decision':decision,'after':normalized(st)})
    path='decisions/%s_k%03d_restore%d.jsonl.gz'%(cid,obj['expanded'],COUNTS['search_prefix_and_restore_calls'])
    with gzip.open(ROOT/path,'wt') as f:
        for r in records:f.write(json.dumps(r)+'\n')
    emit('SEARCH_END', task=cid, role='no_intervention_restore', expanded_after=st['expanded'], steps=len(records), status=st['status'])
    return records


def compare(candidate, a, b):
    n=min(len(a),len(b),64)
    same=n>0 and all(a[i]==b[i] for i in range(n)) and len(a)==len(b)
    # Equality checks actual full content, not just hashes or CLOSED/Jaccard.
    obj={'task':candidate['row']['task'],'k':candidate['row']['k'],'next_pop_identical':bool(n and a[0]['decision']==b[0]['decision']), 'steps_compared':n, 'all_full_state_queue_parent_cache_counter_content_equal':same,'first_divergence':next((i for i in range(n) if a[i]!=b[i]),None),'status':'PASS' if same else 'TECHNICALLY_INVALID'}
    G2.append(obj)
    candidate['row']['G2']=obj
    return same


def main():
    emit('STAGE_START', scope='G1_G2', protocol=PROTOCOL, phase_C_authorized=False)
    # The historical event is checked before any new prefix; its logical state is retained.
    oldpath=REG['old_p12_snapshot']
    class Restricted(pickle.Unpickler):
        def find_class(self,module,name):raise pickle.UnpicklingError('nonprimitive snapshot')
    old=Restricted(io.BytesIO(Path(oldpath).read_bytes())).load()
    seed=blank('ipc_p12')
    seed.update(old)
    for h,ser,s in seed['heap']:seed['score_cache'][format(s,'x')]=bits(h)
    # Original OPEN values are replaced only in a declared controlled-protocol copy.
    original_top=min(e for e in seed['heap'] if e[2] not in seed['closed'])[2]
    changed=[]
    for h,ser,s in seed['heap']:
        v=raw_score('ipc_p12','D0@820',s,'controlled_historical_OPEN_revalue')
        changed.append((v,ser,s));seed['score_cache'][format(s,'x')]=bits(v)
    seed['heap']=changed;heapq.heapify(seed['heap'])
    oldcandidate=screen(seed,'historical_k016',force_three=True)
    save('old_p12_protocol_comparison.json',{'original_next':format(original_top,'x'),'controlled_next':format(min(seed['heap'])[2],'x'),'logical_original_snapshot_retained':True,'queue_revalued':True,'event_survives_controlled_protocol':oldcandidate is not None,'qualification':CANDIDATES[-1]})
    struct_results=[]
    for cid in REG['struct_task_order']:
        c,records,final=prefix(cid,32)
        struct_results.append((c,records))
        save(cid+'_prefix_summary.json',{'status':final['status'],'expanded':final['expanded'],'candidate':None if c is None else c['row']})
    # Earliest p12 event is checked from its root rather than assuming k16 was earliest.
    p12c,p12records,p12final=prefix('ipc_p12',64)
    save('ipc_p12_prefix_summary.json',{'status':p12final['status'],'expanded':p12final['expanded'],'candidate':None if p12c is None else p12c['row']})
    if p12c:
        suffix=[r for r in p12records if r['decision'] and r['decision']['expanded']>p12c['snapshot']['expanded']]
        if compare(p12c,suffix,replay(p12c,p12records)):SELECTED.append(p12c['row'])
    elif oldcandidate:
        # This fallback stays exploratory; earliest status cannot be inferred if the new prefix had none.
        oldcandidate['row']['not_selected_reason']='not observed in canonical root prefix; historical protocol sensitivity remains separate'
    for c,records in struct_results:
        if c:
            suffix=[r for r in records if r['decision'] and r['decision']['expanded']>c['snapshot']['expanded']]
            if compare(c,suffix,replay(c,records)):SELECTED.append(c['row'])
            break


try:
    resource.setrlimit(resource.RLIMIT_CPU,(7000,7100))
    main()
except Stop as err:
    SEARCH_STOP=str(err);emit('STOP',reason=SEARCH_STOP)
except Exception as err:
    SEARCH_STOP='TECHNICAL_ERROR:'+repr(err);emit('ERROR',reason=SEARCH_STOP)
finally:
    cpu=time.process_time()-START
    save('candidates.json',CANDIDATES)
    save('selected_events.json',SELECTED)
    save('G2_replay_checks.json',G2)
    save('gate_receipt.json',{'status':'ELIGIBLE_AWAITING_PHASE_C_APPROVAL' if SELECTED else ('STOPPED_AT_BUDGET' if SEARCH_STOP and 'LIMIT' in SEARCH_STOP else 'NO_ELIGIBLE_EVENT' if not SEARCH_STOP else 'TECHNICALLY_INVALID'),'stop_reason':SEARCH_STOP,'counts':COUNTS,'cpu_seconds':cpu,'wall_seconds':time.perf_counter()-WALL,'GPU_device_seconds':0,'phase_C_runs':0,'selected_event_count':len(SELECTED),'training':0,'certification':0,'graph_builds':0,'scoring_protocol':PROTOCOL})
    emit('STAGE_END',cpu_seconds=cpu,counts=COUNTS,phase_C_runs=0)
    RAW.close();LEDGER.close()
    print(json.dumps(json.loads((ROOT/'gate_receipt.json').read_text()),ensure_ascii=False),flush=True)
