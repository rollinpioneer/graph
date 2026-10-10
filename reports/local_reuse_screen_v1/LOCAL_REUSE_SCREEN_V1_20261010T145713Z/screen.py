"""Bounded, zero-training LOCAL-REUSE-SCREEN-V1. Run only on lab-xushijie3.

Preparation freezes existing inputs before measurement. No exact oracle,
dataset generation, training, dependency installation, or legacy output writes.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, pathlib, platform, resource, subprocess, time
from collections import defaultdict

REPO = pathlib.Path('/home/xushijie3/work/graph_cp_disr')
OLD = REPO / 'runs/final_master/c1_route_b/public_depots_rel_v1/20261008T063544Z_9898106e'
SCOPE = REPO / 'runs/final_master/c1_route_b/search_scope_diagnostic_v1/20261009T063905Z_511b117'
LIB = REPO / 'runs/final_master/c1_route_b/compact_diagnosis_v2/20261009T165825Z/diagnostics/packages.json'
BASE = 'cc7d79d6fcb053bf0c16033e765a2e808de2aaac'
FAST_COMMIT = '511b117495920234cb646016eaf504c7371233aa'
BRANCH = 'codex/cp-disr-local-reuse-screen-v1'

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1048576), b''): h.update(part)
    return h.hexdigest()

def sj(path, value):
    pathlib.Path(path).write_text(json.dumps(value, indent=2, default=str) + '\n')

def cmd(*args):
    return subprocess.check_output(args, cwd=REPO, text=True)

def state_sha(s):
    return hashlib.sha256(format(s, 'x').encode()).hexdigest()

def csv_write(path, rows, fields=None):
    rows = list(rows)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader(); w.writerows(rows)

def prepare(out):
    import torch, torch_geometric
    from cp_disr.pddl.search_match.runner import OldRun, load_neural
    from cp_disr.pddl.task import depots_task
    from cp_disr.pddl.fast_alt.fast_value import FastValue
    import cp_disr.pddl.depots as DP
    start = time.monotonic()
    old = OldRun(OLD)
    m, ck = load_neural(old, 'V_DENSE', torch.device('cpu'))
    assert m.mode == 'dense' and not m.training and not any(p.requires_grad for p in m.parameters())
    packages = json.loads(LIB.read_text())
    joint_ids = []
    for size in (6, 8):
        ids = [k for k in packages if k.startswith('joint_n%d' % size)]
        joint_ids.append(min(ids, key=lambda k: hashlib.sha256(k.encode()).hexdigest()))
    ids = joint_ids + ['ipc_p06', 'ipc_p11', 'ipc_p12', 'ipc_p14', 'ipc_p15', 'ipc_p17']
    all_cases = {c['case_id']: dict(c, set=family) for family in ('joint', 'ipc') for c in old.cases(family)}
    inputs = {str(LIB): sha(LIB), str(OLD/'data/manifest.json'): sha(OLD/'data/manifest.json'), str(ck['path']): sha(ck['path'])}
    tasks = []; parents = []; group_count = 0
    for cid in ids:
        c = all_cases[cid]; domain = DP.DOMAIN_IPC if c['set'] == 'ipc' else DP.DOMAIN_TYPED
        assert sha(c['file']) == c['sha256']
        task = depots_task(domain, c['file']); tpl = task.template()
        if c['set'] == 'ipc':
            source = SCOPE / ('states/local_%s.json' % cid)
            data = json.loads(source.read_text())
            available = {int(x['parent'], 16) for x in data}
        else:
            source = LIB
            available = {int(x['parent'], 16) for x in packages[cid]['packages']}
        inputs[str(source)] = sha(source); inputs[str(domain)] = sha(domain); inputs[c['file']] = sha(c['file'])
        chosen = sorted(available, key=state_sha)[:16]
        fv = FastValue(m, torch.device('cpu')); fv.prepare(task, tpl)
        for s in chosen:
            legal = task.legal(s)
            assert [a.aid for a in legal] == sorted(a.aid for a in legal)
            transitions = [dict(action_index=a.index, action_id=a.aid, state=format(task.apply(s,a),'x'), state_sha256=state_sha(task.apply(s,a))) for a in legal[:16]]
            parents.append(dict(case_id=cid, parent=format(s,'x'), parent_sha256=state_sha(s), source=str(source), source_role='DEVELOPMENT_DIAGNOSTIC', full_legal_successors=len(legal), truncated=len(legal)>16, successors=transitions))
            group_count += len(transitions)
        tasks.append(dict(c, domain=str(domain), domain_sha256=sha(domain), nodes=fv.N, edges=fv.E, goals=len(tpl.goals), dyn_atoms=len(task.dyn_atoms), parent_count=len(chosen), requested_category=('same_family_complex_goals' if c['set']=='joint' else ('historical_node_guidance_disadvantage' if cid in ['ipc_p06','ipc_p11','ipc_p12','ipc_p17'] else 'historically_solved_ipc'))))
        print('PREPARED',cid,fv.N,fv.E,len(chosen),flush=True)
    assert len(tasks)<=8 and len(parents)<=128 and group_count<=2048
    ordered = sorted(tasks,key=lambda x:(x['edges'],x['nodes'],x['case_id']))
    for i, c in enumerate(ordered): c['size_stratum'] = ('small' if i<3 else 'medium' if i<6 else 'large')
    sources = {str(p.relative_to(REPO)):sha(p) for p in (REPO/'src/cp_disr').rglob('*.py')}
    core = ['src/cp_disr/neural.py','src/cp_disr/c1_blocksworld_policies.py','src/cp_disr/blocksworld/method_serial/model.py','src/cp_disr/blocksworld/gp_attribution.py','src/cp_disr/pddl/model.py','src/cp_disr/pddl/task.py','src/cp_disr/pddl/fast_alt/fast_value.py','src/cp_disr/pddl/search_match/engine.py','src/cp_disr/pddl/search_match/evaluators.py']
    manifest = dict(card='CP-DISR-LOCAL-REUSE-SCREEN-V1', authorization='User instruction 执行 approves the full card and conditional gates; standing publication instruction requires a new dedicated branch, without overwriting history', status='FROZEN_BEFORE_MEASUREMENT', experiment_baseline=BASE, actual_server_head=cmd('git','rev-parse','HEAD').strip(), actual_server_branch=cmd('git','branch','--show-current').strip(), publication_branch=BRANCH, historical_fast_commit=FAST_COMMIT, historical_fast_core_diff=cmd('git','diff',FAST_COMMIT,BASE,'--',*core), current_core_diff=cmd('git','diff',BASE,'HEAD','--',*core), worktree_status=cmd('git','status','--porcelain=v1','-uall'), sources=sources, verification_code_sha256=sha(__file__), model=dict(path=str(ck['path']),sha256=sha(ck['path']), update=ck['update'],init_seed=ck['init_seed'],mode=m.mode,kind=m.kind,goal_marks=m.mg.goal_marks,layers=[dict(type=type(l).__name__, aggregation=l.aggr,num_relations=l.num_relations,num_bases=l.num_bases,root=l.root is not None) for l in fv.enc.layers], precision='float32',evaluation=True,weights_requires_grad=False), inputs=inputs, tasks=tasks, fixed_package=dict(parents=len(parents),transitions=group_count,hash_order='sha256(canonical lowercase hexadecimal state bitmask), ascending',task_rule='joint: sha256(case_id)-first available n6 and n8 case; IPC: pre-existing p06,p11,p12,p14,p15,p17 scope cases',max_successors_per_parent=16,missing_categories=['IPC p21/p22 historical time-cutoff large graphs have no scope parent package; p17 is largest accessible selected graph; no new states collected']), search=dict(open_key=['h','insertion_serial'],ties='FIFO',duplicates='complete state',closed_reopen=False,goal_check='root then each generated successor in canonical action order',batch_order='all new successors scored before insertion; no microbenchmark truncation in search',direction='minimize frozen V',first_group=dict(max_expansions=2000,wall_seconds=120,max_rss_growth_bytes=8*2**30), second_group=dict(tasks=[joint_ids[1],'ipc_p17'],repeats=2,max_expansions=10000,wall_seconds=120)), numeric=dict(atol=1e-5,rtol=1e-6,source='inherited FAST PLAN_ATOL/PLAN_RTOL; do not relax post hoc',deterministic_algorithms=False,cudnn_benchmark=False,tf32=False,scatter_atomic_nondeterminism='record F repeat differences; not bitwise equivalence'), benchmark=dict(device_physical_index=2,threads=2,outer_chunk=48,edge_budget=6000000,repeats=5,warmups=2,arm_order='alternate F,I/S then I/S,F',grouped_output='successor scores only; incremental parent full recomputation and cache lifetime charged; F batches successors normally',interleaved='round-robin one successor from each parent; contiguous windows of <=48 requests; parent-cache LRU capacity 2; no tuning',gpu_external_load='other compute PIDs on selected GPU invalidates paired block; maximum one repeat replacement; retain all attempts'), limits=dict(gpu_device_seconds=14400,cpu_core_seconds=86400,parents=128,successors=2048,searches=36,training=0,fitting=0,exact_queries=0,new_tasks=0,correctness_repairs=2,double_checks=32), gates=dict(G1='exact update path plus representative avoidable computation; no off-the-shelf complete scorer adaptation already verified',G2='no stale cache/nonfinite/unexplained tolerance failure, exact mathematical correspondence and complete accounting',G3='task geometric mean full-request speedup >=1.20, gain in >=2 strata, reasonable memory/noise'), environment=dict(python=platform.python_version(),torch=torch.__version__,pyg=torch_geometric.__version__,cpu=platform.processor(),cpu_count=os.cpu_count(),machine=platform.node(),gpu_inventory=cmd('nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader'),load=cmd('uptime'),meminfo=pathlib.Path('/proc/meminfo').read_text().splitlines()[:3]))
    assert not manifest['historical_fast_core_diff'] and not manifest['current_core_diff']
    sj(out/'parents.json',parents);manifest['parents_sha256']=sha(out/'parents.json');sj(out/'manifest.json',manifest)
    usage=resource.getrusage(resource.RUSAGE_SELF)
    sj(out/'prepare_receipt.json',dict(status='COMPLETE',wall_seconds=time.monotonic()-start,cpu_seconds=usage.ru_utime+usage.ru_stime,max_rss_kib=usage.ru_maxrss,parents=len(parents),transitions=group_count,model_loads=1,neural_forward_calls=0,ancillary_audit_cpu_reserve_seconds=120,training=0,exact_queries=0))

def audit(out):
    import numpy as np, torch
    from cp_disr.pddl.search_match.runner import OldRun,load_neural
    from cp_disr.pddl.task import depots_task,atom_id
    from cp_disr.pddl.fast_alt.fast_value import FastValue
    start=time.monotonic(); manifest=json.loads((out/'manifest.json').read_text());parents=json.loads((out/'parents.json').read_text())
    assert sha(out/'parents.json') == manifest['parents_sha256']
    model,_=load_neural(OldRun(OLD),'V_DENSE',torch.device('cpu'))
    rows=[];groups=[]
    for info in manifest['tasks']:
        cid=info['case_id'];task=depots_task(info['domain'],info['file']);fv=FastValue(model,torch.device('cpu'));fv.prepare(task,task.template())
        src,dst=fv.st.ei.numpy(); rel=fv.st.et.numpy();adj=defaultdict(list)
        for e,(s,t) in enumerate(zip(src,dst)):adj[int(s)].append((int(t),e))
        bitnode={i:task.template().node_ids.index(atom_id(*a)) for i,a in enumerate(task.dyn_atoms)}
        full_ops=4*(fv.R+1)*fv.N*128**2
        for p in [p for p in parents if p['case_id']==cid]:
            partial_ops=0
            for ch in p['successors']:
                old=int(p['parent'],16);new=int(ch['state'],16);assert task.apply(old,task.actions[ch['action_index']])==new
                bits=old^new; initial=set()
                while bits:
                    low=bits&-bits; initial.add(bitnode[low.bit_length()-1]);bits^=low
                affected=initial
                for li in range(1,5):
                    changed_edges={e for s in affected for _,e in adj[s]}
                    targets=affected|{int(dst[e]) for e in changed_edges}
                    rel_targets={r:{int(dst[e]) for e in changed_edges if int(rel[e])==r} for r in range(fv.R)}
                    mm=(len(affected)+sum(len(t) for t in rel_targets.values()))*128**2;partial_ops+=mm
                    rows.append(dict(case_id=cid,size_stratum=info['size_stratum'],parent_sha256=p['parent_sha256'],child_sha256=ch['state_sha256'],action_id=ch['action_id'],layer=li,nodes=fv.N,edges=fv.E,direct_changed_nodes=len(initial),input_affected_nodes=len(affected),output_affected_nodes=len(targets),output_node_fraction=len(targets)/fv.N,changed_messages=len(changed_edges),changed_edge_fraction=len(changed_edges)/max(fv.E,1),relation_target_updates=sum(len(t) for t in rel_targets.values()),root_updates=len(affected),avoided_dense_mac_estimate=1-mm/((fv.R+1)*fv.N*128**2),global_action_mean_update=True,global_prop_mean_update=True,goal_phi_attention_rho_recompute=True,parent_cache_bytes=9*fv.N*128*4,scope='exact conservative directed dependencies, not numerical-zero pruning; MAC estimate excludes memory/cache/readout costs'))
                    affected=targets
            k=len(p['successors']);groups.append(dict(case_id=cid,size_stratum=info['size_stratum'],parent_sha256=p['parent_sha256'],successors=k,full_legal_successors=p['full_legal_successors'],truncated=p['truncated'],dense_mac_avoidance_including_parent=1-(full_ops+partial_ops)/(max(k,1)*full_ops),parent_cache_bytes=9*fv.N*128*4))
        print('AUDITED',cid,flush=True)
    csv_write(out/'affected_work.csv',rows);csv_write(out/'group_work.csv',groups)
    good=[r for r in groups if r['successors']>1 and r['dense_mac_avoidance_including_parent']>0.10]
    gate=dict(status='PASS' if len(good)>=8 else 'STOP',supporting_groups=len(good),groups=len(groups),method_rule='Generic fixed-degree mean delta aggregation, relation-specific weights and root term, full ReLU/LayerNorm at affected rows, full global/goal readout; I and S are merged unless a distinct mechanism is established.',existing_method_complete_application_verified=False,nearest_neighbor='InkStream: mean accumulator mechanism supported, supplied GCN/SAGE/GIN edge-stream implementation requires adaptation for feature updates, RGCN relations/root and this global goal scorer; no claim of new algorithm. This gate admits an engineering screen, not novelty.',operation_estimates_only=True)
    sj(out/'G1.json',gate);usage=resource.getrusage(resource.RUSAGE_SELF)
    sj(out/'audit_receipt.json',dict(wall_seconds=time.monotonic()-start,cpu_seconds=usage.ru_utime+usage.ru_stime,max_rss_kib=usage.ru_maxrss,training=0,neural_forward_calls=0,exact_queries=0,enumerated_successors=0,checked_frozen_transitions=sum(len(p['successors']) for p in parents)))
    print('G1',json.dumps(gate),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','audit']);parser.add_argument('--run-root',type=pathlib.Path,required=True);args=parser.parse_args()
    globals()[args.stage](args.run_root)
