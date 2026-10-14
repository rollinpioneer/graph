"""Assemble zero-training acceptance only after all required checks pass."""
import hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads((ROOT/path).read_text())
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
hosts={h:read(f'reports/zero_train_{h}.json') for h in ['gfp','dfql']}
for host,data in hosts.items():
 assert data['status']=='ZERO_TRAIN_VERIFIED' and data['optimizer_steps']==0 and data['forbidden_call_count']==0
 assert all(t['status']=='PASS' for t in data['tests'])
 pid=int((ROOT/'receipts'/f'zero_train_{host}.pid').read_text());assert not Path(f'/proc/{pid}').exists()
aux={name:read('reports/'+name+'.json') for name in ['existing_restore_gfp','existing_restore_fql','existing_restore_floq',
     'source_value_mapping','target_definition_test','zero_update_artifact_audit']}
assert all(d['status']=='PASS' and d['optimizer_steps']==0 for d in aux.values())
index=read('configs/index.json');assert len(index)==12
assert all(read('configs/'+name)['training_authorized'] is False and read('configs/'+name)['optimizer_steps_allowed']==0
           for name in index)
results=dict(card='FLOW-CRITIC-DUAL-HOST-PREP-V1',status='ZERO_TRAIN_VERIFIED',optimizer_steps=0,
             online_learning=False,training_started=False,hosts=hosts,additional_tests=aux,
             test_protocol=dict(initialization_seed=14014,loss_rng_seed=81214,batch_index_seed=14014,
                 batch_size=256,source_restore_batch_size=8,scalar_path_atol=0,scalar_path_rtol=0,
                 source_loss_atol=2e-5,source_loss_rtol=2e-5,value_mapping_atol=5e-4,value_mapping_rtol=5e-4),
             limitations=['No performance, training stability, cost or novelty conclusion',
                 'No DFQL trained weights / Baseline reproduction', 'No new-task Baseline reproduction',
                 'Named prior reports unavailable; card and pinned source define engineering contract',
                 'Old whole Agents are not declared compatible with combined parameter/optimizer trees'],timestamp=time.time())
(ROOT/'reports/zero_train_test_results.json').write_text(json.dumps(results,indent=2))
for host in ['gfp','dfql']:
 w=ROOT/'workspaces'/f'{host}_floq'
 pieces=[]
 for p in [w/'agents/floq_adapter.py',*sorted((w/'floq_value_core').glob('*.py'))]:
  rel=str(p.relative_to(w));lines=p.read_text().splitlines()
  pieces.append(f'diff --git a/{rel} b/{rel}\nnew file mode 100644\n--- /dev/null\n+++ b/{rel}\n@@ -0,0 +1,{len(lines)} @@\n'+'\n'.join('+'+l for l in lines)+'\n')
 (ROOT/'reports'/f'{host}_workspace.patch').write_text('\n'.join(pieces))
codefiles=[*sorted((ROOT/'scripts').glob('*.py')),*sorted((ROOT/'workspaces/gfp_floq/floq_value_core').glob('*.py')),
           ROOT/'workspaces/gfp_floq/agents/floq_adapter.py',ROOT/'workspaces/dfql_floq/agents/floq_adapter.py']
(ROOT/'reports/implementation_code_identity.json').write_text(json.dumps(
    [{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in codefiles],indent=2))
assets=read('assets_manifest/asset_manifest.json')
rows='\n'.join(f"| {d['task']} | {d['train_transitions']:,} | {d['fields']['observations']['shape'][-1]} / {d['fields']['actions']['shape'][-1]} | {d['reward_range']} |" for d in assets['datasets'])
counts={h:len(d['tests']) for h,d in hosts.items()}
receipt=f'''# Implementation receipt — FLOW-CRITIC-DUAL-HOST-PREP-V1

Status: **ZERO_TRAIN_VERIFIED**. GFP + floq Critic and DFQL + floq Critic implementation and zero-training functional acceptance are complete, ready for a separately authorized training design.

**This card executed 0 optimizer/parameter updates, 0 training runs and no online learning.** GPU work consisted solely of forward, loss, gradient and JIT checks. New random-state tests use initialized state step 1, unchanged afterward; restored historical model step values are also unchanged.

Server root: `{ROOT}`.
Local archive branch: `codex/flow-critic-dual-host-prep-v1-20261014`. No push, PR or merge. Original primary repository branch, HEAD and working status are unchanged.

Completed: required pinned sources and private host environments, official data loading/relabeling for four tasks, existing-asset identity audit, namespaced floq expected-value core, host adapters, 12 complete G0/G1/D0/D1 task configurations and six differences, and Z0–Z8 verification. Original host files and network utilities are unchanged; source snapshot files are read-only.

| Task | Train transitions | Observation / action dims | Actual rewards |
|---|---:|---|---|
{rows}

Host GPU suites: GFP {counts['gfp']} checks, DFQL {counts['dfql']} checks, all PASS. Original scalar path controls compare outputs, losses and all parameter gradients exactly (atol=rtol=0). Both suites check real 256-transition batches. Flow/scalar/Actor/VaBC gradient routes are isolated; targets have zero parameter gradients. Synthetic tests cover noise SUM, terminal masking, aggregation ordering, current/next state branches, multi-step TD policy, clipped-endpoint action derivative and target EMA maps. Additional pinned-source critic-loss controls and original floq value mapping are included.

44 old checkpoint files are byte-hash checked before and after. Existing final FQL/GFP seeds 0/1 and floq 1M/2M seeds 0/1 restore read-only in their original environments; Q and action derivatives remain finite. Both new randomly initialized combinations serialize and restore independently. Existing weights are not treated as valid new-host value functions.

GFP CSV matching uses the actual official routine after OmegaConf parsing, no CLI overrides. Humanoid η=0.001 agrees with the card. The earlier generic YAML parse failure is retained as an engineering failure receipt and occurred before parameter updates. floq's actual time input is cos(t), one channel, with inherited inert time_embed_dim=64. See the source/environment manifest for all differences and resolutions.

Unavailable: the two named prior report files. They are logged as MISSING_USER_ACCESS; the card and pinned sources supply the implemented contract. DFQL trained weights are NOT_YET_TRAINED. New-task Baselines are not reproduced. Dataset/weight files and full logs stay on the server and are not committed as lightweight artifacts.

This acceptance makes no claim about performance, superiority, training stability, full-training cost, paper reproduction of the combinations or novelty. The next decision must separately authorize Baseline admission, training/evaluation protocols and any scope or seed expansion. The current preparation stops here.
'''
(ROOT/'reports/implementation_receipt.md').write_text(receipt)
commands=f'''# Zero-training commands (no training main)
ROOT={ROOT}
# P0, direct official OGBench1.1.0 loading; GFP configs are resolved with its pinned OmegaConf/CSV matcher.
$ROOT/envs/dfql/bin/python $ROOT/scripts/prepare_assets.py
# GPU pure checks. Environment: CUDA_VISIBLE_DEVICES=3/7 respectively; XLA_PYTHON_CLIENT_PREALLOCATE=false;
# OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1; PYTHONDONTWRITEBYTECODE=1; no precision override.
taskset -c 52-55 $ROOT/envs/gfp/bin/python $ROOT/scripts/zero_train_verify.py --host gfp
taskset -c 56-59 $ROOT/envs/dfql/bin/python $ROOT/scripts/zero_train_verify.py --host dfql
# Original read-only restores use OLD/envs/gfp312, OLD/envs/fql, FLOQ/envs/floq respectively;
# JAX_PLATFORMS=cpu, CUDA_VISIBLE_DEVICES='', no updates, no main.py.
<original-env-python> $ROOT/scripts/restore_existing.py --method <gfp|fql|floq>
$ROOT/envs/gfp/bin/python $ROOT/scripts/map_source_value.py
$ROOT/envs/gfp/bin/python $ROOT/scripts/verify_target_definition.py
$ROOT/envs/dfql/bin/python $ROOT/scripts/audit_contracts.py
$ROOT/envs/dfql/bin/python $ROOT/scripts/finalize_preparation.py
'''
(ROOT/'reports/ZERO_TRAIN_COMMANDS.txt').write_text(commands)
print('FINAL_ACCEPTANCE',counts,'optimizer_steps=0',flush=True)
