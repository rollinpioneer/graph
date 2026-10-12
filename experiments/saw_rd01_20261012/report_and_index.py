"""Assemble measured resource/reproduction reports and server raw-file identities."""
import csv
import subprocess
from common import *


def textfile(name,content):
    (ROOT/name).write_text(content.rstrip()+'\n')


def main():
    formal={k:json.load(open(ROOT/'results'/f'{k}_training_status.json')) for k in ('navigate','stitch')}
    final={k:json.load(open(ROOT/'results'/f'{k}_FINAL_EVAL.json')) for k in formal}
    resources={label:json.load(open(ROOT/'receipts'/f'{label}_resource.json')) for label in ('pilot_navigate','pilot_stitch','formal_navigate','formal_stitch')}
    s0=sum(resources[k]['gpu_residency_upper_seconds'] for k in resources if k.startswith('pilot_'))
    s1=sum(resources[k]['gpu_residency_upper_seconds'] for k in resources if k.startswith('formal_'))
    peak=max(r['peak_aggregate_mib'] for r in resources.values())
    assert s0<=7200 and s1<=28800 and s0+s1<=43200 and peak<=12288
    assert all(formal[k]['step']==1000000 and formal[k]['status']=='SINGLE_SEED_BASELINE' for k in formal)
    diag={}
    segments=groups=0
    for kind in formal:
        diag[kind]={}
        for phase in ('discovery','confirmation'):
            fp=ROOT/'results'/f'{kind}_{phase}_DIAGNOSIS.json'
            if fp.exists():
                d=json.load(open(fp));diag[kind][phase]=d;segments+=d['D3_segments'];groups+=d['state_goal_groups']
    assert segments<=128 and groups<=128
    accounting=dict(phase_gpu_seconds={'S0':s0,'S1':s1,'S2_S4':0},total_gpu_seconds=s0+s1,total_gpu_hours=(s0+s1)/3600,
                    peak_aggregate_memory_mib=peak,peak_aggregate_memory_gib=peak/1024,formal_jobs={k:resources['formal_'+k] for k in formal},
                    local_policy_segments=segments,candidate_state_goal_groups=groups,training_updates={'navigate':1000000,'stitch':1000000},
                    training_seeds=[0],extra_actor_training_updates=0,extra_baseline_training=False,
                    CPU_final_eval={k:dict(wall_seconds=v['wall_seconds'],process_cpu_seconds=v['cpu_process_seconds']) for k,v in final.items()},
                    CPU_diagnosis={k:{p:dict(wall_seconds=d['seconds'],process_cpu_seconds=d['cpu_process_seconds']) for p,d in phases.items()} for k,phases in diag.items()},
                    measurement_note='All complete GPU-child wall time charged, including import/load and CPU evaluation while GPU model resident; CPU-only final/diagnosis processes explicitly hide CUDA. Formal CPU ticks are separate lower-bound snapshots. Initial CPU install/audit timing was not fully instrumented; logs retained, no invented exact CPU cost.')
    # Avoid copying verbose memory samples twice; their original receipts remain authoritative.
    for d in accounting['formal_jobs'].values(): d.pop('samples',None)
    write_json(ROOT/'results/RESOURCE_TOTAL.json',accounting)
    lines=['# RESOURCE_REPORT','',f"Actual conservative GPU residency: {s0+s1:.6f} seconds = {(s0+s1)/3600:.6f} GPU hours, authorized 12.",
           f"S0: {s0:.6f}/7200 seconds; S1: {s1:.6f}/28800 seconds; S2–S4 GPU: 0/7200 seconds (CPU-only).",
           f"Aggregate task memory peak: {peak} MiB = {peak/1024:.6f} GiB, cap 12 GiB.",'',
           '| Job | Updates | Whole GPU-child wall seconds | Exit | Stop |','|---|---:|---:|---:|---|']
    for label,r in resources.items():
        updates=1011 if label.startswith('pilot') else 1000000
        lines.append(f"| {label} | {updates} | {r['gpu_residency_upper_seconds']:.6f} | {r['exit_code']} | {r['stop_reason']} |")
    lines.extend(['',f'{segments}/128 local diagnostic policy segments; {groups}/128 candidate state-goal groups, ≤16 candidates each. Two independent seed-0 baselines, no extra seed, QRL, or actor-only training.',
                  '',accounting['measurement_note'],'','CPU final-evaluation and diagnosis cost is tabulated in `results/RESOURCE_TOTAL.json`; formal process CPU ticks in `receipts/formal_CPU_ACCOUNTING.json`. Resource monitor overhead and initial CPU setup are not represented as exact per-module CPU costs. Shared GPU load changed after the pilots; measured completed job times are the outcome, pilot projections are estimates.'])
    textfile('RESOURCE_REPORT.md','\n'.join(lines))
    lines=['# REPRODUCTION_REPORT','',
           'Official SAW + local OGBench source: `3535a16c901e5de1233ac5a586401b3ddfa5e30d`. Full default network/batch/config, independent fresh training seed 0 per dataset, 1,000,000 joint updates each. Complete 100k/300k/500k/1M checkpoints include optimizer and all stochastic training state. Qualification pilot checkpoints are never reused for formal training.','',
           '| Baseline | Status | Fresh final success | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |','|---|---|---:|---:|---:|---:|---:|---:|']
    for k,v in final.items():
        status='SINGLE_SEED_BASELINE'+(' + PROTOCOL_APPROXIMATION' if k=='stitch' else '')
        lines.append('| '+k+' | '+status+' | '+f"{v['overall_success']:.1%}"+' | '+' | '.join(f"{v['tasks'][str(t)]:.1%}" for t in range(1,6))+' |')
    lines.extend(['','Final confirmation uses 5 tasks × 50 episodes per model, temperature 0, CPU evaluation, horizon 1000 and unchanged official success tolerance. Final seeds Navigate 91001 / Stitch 91002 are independent of development evaluation. Raw initial/goal/full trajectories are retained. This is a single-training-seed evaluation, not a cross-seed paper reproduction.',
                  '', 'Stitch uses preregistered Large Navigate defaults. No separate official SAW Stitch command or Appendix G hyperparameter row was found in the finite protocol audit. Inferred items: actor future-only goal distribution, 25-step subgoal horizon, temperatures 3/3/3, expectile .7, model/config defaults. The local OGBench source is explicitly fixed rather than silently upgraded through PyPI. Independent evaluator RNG isolation, deterministic zero-temperature mode, local logging, extra checkpoints and video omission are recorded instrumentation choices.',
                  '', 'Tests passed: dataset original identities and finite fields, valid transitions/trajectory bounds, exact original sampler rewards/masks/subgoal indices, independent sampler/validation RNG, exact pilot parameter/optimizer/JAX RNG equality with logging on/off, true gradient/optimizer updates, synchronized cost timings, unchanged evaluator success/non-success/short-positive controls, and hard aggregate memory/budget gates.',
                  '', 'Development evaluations are retained in `results/*_development_eval.json`; raw development trajectories are not recycled as final confirmation. All losses are in `results/*_training.csv`. Source code versions, data hashes and raw checkpoint/trajectory paths are in `EVIDENCE_INDEX.json`.'])
    textfile('REPRODUCTION_REPORT.md','\n'.join(lines))
    audit=json.load(open(ROOT/'results/DATA_AUDIT.json')); coverage=json.load(open(ROOT/'results/D1_COVERAGE.json'))
    lines=['# DATA_AUDIT','', '| Data | Train trajectories | Raw trajectory rows | Effective transitions | Free-cell coverage | Intercell directed edges | High-goal median horizon |','|---|---:|---:|---:|---:|---:|---:|']
    for k,a in audit.items():
        c=coverage[k];f=a['files'][0]
        lines.append(f"| {k} | {f['trajectories']} | {f['trajectory_length_min']} | {c['valid_transitions']} | {c['occupied_cells']}/{c['free_cells']} | {c['directed_intercell_edges']} | {c['high_goal_horizon_quantiles'][2]} |")
    lines.extend(['','Both datasets use exactly the same registered Large map and five evaluation tasks. All five task endpoints are connected in the observed directed cell-transition graph for both datasets. This excludes an obvious missing-cell disconnection; it does not establish full-state controllability, learned policy competence, or a single-factor cause.','',
                  'State observations are qpos (two dimensions), no normalization. Original qpos/qvel/action/terminal fields remain in hashed NPZ files. Server index preserves original trajectory ID, step ID, observation/action/next observation and validity. Invalid terminal rows are explicitly masked; no boundary-crossing tuple is a real transition. Official compact terminal remapping is preserved, including terminal marking at both the final raw row and its predecessor.','',
                  'Each dataset has 4,096 fixed training-rule probes and shared (s,g,w) indices for D1–D4. Current/future/random value-goal mixture and future-only high-actor goals pass source-level and measured sampler checks. Navigate/Stitch differ in horizon, occupancy weights and collection policy simultaneously. Coarse BFS geometry is a privileged diagnostic proxy, not discounted-value truth. See `results/DATA_AUDIT.json`, `results/SAMPLER_AUDIT.json`, `results/D1_COVERAGE.json` and per-sample CSVs.'])
    textfile('DATA_AUDIT.md','\n'.join(lines))
    raw_files=[]
    for base in (ROOT/'vendor/data',ROOT/'runs',ROOT/'logs'):
        for path in sorted(base.rglob('*')):
            if path.is_file(): raw_files.append(dict(path=str(path),relative=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=sha(path)))
    critical=['impls/agents/saw.py','impls/main.py','impls/hyperparameters.sh','impls/utils/datasets.py','impls/utils/evaluation.py','impls/utils/networks.py','impls/utils/flax_utils.py','ogbench/utils.py','ogbench/locomaze/maze.py','ogbench/locomaze/point.py','ogbench/locomaze/__init__.py','pyproject.toml']
    source=[dict(relative=p,sha256=sha(ROOT/'vendor/saw'/p)) for p in critical]
    write_json(ROOT/'EVIDENCE_INDEX.json',dict(server_root=str(ROOT),official_source_commit='3535a16c901e5de1233ac5a586401b3ddfa5e30d',critical_official_source=source,raw_files=raw_files,
                                             lightweight_evidence='LIGHT_MANIFEST.json lists exact committed code/config/reports/results identities; raw weights/data/full logs are excluded from Git'))
    info=json.load(open(ROOT/'BRANCH_INFO.json'));info.update(status='COMPLETED_SINGLE_SEED_WITH_DIAGNOSIS',actual_gpu_seconds=s0+s1,actual_peak_aggregate_mib=peak,baseline_status={k:v['status'] for k,v in formal.items()},stitch_protocol='PROTOCOL_APPROXIMATION',confirmation_executed=all('confirmation' in d for d in diag.values()),local_policy_segments=segments)
    write_json(ROOT/'BRANCH_INFO.json',info)
    card=ROOT/'EXPERIMENT_CARD.md'
    marker='\n## Completed execution annex\n'
    body=card.read_text().split(marker)[0]
    body+=marker+'\nBoth separate seed-0 models completed 1,000,000 joint updates. Fresh final success: '+', '.join(f"{k} {v['overall_success']:.1%}" for k,v in final.items())+'. Stitch additionally remains PROTOCOL_APPROXIMATION.\n\nActual conservative GPU time '+f'{(s0+s1)/3600:.6f} hours; aggregate memory peak {peak/1024:.6f} GiB. '+'D1–D4 evidence, all failures/untriggered cases and the conditional intervention decision are archived in the adjacent reports. No QRL, extra seeds, architecture changes or extra actor training. Raw evidence remains at the server root indexed by EVIDENCE_INDEX.json.\n'
    card.write_text(body)
    print('REPORT_INDEX_DONE',accounting['total_gpu_hours'],peak,segments,groups,flush=True)


if __name__=='__main__':main()
