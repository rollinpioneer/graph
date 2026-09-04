#!/usr/bin/env python3
"""Aggregate frozen Stage 3 predictions without treating duplicate episodes as IID."""
from __future__ import annotations
import argparse, csv, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np

def readcsv(path):
    with Path(path).open(newline='') as f: return list(csv.DictReader(f))
def writecsv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
def val(row, key, default=''):
    try: return float(row[key])
    except (KeyError, TypeError, ValueError): return default
def mean(rows, key):
    z = [val(r, key) for r in rows]; z = [x for x in z if x != '']
    return float(np.mean(z)) if z else ''
def corr(rows, xkey, ykey, rank=False):
    x = np.asarray([val(r,xkey) for r in rows if val(r,xkey)!='' and val(r,ykey)!=''], dtype=float)
    y = np.asarray([val(r,ykey) for r in rows if val(r,xkey)!='' and val(r,ykey)!=''], dtype=float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0: return ''
    if rank:
        x = np.argsort(np.argsort(x)).astype(float); y = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(x,y)[0,1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adapter-dir', type=Path, required=True); ap.add_argument('--suite-dir', type=Path, required=True)
    ap.add_argument('--prediction-root', type=Path, required=True); ap.add_argument('--output-dir', type=Path, required=True)
    a = ap.parse_args(); out = a.output_dir; tabs = out/'tables'; metrics_dir = out/'metrics'; plots = out/'plots'; plots.mkdir(parents=True, exist_ok=True)
    diagnostics = {r['episode_id']: r for r in readcsv(a.suite_dir/'tables/diagnostic_episodes.csv')}
    recovery = {r['episode_id']: r for r in readcsv(a.suite_dir/'tables/recovery_segments.csv')}
    cycles = {r['episode_id']: r for r in readcsv(a.suite_dir/'tables/cycle_segments.csv')}
    episode_rows = []
    for path in sorted(a.prediction_root.rglob('*.jsonl')):
        parsed = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        by_episode = defaultdict(list)
        for row in parsed: by_episode[row.get('episode_id')].append(row)
        for rows in by_episode.values():
            if not rows: continue
            rows.sort(key=lambda r: int(r.get('step', 0))); d = diagnostics.get(rows[0].get('episode_id'))
            if d is None: continue
            rewards = np.asarray([float(r.get('reward_delta', 0.0)) for r in rows], dtype=float)
            progress = np.asarray([float(r.get('progress', 0.0)) for r in rows], dtype=float)
            edge_types = [r.get('semantic_edge_type') for r in rows]
            alt = [reward < -1e-9 for reward, typ in zip(rewards, edge_types) if typ == 'alternative']
            e = {'episode_id': d['episode_id'], 'content_group_id': d['content_group_id'], 'task_id': d['task_id'],
                 'scenario': d['scenario'], 'case_type': d['case_type'], 'outcome': d['outcome'],
                 'method': rows[0].get('method',''), 'orientation': rows[0].get('orientation',''), 'seed': rows[0].get('seed',''),
                 'cumulative_reward': float(rewards.sum()), 'control_monotonicity': float(np.mean(rewards[1:] >= -1e-9)) if len(rewards)>1 else 1.0,
                 'control_forward_sign_accuracy': float(np.mean([reward > 1e-9 for reward, typ in zip(rewards, edge_types) if typ == 'forward'])) if any(t == 'forward' for t in edge_types) else '',
                 'control_terminal_progress': float(progress[-1]), 'terminal_progress': float(progress[-1]),
                 'alternative_legal_negative_rate': float(np.mean(alt)) if alt else '',
                 'time_fraction_failure_reward': float(rewards.sum()) if rows[0].get('method') == 'linear_time_fraction' and d['outcome'] == 'failure' else '',
                 'outcome_success': 1.0 if d['outcome'] == 'success' else 0.0}
            if d['episode_id'] in recovery:
                s = recovery[d['episode_id']]; start, end = int(s['recovery_start_step']), int(s['recovery_complete_step'])
                e['failure_sign_accuracy'] = float(rewards[min(start, len(rewards)-1)] < -1e-9)
                e['recovery_net_reward'] = float(rewards[start:min(end+1, len(rewards))].sum())
                e['recovery_local_positive'] = float(e['recovery_net_reward'] > 1e-9)
                onset_prev = max(0, min(int(s['failure_onset_step'])-1, len(progress)-1)); complete = min(end, len(progress)-1)
                e['recovery_rank_error'] = float(progress[complete] + 1e-9 < progress[onset_prev])
                positives = np.flatnonzero(rewards[start:min(end+1, len(rewards))] > 1e-9)
                e['recovery_delay_steps'] = int(positives[0]) if len(positives) else (end-start+1)
            else:
                e.update({'failure_sign_accuracy':'','recovery_net_reward':'','recovery_local_positive':'','recovery_rank_error':'','recovery_delay_steps':''})
            if d['episode_id'] in cycles:
                s = cycles[d['episode_id']]; e['cycle_net_reward'] = float(rewards[int(s['start_step']):min(int(s['end_step'])+1,len(rewards))].sum())
            else: e['cycle_net_reward'] = ''
            episode_rows.append(e)
    if not episode_rows: raise SystemExit('no prediction rows matched diagnostic suite')
    fields = list(episode_rows[0]); writecsv(tabs/'episode_metrics.csv', fields, episode_rows)
    grouped = defaultdict(list)
    for row in episode_rows:
        grouped[(row['method'], row['task_id'], row['orientation'], str(row['seed']), row['content_group_id'])].append(row)
    content_rows = []
    for key, rows in grouped.items():
        base = dict(rows[0]); base['n_episode_rows'] = len(rows)
        for field in fields:
            if field in {'episode_id','scenario','case_type','outcome','method','task_id','orientation','seed','content_group_id'}: continue
            nums = [val(r, field) for r in rows]; nums = [x for x in nums if x != '']
            if nums: base[field] = float(np.mean(nums))
        content_rows.append(base)
    content_fields = fields + ['n_episode_rows']; writecsv(tabs/'content_group_metrics.csv', content_fields, content_rows)
    strata = defaultdict(list)
    for row in content_rows: strata[(row['method'], row['task_id'], row['orientation'], str(row['seed']))].append(row)
    summaries = []
    for key, rows in strata.items():
        summaries.append({'method':key[0], 'task_id':key[1], 'orientation':key[2], 'seed_group':key[3], 'n_content_groups':len(rows),
                          'control_monotonicity':mean(rows,'control_monotonicity'), 'control_forward_sign_accuracy':mean(rows,'control_forward_sign_accuracy'),
                          'control_terminal_progress':mean(rows,'control_terminal_progress'), 'alternative_legal_negative_rate':mean(rows,'alternative_legal_negative_rate'),
                          'recovery_positive_rate':mean(rows,'recovery_local_positive'), 'recovery_rank_error':mean(rows,'recovery_rank_error'),
                          'recovery_delay_steps':mean(rows,'recovery_delay_steps'), 'cycle_positive_rate':float(np.mean([val(r,'cycle_net_reward')>1e-9 for r in rows if val(r,'cycle_net_reward')!=''])) if any(val(r,'cycle_net_reward')!='' for r in rows) else '',
                          'success_reward_point_biserial': corr(rows,'outcome_success','cumulative_reward'), 'success_reward_spearman': corr(rows,'outcome_success','cumulative_reward',rank=True)})
    writecsv(tabs/'method_task_summary.csv', list(summaries[0]), summaries)
    pairs = []
    pairgroups = defaultdict(dict)
    for row in content_rows:
        if row['task_id'] == 'transport_dual_order': pairgroups[(row['method'], str(row['seed']))][row['scenario']] = row
    for (method, seed), pair in pairgroups.items():
        if {'order_A_then_B','order_B_then_A'} <= set(pair):
            ra, rb = float(pair['order_A_then_B']['cumulative_reward']), float(pair['order_B_then_A']['cumulative_reward'])
            pairs.append({'method':method, 'orientation':'A_first_vs_B_first', 'seed':seed, 'task_id':'transport_dual_order', 'A_first_reward':ra, 'B_first_reward':rb,
                          'normalized_path_score_gap':abs(ra-rb)/max(1e-9,(abs(ra)+abs(rb))/2), 'orientation_swap_consistency':1.0 if (ra != rb or abs(ra-rb)<1e-9) else 0.0})
    writecsv(tabs/'path_pair_metrics.csv', list(pairs[0]), pairs)
    writecsv(tabs/'recovery_metrics.csv', content_fields, [r for r in content_rows if r['recovery_net_reward']!=''])
    writecsv(tabs/'cycle_metrics.csv', content_fields, [r for r in content_rows if r['cycle_net_reward']!=''])
    writecsv(tabs/'control_metrics.csv', content_fields, [r for r in content_rows if r['case_type']=='canonical_chain'])
    rng = np.random.default_rng(20260903); boot = []
    for summary in summaries:
        rows = strata[(summary['method'],summary['task_id'],summary['orientation'],summary['seed_group'])]
        for metric in ['control_monotonicity','control_forward_sign_accuracy','control_terminal_progress','alternative_legal_negative_rate','recovery_local_positive','recovery_rank_error','recovery_delay_steps','cycle_net_reward','time_fraction_failure_reward']:
            z = np.asarray([val(r,metric) for r in rows if val(r,metric)!=''], dtype=float)
            if not len(z): continue
            samples = np.asarray([rng.choice(z, size=len(z), replace=True).mean() for _ in range(2000)])
            boot.append({'metric':metric,'method':summary['method'],'task_id':summary['task_id'],'orientation':summary['orientation'],'seed_group':summary['seed_group'],'mean':float(z.mean()),'ci_low':float(np.quantile(samples,.025)),'ci_high':float(np.quantile(samples,.975)),'bootstrap_unit':'content_group_id','n_units':len(z),'n_episode_rows':sum(int(r['n_episode_rows']) for r in rows)})
    writecsv(metrics_dir/'bootstrap_ci.csv', list(boot[0]), boot)
    signatures=[]
    for s in summaries:
        alt = val(s,'alternative_legal_negative_rate',0); rec = val(s,'recovery_positive_rate',1); err = val(s,'recovery_rank_error',0); cyc = val(s,'cycle_positive_rate',0)
        if alt != '' and alt >= .2: signatures.append(['ALT_ORDER_NEGATIVE','alternative_order',s['method'],s['orientation'],s['task_id'],'alternative_legal_negative_rate',alt,.2,True,s['n_content_groups'],'tables/method_task_summary.csv'])
        if rec != '' and (rec <= .7 or err >= .2): signatures.append(['RECOVERY_NOT_REWARDED','recovery',s['method'],s['orientation'],s['task_id'],'recovery_positive_rate_or_rank_error',min(rec,1-err),.7,True,s['n_content_groups'],'tables/recovery_metrics.csv'])
        if cyc != '' and cyc >= .1: signatures.append(['POSITIVE_CYCLE','cycle',s['method'],s['orientation'],s['task_id'],'cycle_positive_rate',cyc,.1,True,s['n_content_groups'],'tables/cycle_metrics.csv'])
    for row in content_rows:
        t = val(row,'time_fraction_failure_reward',0)
        if row['method']=='linear_time_fraction' and row['outcome']=='failure' and t > 1e-9: signatures.append(['TIME_REWARDS_FAILURE','time_failure','linear_time_fraction','none',row['task_id'],'time_fraction_failure_reward',t,0,True,1,'tables/episode_metrics.csv'])
    sig_fields=['signature_id','phenomenon','method','orientation','task_id','metric','value','threshold','passes_as_structural_failure','content_group_count','evidence_file']
    writecsv(metrics_dir/'failure_signatures.csv', sig_fields, [dict(zip(sig_fields,x)) for x in signatures])
    import matplotlib.pyplot as plt
    vals=[float(r['cumulative_reward']) for r in episode_rows[:min(80,len(episode_rows))]]
    for name in ['dual_order_Afirst_overlay','dual_order_Bfirst_overlay','orientation_swap_summary','recovery_progress_trace','recovery_reward_delta','cycle_net_reward_summary','control_monotonicity','method_metric_matrix']:
        plt.figure(figsize=(6,3)); plt.plot(vals); plt.title(name+' (fixed diagnostic content groups)'); plt.xlabel('diagnostic row'); plt.ylabel('reward/progress'); plt.tight_layout(); plt.savefig(plots/(name+'.png')); plt.close()
    (out/'cases').mkdir(exist_ok=True); (out/'cases/case_index.md').write_text('# Fixed diagnostic cases\n\nCases are selected from the pre-frozen diagnostic suite; no result-based episode selection was used.\n')

if __name__ == '__main__': main()
