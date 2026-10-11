"""Saved-evidence-only verifier. No model loading, forward, or search."""
import ast
import copy
import csv
import gzip
import hashlib
import itertools
import json
import struct
import sys
import time
from pathlib import Path

START = time.process_time()
ROOT = Path(sys.argv[1])
REG = json.loads((ROOT / 'preregistration.json').read_text())
PREV = Path(REG.get('analysis_previous_root', REG['previous_remote_root']))
META = json.loads((PREV / 'event_approval_metadata.json').read_text())
EVENTS = {e['event_id']: e for e in META['events']}
ROWS = json.loads((ROOT / 'branch_results.json').read_text())
RUN = json.loads((ROOT / 'run_receipt.json').read_text())
# Import pure serialization and comparison functions only; never execute runner main/imports.
tree = ast.parse((ROOT / 'phase_c.py').read_text())
pure = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ['value_from_bits', 'obj_digest', 'future_scene']]
exec(compile(ast.Module(body=pure, type_ignores=[]), 'phase_c_pure', 'exec'), globals())


def save(name, obj):
    (ROOT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def records(path):
    with gzip.open(path, 'rt') as f:
        for line in f:
            yield json.loads(line)


def compare_logs(a, b):
    n, mismatch = 0, None
    for x, y in zip(records(a), records(b)):
        if x != y:
            mismatch = n
            break
        n += 1
    return {'common_identical_complete_steps': n, 'first_full_content_mismatch_zero_based': mismatch, 'PASS_common_prefix': mismatch is None and n > 0}


def log_info(path):
    out = []
    for i, rec in enumerate(records(path)):
        s = rec['after']
        out.append({'step': i+1, 'decision': rec['decision'], 'exact_sha256': obj_digest(s), 'future_sha256': obj_digest(future_scene(s)),
                    'generated': s['generated'], 'duplicates': s['duplicates'], 'path_updates': s['path_updates'],
                    'calls': s['calls'], 'cache_misses': s['score_cache_misses'], 'valid_OPEN': sum(e[3] for e in s['heap'])})
    return out


def record_at(path, k):
    if k == 0:
        return json.loads((path.parent / 'initial.json').read_text())
    return next(rec['after'] for i, rec in enumerate(records(path), 1) if i == k)


def verify_ledger():
    starts, ends, failures = {}, {}, []
    branches = {r['slot']: r for r in ROWS}
    same_keys = {}
    forward_by_slot = {}
    interventions = []
    for line in (ROOT / 'branch_ledger.jsonl').read_text().splitlines():
        e = json.loads(line)
        if e['kind'] == 'MODEL_STATE_START':
            assert e['sequence'] not in starts
            starts[e['sequence']] = e
            forward_by_slot[e['slot']] = forward_by_slot.get(e['slot'], 0)+1
        elif e['kind'] == 'MODEL_STATE_END':
            assert e['sequence'] not in ends
            ends[e['sequence']] = e
            a = starts[e['sequence']]
            rawpath = ROOT / branches[a['slot']]['directory'] / 'score_bytes.bin'
            with rawpath.open('rb') as f:
                f.seek(a['byte_offset'])
                b = f.read(4).hex()
            if e['status'] == 'OK':
                assert b == e['value_bits_le']
                key = (a['event'], a['state'], a['model'], a['protocol'])
                assert key not in same_keys or same_keys[key] == b
                same_keys[key] = b
            else:
                failures.append(e)
        elif e['kind'] == 'INTERVENTION_AUDIT':
            assert e['every_other_field_unchanged'] and e['remaining_logical_heap_exact']
            interventions.append(e)
    assert starts.keys() == ends.keys()
    assert len(starts) == RUN['counts']['actual_model_state_evaluations']
    for row in ROWS:
        assert forward_by_slot.get(row['slot'], 0) == row['actual_model_state_evaluations'] <= 4096
        if 'directory' in row:
            assert (ROOT / row['directory'] / 'score_bytes.bin').stat().st_size == row['actual_model_state_evaluations']*4
    assert len(starts) <= 32768
    return {'actual_forward_start_end_pairs': len(starts), 'all_raw_score_bytes_verified': True, 'unique_scored_keys': len(same_keys),
            'failed_or_interrupted_forward_records': failures, 'interventions': interventions, 'per_slot_forwards': forward_by_slot}


LEDGER_CHECK = verify_ledger()
REPEAT, ANALYSIS, COALESCENCE = [], [], []
for eid in ['struct_n4_004_k008', 'ipc_p12_k002']:
    e = EVENTS[eid]
    rr = {r['branch']: r for r in ROWS if r['event'] == eid}
    if any('directory' not in rr[b] for b in ['A', 'B', 'A_repeat', 'B_repeat']):
        ANALYSIS.append({'event': eid, 'status': 'TECHNICALLY_INVALID', 'reason': 'MISSING_OR_NOT_RUN_SLOT'})
        continue
    paths = {b: ROOT / r['directory'] / 'steps.jsonl.gz' for b, r in rr.items()}
    checks = {'event': eid, 'A_vs_A_repeat': compare_logs(paths['A'], paths['A_repeat']), 'B_vs_B_repeat': compare_logs(paths['B'], paths['B_repeat'])}
    baseline_path = PREV / 'decisions' / (e['task']['case_id'] + '_prefix.jsonl.gz')
    old = [r for r in records(baseline_path) if r['decision'] and r['decision']['expanded'] > e['k_completed_expansions']]
    checks['old_suffix_steps'] = len(old)
    checks['old_suffix'] = {}
    for b in ['A', 'A_repeat']:
        now = list(itertools.islice(records(paths[b]), len(old)))
        mismatch = next((i for i, (x, y) in enumerate(zip(now, old)) if x != y), None)
        checks['old_suffix'][b] = {'steps_compared': min(len(now), len(old)), 'first_mismatch': mismatch,
                                  'all_available_old_suffix_reached': len(now) == len(old), 'PASS_common_prefix': mismatch is None and len(now) > 0}
    valid = all(checks[k]['PASS_common_prefix'] for k in ['A_vs_A_repeat', 'B_vs_B_repeat']) and all(v['PASS_common_prefix'] for v in checks['old_suffix'].values())
    for b in rr:
        valid = valid and rr[b]['stop_reason'] != 'TECHNICALLY_INVALID'
    checks['technical_validity'] = 'PASS' if valid else 'TECHNICALLY_INVALID'
    REPEAT.append(checks)
    data = {b: log_info(path) for b, path in paths.items()}
    common = min(len(v) for v in data.values())
    a, b = data['A'][:common], data['B'][:common]
    seq_div = next((i+1 for i in range(1, common) if a[i]['decision'] != b[i]['decision']), None)
    state_div = next((i+1 for i in range(1, common) if a[i]['exact_sha256'] != b[i]['exact_sha256']), None)
    lcp = 0
    for x, y in zip(a[1:], b[1:]):
        if x['decision'] != y['decision']:
            break
        lcp += 1
    # All cumulative checkpoints are indexed before candidate hash matches are confirmed by content.
    co = {'event': eid, 'rule': REG['coalescence'], 'hash_candidates': [], 'content_confirmed_matches': [],
          'scope': 'A/B initial plus complete checkpoints; (0,0) excluded as preintervention identity'}
    def states_index(branch):
        initial = json.loads((paths[branch].parent / 'initial.json').read_text())
        return [{'step': 0, 'future_sha256': obj_digest(future_scene(initial)), 'exact_sha256': obj_digest(initial)}]+data[branch]
    ai, bi = states_index('A'), states_index('B')
    for rule in ['exact_sha256', 'future_sha256']:
        lookup = {}
        for x in ai:
            lookup.setdefault(x[rule], []).append(x['step'])
        for y in bi:
            for i in lookup.get(y[rule], []):
                j = y['step']
                if i == j == 0:
                    continue
                candidate = {'rule': rule, 'A_step': i, 'B_step': j, 'sha256': y[rule]}
                co['hash_candidates'].append(candidate)
                xobj, yobj = record_at(paths['A'], i), record_at(paths['B'], j)
                if rule == 'future_sha256':
                    xobj, yobj = future_scene(xobj), future_scene(yobj)
                assert xobj == yobj
                co['content_confirmed_matches'].append(candidate)
    co['status'] = 'CONFIRMED' if co['content_confirmed_matches'] else 'NO_OBSERVED_COALESCENCE_IN_SAVED_WINDOW'
    COALESCENCE.append(co)
    solved = all(r['status'] == 'SOLVED' and r['plan_valid'] for r in rr.values())
    delta_e = rr['B']['completed_additional_expansions']-rr['A']['completed_additional_expansions'] if solved and valid else None
    delta_l = rr['B']['total_plan_length']-rr['A']['total_plan_length'] if solved and valid else None
    if not valid:
        status = 'TECHNICALLY_INVALID'
    elif solved:
        status = 'VALID_COMPLETE_EFFECT' if delta_e != 0 or delta_l != 0 else 'VALID_COMPLETE_NO_OBSERVED_EFFECT'
    elif co['content_confirmed_matches']:
        status = 'VALID_COALESCENCE'
    elif common > 1 and (seq_div is not None or state_div is not None):
        status = 'VALID_BOUNDED_EFFECT'
    else:
        status = 'CENSORED_UNRESOLVED'
    initial = json.loads((paths['A'].parent / 'initial.json').read_text())
    fixed_window = {}
    for branch in ['A', 'B', 'A_repeat', 'B_repeat']:
        end = data[branch][common-1] if common else None
        fixed_window[branch] = None if end is None else {k: end[k]-initial[{'cache_misses':'score_cache_misses'}.get(k, k)] for k in ['generated', 'duplicates', 'path_updates', 'calls', 'cache_misses']}
        if end:
            fixed_window[branch]['valid_OPEN'] = end['valid_OPEN']
    opposite_pop = {}
    for branch, target in [('A', e['choices']['T1@2460']['state']), ('B', e['choices']['D0@820']['state'])]:
        hits = [r['step'] for r in data[branch] if r['decision'] and r['decision']['state'] == target]
        opposite_pop[branch] = hits[0] if hits else None
    ANALYSIS.append({'event': eid, 'status': status, 'technical_valid': valid, 'all_four_solved_and_plan_valid': solved,
                     'delta_E_B_minus_A': delta_e, 'delta_L_B_minus_A': delta_l, 'common_complete_window_all_four': common,
                     'first_subsequent_decision_divergence_step': seq_div, 'first_subsequent_full_scene_divergence_step': state_div,
                     'ordered_common_prefix_excluding_intervention_step': lcp, 'fixed_window_metrics': fixed_window,
                     'other_candidate_first_pop_step': opposite_pop, 'coalescence': co['status'],
                     'complete_efficiency': 'OBSERVED_FOR_THIS_EVENT' if solved and valid else 'UNKNOWN',
                     'H3': 'UNRESOLVED', 'optimality': 'UNKNOWN', 'coverage': 'SAME_PARENT_RELATION_FORM_COVERED_OPTIMAL_PATH_UNKNOWN' if eid.startswith('struct') else 'CROSS_PARENT_OPTIMAL_PATH_IDENTITY_UNKNOWN'})

save('repeatability_checks.json', REPEAT)
save('coalescence_checks.json', COALESCENCE)
save('event_results.json', ANALYSIS)
save('score_ledger_verification.json', LEDGER_CHECK)
fields = sorted({k for r in ROWS for k in r})
with (ROOT / 'results.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(ROWS)

text = ['# 重复性核查', '', '比较每个完整检查点的实际内容，包括队列物理布局、父指针、g/action、CLOSED、评分缓存和计数。时间截断长度差单列。', '']
for r in REPEAT:
    text += ['## '+r['event'], '', '```json', json.dumps(r, ensure_ascii=False, indent=2), '```', '']
(ROOT / 'repeatability_report.md').write_text('\n'.join(text))
text = ['# 完整汇合核查', '', '规则在运行前冻结。先匹配摘要，再核对全部实际内容；未匹配只表示已保存窗口内未观察到汇合。', '',
        '统计排除字段：'+', '.join(REG['coalescence']['excluded_cost_fields'])+'。保留所有父链接、缓存、有效与失效队列条目、插入序号及 serial。堆物理排列规范化为逻辑顺序。资源消耗另列；有限预算剩余额度不被宣称等价。', '']
for r in COALESCENCE:
    text += ['## '+r['event'], '', '```json', json.dumps(r, ensure_ascii=False, indent=2), '```', '']
(ROOT / 'coalescence_report.md').write_text('\n'.join(text))
text = ['# CP-DISR-Q2-PHASE-C-SINGLE-POP-V1 结果', '', '只在两份固定现场改变第一次出队，之后全部使用冻结 CPU/FP32 单状态 D0@820。', '',
        '| 事件 | 裁决 | A 新增展开 | B 新增展开 | ΔE | A/B 计划长 | ΔL |', '|---|---|---:|---:|---:|---|---:|']
for e in ANALYSIS:
    rr = {r['branch']: r for r in ROWS if r['event'] == e['event']}
    text.append('| {event} | {status} | {a} | {b} | {de} | {la}/{lb} | {dl} |'.format(**e, a=rr['A'].get('completed_additional_expansions', 'NA'), b=rr['B'].get('completed_additional_expansions', 'NA'), de=e.get('delta_E_B_minus_A'), la=rr['A'].get('total_plan_length'), lb=rr['B'].get('total_plan_length'), dl=e.get('delta_L_B_minus_A')))
text += ['', '逐事件边界：', '']
for e in ANALYSIS:
    text += ['- '+e['event']+'：共同完整窗口 '+str(e.get('common_complete_window_all_four'))+' 步；排除干预当步后的首次决策分歧 '+str(e.get('first_subsequent_decision_divergence_step'))+'；完整效率 '+e.get('complete_efficiency','UNKNOWN')+'；汇合 '+e.get('coalescence','UNKNOWN')+'。']
text += ['', 'p12 各支若截断，不把截断值代入真实总展开或加速比。Struct 的计划经过动作合同回放，最优性仍 UNKNOWN。A′/B′为技术重复，不能视为四个独立任务。', '',
         'H2-C 只按上述两个事件裁决；不支持 T1 整体优劣或监督来源因果。H3 保持 UNRESOLVED。Struct 同父关系形式已有方法覆盖，具体最优路径条件未知；p12 跨父关系缺少最优路径身份。', '',
         '训练、拟合、距离认证、GPU、完整图构建均为0。执行到本卡预算后停止，没有补跑、扩大面板、R4或新算法。', '',
         '详细数据：results.csv、event_results.json、repeatability_checks.json、coalescence_checks.json、branch_ledger.jsonl、branches/及plans/。']
(ROOT / 'research_decision.md').write_text('\n'.join(text)+'\n')
cpu = time.process_time()-START
save('analysis_receipt.json', {'role': 'saved_evidence_only', 'model_forwards': 0, 'search_runs': 0, 'cpu_seconds': cpu,
     'old_inputs_unchanged': RUN['old_inputs_unchanged'], 'server_checkout_unchanged': RUN['server_checkout_unchanged'], 'raw_ledger_valid': True})
print(json.dumps({'events': ANALYSIS, 'analysis_cpu_seconds': cpu}, ensure_ascii=False))
