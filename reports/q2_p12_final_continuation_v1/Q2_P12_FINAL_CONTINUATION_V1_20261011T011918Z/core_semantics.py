"""Frozen phase-C normalization and ordinary step. No forced-candidate entry point."""
import copy, hashlib, heapq, json, struct

def bits(v):
    return struct.pack('<f', v).hex()

def value_from_bits(h):
    return struct.unpack('<f', bytes.fromhex(h))[0]

def normalized(st):
    obj = copy.deepcopy(st)
    obj['heap'] = [[bits(h), serial, format(s, 'x'), s not in st['closed']] for h, serial, s in st['heap']]
    obj['closed'] = [format(s, 'x') for s in sorted(st['closed'])]
    obj['node'] = [[format(s, 'x'), g, None if p is None else format(p, 'x'), a] for s, (g, p, a) in sorted(st['node'].items())]
    return obj

def restored(obj):
    st = copy.deepcopy(obj)
    st['heap'] = [(value_from_bits(h), ser, int(s, 16)) for h, ser, s, valid in obj['heap']]
    st['closed'] = {int(s, 16) for s in obj['closed']}
    st['node'] = {int(s, 16): (g, None if p is None else int(p, 16), a) for s, g, p, a in obj['node']}
    assert normalized(st) == obj and (not st['pending_scores'])
    assert all((st['heap'][(i - 1) // 2] <= st['heap'][i] for i in range(1, len(st['heap']))))
    return st

def obj_digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def future_scene(obj):
    out = copy.deepcopy(obj)
    for field in REG['coalescence']['excluded_cost_fields']:
        out.pop(field)
    out['heap'] = sorted(out['heap'], key=lambda e: (value_from_bits(e[0]), e[1], int(e[2], 16)))
    return out

def ordinary_step(st):
    task, index = (TASKS[st['task']], INDICES[st['task']])
    ACTIVE['interruption_position'] = {'phase': 'pop'}
    guard()
    while st['heap']:
        guard()
        h, serial, s = heapq.heappop(st['heap'])
        if s not in st['closed']:
            break
    else:
        st['status'] = 'OPEN_EXHAUSTED'
        return None
    st['closed'].add(s)
    st['expanded'] += 1
    g = st['node'][s][0] + 1
    pending = []
    decision = {'expanded': st['expanded'], 'state': format(s, 'x'), 'score_bits': bits(h), 'serial': serial}
    for j, a in enumerate(index.applicable(s)):
        ACTIVE['interruption_position'] = {'phase': 'generate_successor', 'action_position': j, 'action_index': a.index, 'popped': decision}
        guard()
        t = s & ~a.del_mask | a.add_mask
        st['generated'] += 1
        if task.solvable_by_relaxation and t & task.goal_mask == task.goal_mask:
            st['node'][t] = (g, s, a.index)
            acts, p = ([], t)
            while st['node'][p][1] is not None:
                _, p0, act = st['node'][p]
                acts.append(act)
                p = p0
            st['plan'] = list(reversed(acts))
            st['status'] = 'SOLVED'
            return decision
        if t in st['closed']:
            st['duplicates'] += 1
            continue
        old = st['node'].get(t)
        if old is None:
            st['node'][t] = (g, s, a.index)
            pending.append(t)
        elif g < old[0]:
            st['node'][t] = (g, s, a.index)
            st['path_updates'] += 1
        else:
            st['duplicates'] += 1
    if pending:
        vals = search_score(st, pending)
        for j, (t, v) in enumerate(zip(pending, vals)):
            ACTIVE['interruption_position'] = {'phase': 'enqueue_successors', 'enqueue_index': j, 'count': len(pending)}
            guard()
            heapq.heappush(st['heap'], (v, st['serial'], t))
            st['serial'] += 1
            if v < st['best_h']:
                st['best_h'] = v
    return decision
