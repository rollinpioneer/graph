"""One isolated, bounded ordinary-D0 continuation. No intervention or restart."""
import os
import sys
import time
CPU_ORIGIN = time.process_time()
WALL_ORIGIN = time.perf_counter()
os.sched_setaffinity(0, {int(sys.argv[3])})
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
sys.dont_write_bytecode = True
import gzip
import hashlib
import json
import math
import resource
import signal
import struct
import traceback
from pathlib import Path
import core_semantics as CORE
import log_codec as CODEC

ROOT, NAME = Path(sys.argv[1]), sys.argv[2]
assert NAME in ('A', 'B', 'B_repeat', 'A_repeat')
REG = json.loads((ROOT / 'preregistration.json').read_text())
DIR = ROOT / 'branches' / NAME
DIR.mkdir(parents=True, exist_ok=True)
assert not any((DIR / name).exists() for name in ('model_ledger.jsonl', 'score_bytes.bin', 'delta.jsonl.gz', 'receipt.json'))
GROUP = NAME[0]
LIMIT = 4 * 2**30
USED = 0
COUNT = 0
EVALUATIONS = 0
FAILED = 0
DEADLINE = float('inf')
INITIAL = COMPLETE = ST = None
ACTIVE = {'interruption_position': None}
LEDGER = (DIR / 'model_ledger.jsonl').open('xb', buffering=0)
RAW = (DIR / 'score_bytes.bin').open('xb', buffering=0)
LOCAL_OBSERVER = json.loads((ROOT / 'observed_initial.json').read_text())
REFS = json.loads((ROOT / 'legacy_suffix_reference.json').read_text())[NAME]
STARTUP = []
LOAD_WALL = LOAD_CPU = RESTORE_WALL = RESTORE_CPU = 0.0
SEARCH_START = SEARCH_CPU = None
LOG_WALL = LOG_CPU = 0.0


class Stop(Exception):
    pass


class Technical(Exception):
    pass


def size_refresh():
    global USED
    USED = sum(p.stat().st_size for p in DIR.rglob('*') if p.is_file())


def write_bytes(stream, data):
    global USED
    n = stream.write(data)
    USED += len(data) if n is None else n
    return n


def save(path, obj):
    global USED
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = CODEC.canonical_bytes(obj) + b'\n'
    old = p.stat().st_size if p.exists() else 0
    if USED - old + len(data) > LIMIT:
        raise Stop('BRANCH_DATA_LIMIT_DURING_FINAL_SAVE')
    with p.open('wb') as f:
        f.write(data)
    USED += len(data) - old


def emit(kind, **fields):
    write_bytes(LEDGER, CODEC.encoded_line({'kind': kind, 'branch': NAME,
                'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **fields}))


def soft_stop(reason):
    (ROOT / ('stop_' + GROUP)).write_text(reason + '\n')


def guard(forward=False, pending_bytes=0, scene=None):
    if (ROOT / 'stop_all').exists() or (ROOT / ('stop_' + GROUP)).exists():
        raise Stop('ORCHESTRATOR_SOFT_STOP')
    if time.perf_counter() >= DEADLINE:
        raise Stop('SEARCH_WALL_LIMIT')
    if forward and EVALUATIONS >= 40000:
        raise Stop('BRANCH_MODEL_STATE_LIMIT')
    if time.process_time() - CPU_ORIGIN >= 7650:
        raise Stop('BRANCH_CPU_SAFETY_LIMIT')
    reserve = max(256 * 2**20, CODEC.save_reserve_bytes(scene or COMPLETE)
                  if (scene or COMPLETE) is not None and pending_bytes else 0)
    if USED + pending_bytes + reserve > LIMIT:
        raise Stop('BRANCH_DATA_LIMIT')
    if pending_bytes:
        v = os.statvfs(DIR)
        if v.f_bavail * v.f_frsize < pending_bytes + reserve:
            raise Stop('FILESYSTEM_FREE_SPACE_LIMIT')


def alarm(signum, frame):
    raise Stop('SEARCH_WALL_LIMIT' if signum == signal.SIGALRM else 'BRANCH_CPU_SAFETY_LIMIT')


def note(key, score):
    old = LOCAL_OBSERVER.get(key)
    if old is not None and old != score:
        soft_stop('SAME_KEY_SCORE_BYTES_CHANGED')
        raise Technical('SAME_KEY_SCORE_BYTES_CHANGED')
    LOCAL_OBSERVER[key] = score


def raw_score(s):
    global EVALUATIONS, FAILED
    guard(forward=True)
    EVALUATIONS += 1
    key, offset = format(s, 'x'), RAW.tell()
    emit('MODEL_STATE_START', sequence=EVALUATIONS, cache_key=key, state=key,
         byte_offset=offset, input_graphs=1, model='D0@820', protocol=REG['protocol'])
    tw, tc = time.perf_counter(), time.process_time()
    try:
        signal.setitimer(signal.ITIMER_REAL, max(0.000001, DEADLINE-time.perf_counter()))
        try:
            with torch.inference_mode():
                value = EV.evaluate([s], CHECKS)[0]
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        score = CORE.bits(value)
        assert math.isfinite(value) and CORE.value_from_bits(score) == value
        write_bytes(RAW, bytes.fromhex(score))
        emit('MODEL_STATE_END', sequence=EVALUATIONS, cache_key=key, state=key,
             status='OK', value_bits_le=score, byte_offset=offset,
             wall_seconds=time.perf_counter()-tw, cpu_seconds=time.process_time()-tc)
        note(key, score)
    except Exception as error:
        FAILED += 1
        # A valid END already written before note failure must not get a second END.
        if RAW.tell() == offset:
            write_bytes(RAW, bytes(4))
            emit('MODEL_STATE_END', sequence=EVALUATIONS, cache_key=key, state=key,
                 status='INTERRUPTED' if isinstance(error, Stop) else 'ERROR',
                 invalid_placeholder=True, byte_offset=offset, error=repr(error),
                 wall_seconds=time.perf_counter()-tw, cpu_seconds=time.process_time()-tc)
        raise
    guard()
    return value


def search_score(st, states):
    vals = []
    st['pending_scores'] = [format(s, 'x') for s in states]
    for i, s in enumerate(states):
        ACTIVE['interruption_position'] = {'phase': 'score_successors',
            'pending_index': i, 'pending_count': len(states)}
        guard()
        key = format(s, 'x')
        if key in st['score_cache']:
            st['score_cache_hits'] += 1
            note(key, st['score_cache'][key])
            emit('CACHE_HIT', state=key)
        else:
            st['score_cache_misses'] += 1
            emit('CACHE_MISS', state=key)
            st['score_cache'][key] = CORE.bits(raw_score(s))
        vals.append(CORE.value_from_bits(st['score_cache'][key]))
    st['calls'] += 1
    st['pending_scores'] = []
    return vals


class CountedFile:
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        return write_bytes(self.stream, data)
    def flush(self):
        self.stream.flush()


def progress():
    save(DIR / 'progress.json', {'branch': NAME, 'pid': os.getpid(),
         'completed_additional_expansions': COUNT,
         'actual_model_state_evaluations': EVALUATIONS,
         'process_cpu_seconds': time.process_time()-CPU_ORIGIN,
         'search_and_logging_wall_seconds': time.perf_counter()-SEARCH_START if SEARCH_START else 0,
         'valid_OPEN': sum(row[3] for row in COMPLETE['heap']) if COMPLETE else None,
         'data_bytes': USED, 'status': COMPLETE['status'] if COMPLETE else 'LOADING'})


reason, error, partial, plan_valid, plan_length = None, None, None, None, None
search_end = time.perf_counter()
try:
    signal.signal(signal.SIGALRM, alarm)
    signal.signal(signal.SIGXCPU, alarm)
    resource.setrlimit(resource.RLIMIT_CPU, (7650, 7680))
    sys.path.insert(0, str(Path(REG['server_repository']) / 'src'))
    tl, cl = time.perf_counter(), time.process_time()
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    from cp_disr.pddl import task as T
    from cp_disr.pddl.compact import cli_ext as CE
    from cp_disr.pddl.search_match.budget import Budget
    from cp_disr.pddl.search_match.engine import SuccessorIndex
    info = REG['task']
    TASK = T.depots_task(info['domain_file'], info['file'])
    # This inner API timer is not the experiment budget. Our post-restore guard
    # and native-call alarm enforce the stricter registered search deadline.
    CHECKS = Budget(14000, 10**9, 8 * 2**30, 0)
    EV = CE.make_evaluator(REG['model_run_root'], 'D0@820', 'cpu')
    EV.model.eval()
    for parameter in EV.model.parameters():
        assert parameter.dtype == torch.float32 and parameter.device.type == 'cpu'
        parameter.requires_grad_(False)
    EV.prepare(TASK, {'domain': info['domain_file'], 'problem': info['file']}, CHECKS)
    INDEX = SuccessorIndex(TASK)
    LOAD_WALL, LOAD_CPU = time.perf_counter()-tl, time.process_time()-cl
    tr, cr = time.perf_counter(), time.process_time()
    INITIAL = json.loads((ROOT / 'checkpoints' / (NAME + '_108.json')).read_text())
    assert INITIAL['expanded'] == 110 and INITIAL['status'] == 'RUNNING'
    assert INITIAL['pending_scores'] == []
    assert INITIAL['task_sha256'] == info['sha256'] and INITIAL['domain_sha256'] == info['domain_sha256']
    assert INITIAL['scoring_protocol'] == REG['protocol']
    assert INITIAL['model_sha256'] == REG['model']['sha256_actual']
    for key, score in INITIAL['score_cache'].items():
        note(key, score)
    ST = CORE.restored(INITIAL)
    COMPLETE = INITIAL
    save(DIR / 'initial.json', INITIAL)
    CORE.REG, CORE.TASKS, CORE.INDICES = REG, {ST['task']: TASK}, {ST['task']: INDEX}
    CORE.ACTIVE, CORE.guard, CORE.search_score = ACTIVE, guard, search_score
    RESTORE_WALL, RESTORE_CPU = time.perf_counter()-tr, time.process_time()-cr
    size_refresh()
    SEARCH_START, SEARCH_CPU = time.perf_counter(), time.process_time()
    DEADLINE = SEARCH_START + 7200
    emit('BRANCH_START', forced_candidate=None, cpu_affinity=sorted(os.sched_getaffinity(0)),
         initial_semantic_sha256=CODEC.canonical_hash(INITIAL))
    with (DIR / 'delta.jsonl.gz').open('xb') as output:
        with gzip.GzipFile(fileobj=CountedFile(output), mode='wb', compresslevel=1, mtime=0) as log:
            while ST['status'] == 'RUNNING' and COUNT < 4892:
                guard()
                decision = CORE.ordinary_step(ST)
                guard()
                tlog, clog = time.perf_counter(), time.process_time()
                after = CORE.normalized(ST)
                record = CODEC.append(COMPLETE, after, decision)
                data = CODEC.encoded_line(record)
                ncount = COUNT + (1 if decision is not None else 0)
                checkpoint_data = CODEC.encoded_line(CODEC.checkpoint(after, ncount)) if ncount and ncount % 128 == 0 else b''
                guard(pending_bytes=len(data)+len(checkpoint_data), scene=after)
                if COUNT < len(REFS):
                    old = REFS[COUNT]
                    passed = old['before_sha256'] == record['before_sha256'] and old['decision'] == decision and old['after'] == after
                    STARTUP.append({'new_step': COUNT+1, 'old_event_after_step': 109+COUNT,
                                    'pass': passed, 'actual_after_sha256': record['after_sha256']})
                    if not passed:
                        soft_stop('LEGACY_SUFFIX_MISMATCH')
                        raise Technical('LEGACY_SUFFIX_MISMATCH')
                log.write(data)
                if checkpoint_data:
                    log.write(checkpoint_data)
                log.flush()
                COMPLETE, COUNT = after, ncount
                LOG_WALL += time.perf_counter()-tlog
                LOG_CPU += time.process_time()-clog
                search_end = time.perf_counter()
                if COUNT % 16 == 0:
                    size_refresh()
                    progress()
                guard()
    reason = COMPLETE['status'] if COMPLETE['status'] != 'RUNNING' else 'EXPANSION_LIMIT'
except Exception as exc:
    signal.setitimer(signal.ITIMER_REAL, 0)
    reason = str(exc) if isinstance(exc, Stop) else 'TECHNICALLY_INVALID'
    error = repr(exc)
    if not isinstance(exc, Stop):
        soft_stop(reason + ': ' + error)
        (DIR / 'error_trace.txt').write_text(traceback.format_exc())
    if ST is not None and COMPLETE is not None:
        live = CORE.normalized(ST)
        if live != COMPLETE:
            partial = {'scene': live, 'position': ACTIVE['interruption_position'],
                       'last_complete_sha256': CODEC.canonical_hash(COMPLETE),
                       'invalid_for_restore': True}
finally:
    signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGXCPU, signal.SIG_IGN)
    end = time.perf_counter()
    search_wall = end-SEARCH_START if SEARCH_START is not None else 0.0
    search_cpu = time.process_time()-SEARCH_CPU if SEARCH_CPU is not None else 0.0
    try:
        if COMPLETE is not None:
            save(DIR / 'final_complete.json', COMPLETE)
        if partial is not None:
            save(DIR / 'partial_interrupted.json', partial)
        save(DIR / 'startup_suffix_verify.json', STARTUP)
        if COMPLETE is not None and COMPLETE['status'] == 'SOLVED':
            indices = COMPLETE['plan']
            action_ids = [TASK.actions[i].aid for i in indices]
            final = TASK.replay(action_ids)
            plan_valid = final is not None and TASK.goal_satisfied(final)
            plan_length = len(indices)
            plan = {'indices': indices, 'action_ids': action_ids, 'legal_and_goal': plan_valid,
                    'final_state': None if final is None else format(final, 'x'), 'optimality': 'UNKNOWN'}
            plan['action_sequence_sha256'] = CODEC.canonical_hash(action_ids)
            # Plans are root summary data, independently included by the orchestrator quota.
            p = ROOT / 'plans' / (NAME + '.json')
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(CODEC.canonical_bytes(plan) + b'\n')
            if not plan_valid:
                reason = 'TECHNICALLY_INVALID'
                soft_stop('PLAN_CONTRACT_FAILURE')
        row = {'branch': NAME, 'pid': os.getpid(), 'cpu_affinity': sorted(os.sched_getaffinity(0)),
               'status': COMPLETE['status'] if COMPLETE else 'FAILED_BEFORE_SEARCH',
               'stop_reason': reason, 'error': error, 'completed_additional_expansions': COUNT,
               'event_after_expansions': 108+COUNT, 'task_total_expansions': 110+COUNT,
               'actual_model_state_evaluations': EVALUATIONS, 'failed_model_calls': FAILED,
               'plan_valid': plan_valid, 'total_plan_length': plan_length,
               'load_wall_seconds': LOAD_WALL, 'load_cpu_seconds': LOAD_CPU,
               'restore_wall_seconds': RESTORE_WALL, 'restore_cpu_seconds': RESTORE_CPU,
               'search_and_logging_wall_seconds': search_wall, 'search_and_logging_cpu_seconds': search_cpu,
               'logging_wall_seconds': LOG_WALL, 'logging_cpu_seconds': LOG_CPU,
               'last_complete_search_end_seconds': search_end-SEARCH_START if SEARCH_START else 0,
               'deadline_detection_latency_seconds': max(0, search_wall-7200),
               'partial_expansion_saved': partial is not None,
               'initial_semantic_sha256': CODEC.canonical_hash(INITIAL) if INITIAL else None,
               'final_complete_sha256': CODEC.canonical_hash(COMPLETE) if COMPLETE else None,
               'valid_OPEN': sum(x[3] for x in COMPLETE['heap']) if COMPLETE else None,
               'process_cpu_seconds': time.process_time()-CPU_ORIGIN,
               'process_wall_seconds': time.perf_counter()-WALL_ORIGIN,
               'cleanup_wall_seconds': time.perf_counter()-end,
               'GPU_device_seconds': 0, 'training': 0, 'fitting': 0, 'certification': 0,
               'startup_suffix_pass': all(x['pass'] for x in STARTUP) and len(STARTUP) == min(COUNT, len(REFS))}
        if COMPLETE is not None:
            for key in ('generated', 'duplicates', 'path_updates', 'calls', 'score_cache_hits', 'score_cache_misses'):
                row['new_' + key] = COMPLETE[key] - INITIAL[key]
        emit('BRANCH_END', **{k: v for k, v in row.items() if k != 'branch'})
        RAW.close()
        LEDGER.close()
        size_refresh()
        row['data_bytes_before_receipt'] = USED
        save(DIR / 'receipt.json', row)
        print(json.dumps(row), flush=True)
    except Exception:
        soft_stop('FINAL_SAVE_FAILURE')
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        raise
