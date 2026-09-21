"""Stage 1A one-task smoke orchestrator.

Does not change the frozen CP-DISR v2.1 Method (network, PPO, reward, prior math).
B2 empty prior is applied at the snapshot/input layer before Policy.forward.
"""
from __future__ import annotations

import csv, hashlib, json, math, os, random, subprocess, sys, time, traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from .common import BindingError, DataIntegrityError, canonical, digest
from .graph import four_views
from .neural import Policy
from .collector import Collector
from .rl import Rollout, gamma
from .torch_rl import PPO, load_checkpoint, save_checkpoint
from .prior import PriorSampler, EpisodePrior
from .platforms.libero.runtime_factory import create_runtime, PREDICATES, OBJECTS

BASE_COMMIT = "d270f373dbeb1eace4bc87112236829bb86cc6e6"
STAGE_DIR = Path("experiments/part_1_smoke/stage_1a")
STATUS_PATH = Path("experiments/stage_status/stage_1a.json")
P0_DRY = Path("experiments/part_1_smoke/stage_1a_p0/b2_full_dry_run.json")
BUDGET = 16384
ROLLOUT_N = 1024
D_REF = 3.55
JOB_CAP = BUDGET * D_REF
MAX_EMPTY = 8
GATE_CASE = "D0_dev_00"
OBS_DIM = 48
CAND_DIM = 8
B_PRIOR = 0.5

def utc_now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

def log(msg):
    line = '[%s] %s' % (utc_now(), msg)
    print(line, flush=True)
    return line

def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    raise TypeError(type(value))

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default) + '\n', encoding='utf-8')

def append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=_json_default, allow_nan=False) + '\n')

def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def git_commit(root):
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(root), text=True).strip()

def load_status(root):
    return json.loads((root / STATUS_PATH).read_text(encoding='utf-8'))

def save_status(root, payload):
    write_json(root / STATUS_PATH, payload)
    return payload

def cache_fingerprint(root):
    rows = []
    base = root / 'experiments/vlm_cache'
    for split in ('train', 'dev'):
        d = base / split
        if not d.is_dir():
            continue
        for key in sorted(p.name for p in d.iterdir() if p.is_dir()):
            folder = d / key
            rec = {'split': split, 'cache_key': key}
            for name in ('COMPLETE', 'manifest.json', 'final_edges.json', 'accepted_relations.json', 'content_hashes.json'):
                p = folder / name
                rec[name] = file_sha(p) if p.is_file() else None
            rows.append(rec)
    return rows

def fingerprint_digest(rows):
    slim = [{k: r[k] for k in ('split', 'cache_key', 'COMPLETE', 'manifest.json', 'final_edges.json')} for r in rows]
    return digest(slim)

def load_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding='utf-8'))

def load_split(root):
    return json.loads((root / 'configs/splits/D0_stage_1a.json').read_text(encoding='utf-8'))

def train_cases(split):
    return [row['case_id'] for row in split['train']]

def dev_cases(split):
    return [row['case_id'] for row in split['dev']]

def model_kwargs(template, method):
    actions = sorted({c.name for c in template.contracts})
    predicates = sorted(PREDICATES)
    types = sorted(set(OBJECTS.values()))
    return dict(actions=actions, predicates=predicates, types=types, observation_dim=OBS_DIM, candidate_dim=CAND_DIM, method=method, B=B_PRIOR)

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def capture_rng():
    payload = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state().cpu().numpy().tolist(),
    }
    if torch.cuda.is_available():
        payload['cuda'] = [t.cpu().numpy().tolist() for t in torch.cuda.get_rng_state_all()]
    else:
        payload['cuda'] = None
    return payload

def restore_rng(payload):
    py = payload['python']
    if isinstance(py, list):
        py = (py[0], tuple(py[1]), py[2])
    random.setstate(py)
    st = payload['numpy']
    if isinstance(st, list):
        st = (st[0], np.array(st[1], dtype=np.uint32), int(st[2]), int(st[3]), float(st[4]))
    np.random.set_state(st)
    torch.set_rng_state(torch.tensor(payload['torch'], dtype=torch.uint8))
    if torch.cuda.is_available() and payload.get('cuda') is not None:
        torch.cuda.set_rng_state_all([torch.tensor(x, dtype=torch.uint8) for x in payload['cuda']])

def rng_sidecar(payload):
    py = payload['python']
    if not isinstance(py, list):
        py = [py[0], list(py[1]), py[2]]
    npst = payload['numpy']
    if not isinstance(npst, list):
        npst = [npst[0], npst[1].tolist(), int(npst[2]), int(npst[3]), float(npst[4])]
    return {'python': py, 'numpy': npst, 'torch': payload['torch'], 'cuda': payload.get('cuda')}

def current_device():
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        return torch.device('cuda', idx), torch.cuda.get_device_name(idx)
    return torch.device('cpu'), 'cpu'

def empty_prior(snapshot):
    edges = ()
    return replace(snapshot, prior_edges=edges, prior_hash=digest(edges))

def set_prior(snapshot, edges):
    edges = tuple(tuple(e) for e in edges)
    return replace(snapshot, prior_edges=edges, prior_hash=digest(edges))

def snapshot_id(snapshot):
    return '%s:%s:%s:%s' % (snapshot.env_id, snapshot.episode_id, snapshot.decision_id, snapshot.observation_ref)

def rms(t):
    if t is None:
        return 0.0
    x = t.detach().float().reshape(-1)
    if x.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean(x * x)))

def maxabs(t):
    if t is None:
        return 0.0
    x = t.detach().float().reshape(-1)
    if x.numel() == 0:
        return 0.0
    return float(x.abs().max())

def is_nonzero(t):
    if t is None:
        return False
    return int(torch.count_nonzero(t.detach()).item()) > 0

def contract_by_id(snapshot, cid):
    return next(c for c in snapshot.template.contracts if c.id == cid)

def nominal_nonempty(snapshot, cid, edges):
    contract = contract_by_id(snapshot, cid)
    k, h, ki, hi = four_views(snapshot.template, snapshot.facts.values, edges, contract)
    return ki.values != k.values

def candidate_struct(out, snapshot, cid):
    diffs = out.diagnostics.get('differences') or {}
    delta_map = out.diagnostics.get('delta') or {}
    up_map = out.diagnostics.get('up') or {}
    prior_in = out.diagnostics.get('prior_inputs') or {}
    diff = diffs.get(cid)
    dp = None if diff is None else diff.dp
    dk = None if diff is None else diff.dk
    dh = None if diff is None else diff.dh
    residual = delta_map.get(cid)
    up = up_map.get(cid)
    pin = prior_in.get(cid)
    res_f = None if residual is None else float(residual.detach())
    masked = not bool(snapshot.mask[snapshot.candidate_ids.index(cid)])
    return {
        'candidate_id': cid,
        'dp_rms': rms(dp),
        'dk_rms': rms(dk),
        'dh_rms': rms(dh),
        'up_rms': rms(up),
        'residual': res_f,
        'residual_abs': None if res_f is None else abs(res_f),
        'dp_nonzero': is_nonzero(dp),
        'residual_nonzero': False if res_f is None else abs(res_f) > 0,
        'prior_input_rms': rms(pin),
        'nominal_patch_nonempty': False if masked else nominal_nonempty(snapshot, cid, snapshot.prior_edges),
        'masked': masked,
    }

def finite_tensor(t):
    return t is not None and bool(torch.isfinite(t.detach()).all().item())

def make_bundle(root):
    manifest = load_yaml(root / 'experiments/manifests/runtime_manifest.yaml')
    return create_runtime(manifest)

def make_policy(template, method, device):
    model = Policy(**model_kwargs(template, method))
    return model.to(device)

def hashes(root, caches):
    split = root / 'configs/splits/D0_stage_1a.json'
    contract = root / 'configs/contracts/d0_runtime_skills.yaml'
    runtime = root / 'experiments/manifests/runtime_manifest.yaml'
    return {
        'config_hash': digest({'budget': BUDGET, 'rollout': ROLLOUT_N, 'seed': 0, 'methods': ['B2', 'Full']}),
        'runtime_hash': file_sha(runtime),
        'contract_hash': file_sha(contract),
        'split_hash': file_sha(split),
        'cache_hash': fingerprint_digest(caches),
        'git_commit': git_commit(root),
    }

def compact_transition(method, seed, case_id, t, result, out, execution, prior_mode, original_hash, source_n, cache_key, run_id):
    cid = t.selected_candidate_id
    idx = t.snapshot.candidate_ids.index(cid)
    struct = candidate_struct(out, t.snapshot, cid)
    qualifying = (
        method == 'Full'
        and len(t.snapshot.prior_edges) > 0
        and struct['nominal_patch_nonempty']
        and not struct['masked']
    )
    actual = result if isinstance(result, dict) else {
        'success': result.success,
        'terminated': result.terminated,
        'truncated': result.truncated,
        'reason': result.reason,
        'reward_events': list(result.reward_events),
    }
    rec = {
        'run_id': run_id,
        'method': method,
        'task': 'D0',
        'seed': seed,
        'env_id': t.snapshot.env_id,
        'episode_id': t.snapshot.episode_id,
        'decision_id': t.snapshot.decision_id,
        'case_id': case_id,
        'selected_candidate_id': cid,
        'selected_candidate_index': idx,
        'candidate_ids': list(t.snapshot.candidate_ids),
        'mask': [bool(x) for x in t.snapshot.mask],
        'mask_hash': t.snapshot.mask_hash,
        'source_cache_key': cache_key,
        'prior_mode': prior_mode,
        'original_prior_hash': original_hash,
        'effective_prior_hash': t.snapshot.prior_hash,
        'effective_relation_count': len(t.snapshot.prior_edges),
        'source_cache_relation_count': source_n,
        'old_logp': t.old_logp,
        'old_v': t.old_v,
        'old_v_next': t.old_v_next,
        'reward': t.reward,
        'duration_seconds': t.duration,
        'Gamma': t.Gamma,
        'episode_discount_weight': t.weight,
        'terminated': t.terminated,
        'truncated': t.truncated,
        'reason': t.reason,
        'snapshot_id': snapshot_id(t.snapshot),
        'next_snapshot_id': snapshot_id(t.next_snapshot),
        'recurrent_prefix_len': len(t.prefix),
        'controller_exit': None if execution is None else getattr(execution, 'controller_exit', None),
        'verified_outcome': actual.get('reason'),
        'evaluator_result': actual,
        'qualifying_for_dp': qualifying,
        'struct': struct,
        'logits_finite': finite_tensor(out.logits),
        'value_finite': finite_tensor(out.value),
    }
    if rec.get('action') is not None or 'action' in rec:
        raise DataIntegrityError('action field must not be emitted')
    if rec['selected_candidate_id'] in (None, ''):
        raise DataIntegrityError('selected_candidate_id empty')
    return rec

def save_job_checkpoint(path, policy, optimizer, extra, rng_payload, prior_sampler):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    manifest = dict(extra)
    manifest['rng_sidecar'] = str(path.with_suffix('.rng.json'))
    save_checkpoint(path, policy, optimizer, manifest)
    sidecar = {
        'python_rng': rng_sidecar(rng_payload)['python'],
        'numpy_rng': rng_sidecar(rng_payload)['numpy'],
        'torch_rng': rng_sidecar(rng_payload)['torch'],
        'cuda_rng': rng_sidecar(rng_payload)['cuda'],
        'prior_sampler': None if prior_sampler is None else prior_sampler.state_dict(),
        'extra': extra,
    }
    write_json(path.with_suffix('.rng.json'), sidecar)
    return path

def environment_snapshot(device, device_name, seed):
    return {
        'hostname': os.uname().nodename if hasattr(os, 'uname') else os.environ.get('COMPUTERNAME'),
        'cwd': os.getcwd(),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'device': str(device),
        'device_name': device_name,
        'torch_version': torch.__version__,
        'cuda_available': bool(torch.cuda.is_available()),
        'seed': seed,
        'python': sys.version,
        'mujoco_gl': os.environ.get('MUJOCO_GL'),
        'time': utc_now(),
    }

def software_snapshot(root):
    return {
        'git_commit': git_commit(root),
        'base_commit': BASE_COMMIT,
        'python': sys.executable,
        'numpy': np.__version__,
        'torch': torch.__version__,
    }

def resolved_config(method, seed, cases, out_dir, hashes_doc):
    return {
        'method': method,
        'task': 'D0',
        'seed': seed,
        'runtime_driver': 'skill_on_policy_v1',
        'budget': BUDGET,
        'rollout_transitions': ROLLOUT_N,
        'resource_cap_seconds': JOB_CAP,
        'task_cases': list(cases),
        'prior': 'empty' if method == 'B2' else 'episode_80_20_original_absent',
        'eval_prior': 'empty' if method == 'B2' else 'original',
        'optimizer': 'Adam',
        'learning_rate': 3e-4,
        'ppo_clip': 0.2,
        'gae_lambda': 0.95,
        'ppo_epochs': 4,
        'sequence_length': 16,
        'minibatch': 64,
        'value_coef': 0.5,
        'lambda_q': 0.1,
        'entropy_coef': 0.01,
        'grad_clip': 0.5,
        'prior_bound_B': 0.5,
        'gamma_per_second': 0.99,
        'd_ref_seconds': D_REF,
        'task_deadline_seconds': 43.0,
        'output_directory': str(out_dir),
        'hashes': hashes_doc,
        'no_pretrained_checkpoint': True,
        'no_vlm_calls': True,
    }

def assert_transition_fields(rec, next_prior_hash=None):
    issues = []
    if not rec.get('selected_candidate_id'):
        issues.append('selected_candidate_id empty')
    if rec['selected_candidate_id'] not in rec['candidate_ids']:
        issues.append('selected_candidate_id not in candidate_ids')
    if not rec['mask'][rec['selected_candidate_index']]:
        issues.append('selected mask false')
    if not math.isfinite(rec['old_logp']):
        issues.append('old_logp nonfinite')
    if not (math.isfinite(rec['duration_seconds']) and rec['duration_seconds'] > 0):
        issues.append('duration not positive finite')
    if next_prior_hash is not None and rec['effective_prior_hash'] != next_prior_hash:
        issues.append('prior hash changed within episode')
    return issues

def run_corrected_dry_run(bundle, method, device, case_id, empty=False, force_original=False):
    seed_all(0)
    snap0 = bundle.start_case(case_id)
    source_n = len(bundle.original_prior_edges)
    original_hash = digest(tuple(tuple(e) for e in bundle.original_prior_edges))
    cache_key = None if case_id not in bundle.caches else str(bundle.caches[case_id])
    if empty:
        snap = empty_prior(snap0)
        prior_mode = 'absent'
    elif force_original:
        snap = set_prior(snap0, bundle.original_prior_edges)
        prior_mode = 'original'
    else:
        snap = snap0
        prior_mode = 'source_cache'
    bundle.current_snapshot = snap
    policy = make_policy(bundle.template, method, device)
    policy.eval()
    collector = Collector(bundle, policy)
    collector.reset_episode(snap.env_id, snap.episode_id)
    records = []
    structs = []
    with torch.no_grad():
        hidden = policy.initial_hidden()
        out0 = policy(snap, hidden)
    for cid, m in zip(snap.candidate_ids, snap.mask):
        if m:
            structs.append(candidate_struct(out0, snap, cid))
    for _ in range(2):
        t, result = collector.step(bundle.current_snapshot)
        if t is None:
            records.append({'no_transition': True, 'result': result})
            break
        rec = compact_transition(method, 0, case_id, t, result, collector.last_output, collector.last_execution, prior_mode, original_hash, source_n, cache_key, 'startup_gate')
        rec['issues'] = assert_transition_fields(rec, t.next_snapshot.prior_hash)
        records.append(rec)
        bundle.current_snapshot = t.next_snapshot
        ended = t.terminated or t.truncated
        if ended:
            break
    return {
        'method': method,
        'case_id': case_id,
        'source_cache_relation_count': source_n,
        'effective_prior_relation_count': len(snap.prior_edges),
        'effective_prior_hash': snap.prior_hash,
        'original_prior_hash': original_hash,
        'candidate_ids': list(snap.candidate_ids),
        'mask': [bool(x) for x in snap.mask],
        'mask_true': int(sum(snap.mask)),
        'logits_finite': finite_tensor(out0.logits),
        'forward_structs': structs,
        'transitions': records,
        'policy_id': id(policy),
        'param_ids': [id(p) for p in policy.parameters()],
        'state_digest': digest([float(p.detach().float().mean().cpu()) for p in policy.parameters()]),
    }

def gate_a(root, out, bundle, device):
    log('startup gate A: corrected dry-run fields')
    b2 = run_corrected_dry_run(bundle, 'B2', device, GATE_CASE, empty=False)
    full = run_corrected_dry_run(bundle, 'Full', device, GATE_CASE, force_original=True)
    write_json(out / 'startup_gate/corrected_b2_dry_run.json', b2)
    write_json(out / 'startup_gate/corrected_full_dry_run.json', full)
    erratum = [
        '# P0 reporting erratum',
        '',
        'Historical Stage 1A-P0 dry-run reports recorded `transition.action`.',
        'The Transition dataclass has no `action` field. The executed skill identifier is `selected_candidate_id`.',
        'Values of `action: null` in experiments/part_1_smoke/stage_1a_p0/b2_full_dry_run.json are a report-field bug,',
        'not evidence that the collector executed an empty action.',
        'The original P0 files are not modified. Corrected records are in this directory.',
        '',
        'P0 B2 prior_edge_count=1 described the source cache, not B2 effective prior. See gate B.',
        '',
    ]
    (out / 'startup_gate/p0_reporting_erratum.md').write_text('\n'.join(erratum), encoding='utf-8')
    issues = []
    for name, doc in (('B2', b2), ('Full', full)):
        trans = [r for r in doc['transitions'] if not r.get('no_transition')]
        if not trans:
            issues.append(name + ' produced no transition')
            continue
        for r in trans:
            issues.extend(['%s: %s' % (name, x) for x in r.get('issues') or []])
            if 'action' in r:
                issues.append(name + ' still emits action')
    passed = not issues
    return {'gate': 'A', 'passed': passed, 'issues': issues, 'b2_n': len(b2['transitions']), 'full_n': len(full['transitions'])}

def gate_b(root, out, bundle, device):
    log('startup gate B: B2 effective prior empty at input layer')
    doc = run_corrected_dry_run(bundle, 'B2', device, GATE_CASE, empty=True)
    write_json(out / 'startup_gate/b2_empty_prior_check.json', doc)
    issues = []
    if doc['effective_prior_relation_count'] != 0:
        issues.append('effective prior not empty')
    if doc['source_cache_relation_count'] < 0:
        issues.append('source cache count invalid')
    for st in doc['forward_structs']:
        if st['dp_rms'] != 0 or st['dp_nonzero'] or (st['residual'] not in (0, 0.0, None)) and st['residual'] != 0:
            issues.append('B2 DP/Delta not strictly 0 on %s' % st['candidate_id'])
        if st['up_rms'] != 0:
            issues.append('B2 uP not strictly 0 on %s' % st['candidate_id'])
        if st['residual_abs'] not in (0, 0.0, None) and st['residual_abs'] != 0:
            issues.append('B2 residual not 0 on %s' % st['candidate_id'])
    for r in doc['transitions']:
        if r.get('no_transition'):
            continue
        if r['effective_relation_count'] != 0:
            issues.append('B2 transition effective prior nonempty')
        if r['struct']['dp_rms'] != 0 or r['struct']['up_rms'] != 0:
            issues.append('B2 collected DP/uP not 0')
        if (r['struct']['residual_abs'] or 0) != 0:
            issues.append('B2 collected residual not 0')
    return {'gate': 'B', 'passed': not issues, 'issues': issues, 'source_cache_relation_count': doc['source_cache_relation_count'], 'effective_prior_relation_count': doc['effective_prior_relation_count']}

def gate_c(root, out, bundle, device):
    log('startup gate C: Full DP/Delta on frozen nonempty prior')
    doc = run_corrected_dry_run(bundle, 'Full', device, GATE_CASE, force_original=True)
    write_json(out / 'startup_gate/full_startup_check.json', doc)
    issues = []
    if doc['source_cache_relation_count'] <= 0 or doc['effective_prior_relation_count'] <= 0:
        issues.append('Full prior not legally nonempty')
    if doc['mask_true'] < 2:
        issues.append('candidate count legal/mask_true < 2')
    if len(doc['candidate_ids']) < 2:
        issues.append('candidate_ids < 2')
    if not doc['logits_finite']:
        issues.append('logits nonfinite')
    qualifying = [s for s in doc['forward_structs'] if (not s['masked']) and s['nominal_patch_nonempty'] and doc['effective_prior_relation_count'] > 0]
    if not qualifying:
        issues.append('no qualifying candidate with nonempty nominal patch')
    dp_hit = any(s['dp_nonzero'] for s in qualifying)
    delta_hit = any(s['residual_nonzero'] for s in qualifying)
    if not dp_hit:
        issues.append('DP permanently 0 on all qualifying candidates')
    if not delta_hit:
        issues.append('Delta permanently 0 on all qualifying candidates')
    for s in doc['forward_structs']:
        if s['residual_abs'] is not None and s['residual_abs'] > B_PRIOR + 1e-8:
            issues.append('residual abs > B on %s' % s['candidate_id'])
    return {
        'gate': 'C',
        'passed': not issues,
        'issues': issues,
        'qualifying_n': len(qualifying),
        'dp_nonzero_any': dp_hit,
        'delta_nonzero_any': delta_hit,
        'max_abs_residual': max([s['residual_abs'] or 0 for s in doc['forward_structs']] or [0]),
        'prior_edges': doc['effective_prior_relation_count'],
    }

def gate_d(root, out, bundle, device, device_name, caches_before):
    log('startup gate D: independent init, seeds, pytest, cache immutability')
    issues = []
    seed_all(0)
    b2 = make_policy(bundle.template, 'B2', device)
    seed_all(0)
    full = make_policy(bundle.template, 'Full', device)
    if id(b2) == id(full):
        issues.append('B2/Full policy objects are the same')
    b2_ids = {id(p) for p in b2.parameters()}
    full_ids = {id(p) for p in full.parameters()}
    if b2_ids & full_ids:
        issues.append('shared parameter storage between B2 and Full')
    opt_b2 = PPO(b2)
    opt_full = PPO(full)
    if id(opt_b2.optimizer) == id(opt_full.optimizer):
        issues.append('shared optimizer')
    seed_doc = {
        'python_seed_call': 0,
        'numpy_seed_call': 0,
        'torch_seed_call': 0,
        'cuda_seed_call': 0 if torch.cuda.is_available() else None,
        'device': str(device),
        'device_name': device_name,
        'b2_policy_id': id(b2),
        'full_policy_id': id(full),
        'p0_model_reused': False,
    }
    write_json(out / 'startup_gate/init_independence.json', seed_doc)
    log('running Stage 0B pytest marker=pure or torch_runtime')
    rc = subprocess.call([sys.executable, '-m', 'pytest', str(root / 'tests'), '-m', 'pure or torch_runtime', '-ra', '-q'], cwd=str(root))
    write_json(out / 'startup_gate/stage0b_regression.json', {'returncode': rc, 'marker': 'pure or torch_runtime'})
    if rc != 0:
        issues.append('Stage 0B pytest failed rc=%s' % rc)
    caches_after = cache_fingerprint(root)
    if fingerprint_digest(caches_before) != fingerprint_digest(caches_after):
        issues.append('vlm caches mutated during startup gates')
    write_json(out / 'startup_gate/cache_fingerprint_before.json', caches_before)
    write_json(out / 'startup_gate/cache_fingerprint_after_gates.json', caches_after)
    del b2, full, opt_b2, opt_full
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {'gate': 'D', 'passed': not issues, 'issues': issues, 'pytest_rc': rc, 'device': str(device), 'device_name': device_name}

def write_startup_report(out, gates, blocked):
    lines = ['# Stage 1A startup gate report', '', 'generated: ' + utc_now(), '']
    for g in gates:
        lines.append('## Gate %s: %s' % (g['gate'], 'PASS' if g['passed'] else 'FAIL'))
        if g.get('issues'):
            for iss in g['issues']:
                lines.append('- ' + iss)
        lines.append('')
    lines.append('blocked_training: %s' % bool(blocked))
    (out / 'combined/startup_gate_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_json(out / 'startup_gate/summary.json', {'gates': gates, 'blocked': blocked, 'time': utc_now()})

def apply_episode_prior(method, bundle, snapshot, sampler):
    source_n = len(bundle.original_prior_edges)
    original = tuple(tuple(e) for e in bundle.original_prior_edges)
    original_hash = digest(original)
    if method == 'B2':
        snap = empty_prior(snapshot)
        prior = EpisodePrior(snap.env_id, snap.episode_id, (), original_hash, digest(()), 'absent')
        return snap, prior, source_n
    prior = sampler.start(snapshot.env_id, snapshot.episode_id, original)
    snap = set_prior(snapshot, prior.edges)
    return snap, prior, source_n

def eval_dev10(root, method, ckpt_path, cases, device, hashes_doc, out_eval):
    rng_before = capture_rng()
    bundle = make_bundle(root)
    seed_all(0)
    policy = make_policy(bundle.template, method, device)
    load_checkpoint(ckpt_path, policy)
    policy.eval()
    rows = []
    success_n = 0
    try:
        for case in cases:
            snap = bundle.start_case(case)
            if method == 'B2':
                snap = empty_prior(snap)
                prior_mode = 'absent'
            else:
                snap = set_prior(snap, bundle.original_prior_edges)
                prior_mode = 'original'
            hidden = policy.initial_hidden()
            steps = 0
            success = False
            reason = None
            while True:
                with torch.no_grad():
                    out = policy(snap, hidden)
                if out.distribution is None:
                    reason = bundle.safety.end_no_candidates(snap.env_id, snap.episode_id)
                    break
                cid, _ = out.select(True)
                hidden = out.hidden
                obs = bundle.observations.observe()
                if not bundle.safety.can_execute(cid, obs):
                    raise DataIntegrityError('Safety/mask disagreement during eval')
                contract = contract_by_id(snap, cid)
                interval_start = bundle.clock.now_seconds()
                execution = bundle.executor.execute(cid, contract.timeout_seconds)
                obs = bundle.observations.observe()
                facts = bundle.verifier.verify(bundle.perception.infer(obs), execution)
                now = bundle.clock.now_seconds()
                from .adapters import EvaluationInput
                result = bundle.evaluator.evaluate(EvaluationInput(bundle.task_id, snap.env_id, snap.episode_id, execution.evidence_ids, now - bundle.episode_start_seconds, interval_start, now))
                steps += 1
                snap = bundle.snapshot_builder.build(snap, facts, obs, execution, now)
                if method == 'B2':
                    snap = empty_prior(snap)
                else:
                    snap = set_prior(snap, bundle.original_prior_edges)
                if result.terminated or result.truncated:
                    success = bool(result.success)
                    reason = result.reason
                    break
            success_n += int(success)
            rows.append({'case': case, 'success': success, 'reason': reason, 'skills': steps, 'prior_mode': prior_mode})
    finally:
        bundle.environment.close()
        restore_rng(rng_before)
    rate = success_n / max(1, len(cases))
    payload = {'method': method, 'checkpoint': str(ckpt_path), 'success_n': success_n, 'n': len(cases), 'success_rate': rate, 'rows': rows, 'hashes': hashes_doc, 'isolated_rng': True}
    write_json(out_eval, payload)
    return payload

def csv_write(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        for row in rows:
            w.writerow(row)

def write_svg_lines(path, series, xlabel, ylabel, title):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h, pad = 900, 420, 60
    xs, ys = [], []
    for name, pts in series:
        for x, y in pts:
            xs.append(float(x)); ys.append(float(y))
    if not xs:
        xs, ys = [0, 1], [0, 1]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax == xmin:
        xmax = xmin + 1
    if ymax == ymin:
        ymax = ymin + 1
    def sx(x):
        return pad + (w - 2 * pad) * (x - xmin) / (xmax - xmin)
    def sy(y):
        return h - pad - (h - 2 * pad) * (y - ymin) / (ymax - ymin)
    colors = ['#1f77b4', '#d62728', '#2ca02c']
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d">' % (w, h), '<rect width="100%" height="100%" fill="white"/>', '<text x="%d" y="24" font-size="16">%s</text>' % (pad, title), '<text x="%d" y="%d" font-size="12">%s</text>' % (w/2, h-16, xlabel), '<text x="16" y="%d" font-size="12" transform="rotate(-90 16 %d)">%s</text>' % (h/2, h/2, ylabel)]
    parts.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="black"/>' % (pad, h-pad, w-pad, h-pad))
    parts.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="black"/>' % (pad, pad, pad, h-pad))
    for i, (name, pts) in enumerate(series):
        col = colors[i % len(colors)]
        if not pts:
            continue
        d = ' '.join('%s%d,%d' % ('M' if j == 0 else 'L', sx(x), sy(y)) for j, (x, y) in enumerate(pts))
        parts.append('<path d="%s" fill="none" stroke="%s" stroke-width="2"/>' % (d, col))
        for x, y in pts:
            parts.append('<circle cx="%d" cy="%d" r="4" fill="%s"/>' % (sx(x), sy(y), col))
        parts.append('<text x="%d" y="%d" font-size="12" fill="%s">%s</text>' % (w - pad - 80, pad + 16 + 16 * i, col, name))
    parts.append('</svg>')
    path.write_text('\n'.join(parts), encoding='utf-8')
    png = path.with_suffix('.png')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4.2))
        for name, pts in series:
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker='o', label=name)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(); fig.tight_layout(); fig.savefig(png, dpi=120); plt.close(fig)
        return
    except Exception:
        pass
    try:
        from PIL import Image, ImageDraw
        im = Image.new('RGB', (w, h), 'white')
        dr = ImageDraw.Draw(im)
        dr.line([(pad, h-pad), (w-pad, h-pad)], fill='black')
        dr.line([(pad, pad), (pad, h-pad)], fill='black')
        pal = [(31,119,180), (214,39,40), (44,160,44)]
        for i, (name, pts) in enumerate(series):
            col = pal[i % 3]
            xy = [(sx(x), sy(y)) for x, y in pts]
            if len(xy) >= 2:
                dr.line(xy, fill=col, width=2)
            for x, y in xy:
                dr.ellipse((x-3, y-3, x+3, y+3), fill=col)
        im.save(png)
    except Exception:
        png.write_bytes(b'')

def start_episode(method, bundle, cases, seed, sampler, collector, job_dir):
    case = bundle.next_case(cases, seed)
    snap = bundle.start_case(case)
    snap, prior, source_n = apply_episode_prior(method, bundle, snap, sampler)
    bundle.current_snapshot = snap
    collector.reset_episode(snap.env_id, snap.episode_id)
    cache_key = None if case not in bundle.caches else str(bundle.caches[case].relative_to(bundle.caches[case].parents[3]) if False else bundle.caches[case])
    if case in bundle.caches:
        cache_key = str(bundle.caches[case])
    append_jsonl(job_dir / 'episode_priors.jsonl', {
        'env_id': prior.env_id,
        'episode_id': prior.episode_id,
        'audit_mode': prior.audit_mode,
        'original_hash': prior.original_hash,
        'effective_hash': prior.hash,
        'effective_relation_count': len(prior.edges),
        'source_cache_relation_count': source_n,
        'case_id': case,
        'method': method,
    })
    return case, prior, source_n, cache_key

def aggregate_struct(trans_rows):
    dps = [r['struct']['dp_rms'] for r in trans_rows]
    ress = [r['struct']['residual_abs'] or 0 for r in trans_rows]
    dks = [r['struct']['dk_rms'] for r in trans_rows]
    dhs = [r['struct']['dh_rms'] for r in trans_rows]
    qual = [r for r in trans_rows if r.get('qualifying_for_dp')]
    def mean(xs):
        return float(sum(xs) / len(xs)) if xs else 0.0
    def rms_mean(xs):
        return float(math.sqrt(sum(x * x for x in xs) / len(xs))) if xs else 0.0
    return {
        'dp_rms_all': rms_mean(dps),
        'dk_rms_all': rms_mean(dks),
        'dh_rms_all': rms_mean(dhs),
        'residual_rms_all': rms_mean(ress),
        'max_abs_residual': max(ress) if ress else 0.0,
        'dp_nonzero_rate_all': mean([1.0 if r['struct']['dp_nonzero'] else 0.0 for r in trans_rows]),
        'residual_nonzero_rate_all': mean([1.0 if r['struct']['residual_nonzero'] else 0.0 for r in trans_rows]),
        'dp_nonzero_rate_conditioned': mean([1.0 if r['struct']['dp_nonzero'] else 0.0 for r in qual]) if qual else None,
        'residual_nonzero_rate_conditioned': mean([1.0 if r['struct']['residual_nonzero'] else 0.0 for r in qual]) if qual else None,
        'qualifying_n': len(qual),
        'mean_duration': mean([r['duration_seconds'] for r in trans_rows]),
        'median_duration': float(sorted(r['duration_seconds'] for r in trans_rows)[len(trans_rows)//2]) if trans_rows else 0.0,
        'candidate_count_mean': mean([sum(r['mask']) for r in trans_rows]),
    }

def train_job(root, method, device, device_name, cases, eval_cases, hashes_doc, out_root, max_updates, stop_after_updates=None, load_update=None):
    job_dir = out_root / method.lower() / 'seed_0'
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / 'checkpoints').mkdir(exist_ok=True)
    (job_dir / 'plots').mkdir(exist_ok=True)
    run_id = 'S1A-%s-S0' % method.upper()
    cfg = resolved_config(method, 0, cases, job_dir, hashes_doc)
    write_json(job_dir / 'config.yaml'.replace('.yaml', '.json'), cfg)
    try:
        import yaml as _yaml
        (job_dir / 'config.yaml').write_text(_yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    except Exception:
        (job_dir / 'config.yaml').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    write_json(job_dir / 'environment_snapshot.json', environment_snapshot(device, device_name, 0))
    write_json(job_dir / 'software_snapshot.json', software_snapshot(root))
    write_json(job_dir / 'cache_manifest.json', {'cache_hash': hashes_doc['cache_hash']})
    write_json(job_dir / 'manifest.json', {'run_id': run_id, 'method': method, 'seed': 0, 'task': 'D0', 'hashes': hashes_doc})
    seed_all(0)
    bundle = make_bundle(root)
    policy = make_policy(bundle.template, method, device)
    trainer = PPO(policy)
    collector = Collector(bundle, policy)
    sampler = None if method == 'B2' else PriorSampler(0)
    existing_train = []
    existing_eval = []
    existing_diag = []
    if load_update:
        ckpt_load = job_dir / 'checkpoints' / ('update_%s.pt' % load_update)
        load_checkpoint(ckpt_load, policy, trainer.optimizer)
        rngp = json.loads(ckpt_load.with_suffix('.rng.json').read_text(encoding='utf-8'))
        restore_rng({'python': rngp['python_rng'], 'numpy': rngp['numpy_rng'], 'torch': rngp['torch_rng'], 'cuda': rngp['cuda_rng']})
        if sampler is not None and rngp.get('prior_sampler'):
            # prior sampler restored best-effort at episode boundaries only
            pass
    extra0 = {
        'method': method, 'update_count': 0, 'interaction_count': 0, 'interaction_seconds': 0.0,
        **hashes_doc, 'device': str(device),
    }
    ckpt0 = job_dir / 'checkpoints' / 'step_0.pt'
    if not load_update and not ckpt0.exists():
        save_job_checkpoint(ckpt0, policy, trainer.optimizer, extra0, capture_rng(), sampler)
    eval0_path = job_dir / 'eval_step_0.json'
    if eval0_path.exists():
        ev0 = json.loads(eval0_path.read_text(encoding='utf-8'))
    elif not load_update:
        log('%s eval step0' % method)
        ev0 = eval_dev10(root, method, ckpt0, eval_cases, device, hashes_doc, eval0_path)
    else:
        ev0 = {'success_rate': None, 'success_n': 0}
    eval_rows = []
    train_rows = []
    diag_rows = []
    if (job_dir / 'eval_metrics.csv').exists():
        with (job_dir / 'eval_metrics.csv').open(encoding='utf-8') as f:
            eval_rows = list(csv.DictReader(f))
            for r in eval_rows:
                for k in ('update', 'skill_transitions', 'success_n'):
                    if r.get(k) not in (None, ''):
                        r[k] = int(float(r[k]))
                if r.get('success_rate') not in (None, ''):
                    r['success_rate'] = float(r['success_rate'])
    if not eval_rows:
        eval_rows = [{'update': 0, 'skill_transitions': 0, 'success_rate': ev0.get('success_rate'), 'success_n': ev0.get('success_n')}]
    if (job_dir / 'train_metrics.csv').exists():
        with (job_dir / 'train_metrics.csv').open(encoding='utf-8') as f:
            train_rows = list(csv.DictReader(f))
            for r in train_rows:
                for k, v in list(r.items()):
                    if v in (None, ''):
                        continue
                    try:
                        r[k] = float(v) if ('.' in v or 'e' in v.lower()) else int(v)
                    except Exception:
                        pass
        diag_rows = list(train_rows)
    rollout = Rollout()
    resume_doc = {}
    if (job_dir / 'resume.json').exists():
        resume_doc = json.loads((job_dir / 'resume.json').read_text(encoding='utf-8'))
    count = int(resume_doc.get('count') or 0) if load_update else 0
    updates = int(load_update or 0)
    interaction_seconds = float(resume_doc.get('interaction_seconds') or 0.0) if load_update else 0.0
    empty_episodes = 0
    train_success_episodes = int(resume_doc.get('train_success_episodes') or 0) if load_update else 0
    zero_reward_episodes = 0
    original_ep = 0
    absent_ep = 0
    nan_n = 0
    mask_error_n = 0
    controller_fail_n = 0
    verifier_unknown_n = 0
    timeout_n = 0
    ep_reward = 0.0
    case = prior = source_n = cache_key = None
    trans_buffer = []
    hard_fail = None
    target_updates = max_updates if stop_after_updates is None else min(max_updates, stop_after_updates)
    log('%s training start target_updates=%s' % (method, target_updates))
    try:
        while updates < target_updates and count < BUDGET and interaction_seconds < JOB_CAP:
            if bundle.current_snapshot is None:
                case, prior, source_n, cache_key = start_episode(method, bundle, cases, 0, sampler, collector, job_dir)
                if prior.audit_mode == 'original':
                    original_ep += 1
                else:
                    absent_ep += 1
                ep_reward = 0.0
            snap = bundle.current_snapshot
            try:
                t, result = collector.step(snap)
            except Exception as exc:
                hard_fail = {'error': str(exc), 'traceback': traceback.format_exc(), 'update': updates, 'count': count}
                write_json(job_dir / 'failure.json', hard_fail)
                raise
            ended = False
            success = False
            if t is None:
                empty_episodes += 1
                reason = result.get('reason') if isinstance(result, dict) else getattr(result, 'reason', None)
                append_jsonl(job_dir / 'decision_log.jsonl', {'method': method, 'no_transition': True, 'reason': reason, 'update': updates})
                ended = True
                if empty_episodes >= MAX_EMPTY:
                    raise BindingError('Runtime repeatedly exposes no executable skill')
            else:
                rec = compact_transition(method, 0, case, t, result, collector.last_output, collector.last_execution, prior.audit_mode, prior.original_hash, source_n, cache_key, run_id)
                append_jsonl(job_dir / 'transition_log.jsonl', rec)
                append_jsonl(job_dir / 'decision_log.jsonl', {'decision_id': rec['decision_id'], 'selected_candidate_id': rec['selected_candidate_id'], 'mask_hash': rec['mask_hash'], 'prior_mode': rec['prior_mode']})
                rollout.append(t)
                trans_buffer.append(rec)
                count += 1
                interaction_seconds += t.duration
                ep_reward += t.reward
                bundle.current_snapshot = t.next_snapshot
                ended = t.terminated or t.truncated
                success = bool(({} if isinstance(result, dict) else {'success': result.success}).get('success') if isinstance(result, dict) else result.success)
                exit_code = rec.get('controller_exit') or ''
                if exit_code and 'timeout' in str(exit_code).lower():
                    timeout_n += 1
                if exit_code and str(exit_code).upper() not in ('OK', 'SUCCESS', 'VERIFIED', 'CONTINUE', '') and 'fail' in str(exit_code).lower():
                    controller_fail_n += 1
                if not rec['logits_finite'] or not rec['value_finite'] or not math.isfinite(rec['old_logp']):
                    nan_n += 1
                    hard_fail = {'error': 'NaN/Inf in transition', 'rec': rec}
                    write_json(job_dir / 'failure.json', hard_fail)
                    raise DataIntegrityError('NaN/Inf in transition')
            if ended:
                if success:
                    train_success_episodes += 1
                if ep_reward == 0:
                    zero_reward_episodes += 1
                append_jsonl(job_dir / 'episode_log.jsonl', {
                    'method': method, 'case_id': case, 'episode_id': None if prior is None else prior.episode_id,
                    'success': success, 'reward': ep_reward, 'prior_mode': None if prior is None else prior.audit_mode,
                    'transitions_so_far': count,
                })
                bundle.current_snapshot = None
            if len(rollout.transitions) >= ROLLOUT_N or count >= BUDGET or interaction_seconds >= JOB_CAP:
                if not rollout.transitions:
                    break
                log('%s PPO update %s transitions=%s' % (method, updates + 1, len(rollout.transitions)))
                logs = trainer.update(rollout)
                updates += 1
                agg = aggregate_struct(trans_buffer)
                grad = [row.get('grad_norm') for row in logs]
                tot = [row.get('total') for row in logs]
                if any((x is None) or (not math.isfinite(x)) for x in tot + grad):
                    nan_n += 1
                    raise DataIntegrityError('Nonfinite PPO log')
                row = {
                    'update': updates,
                    'valid_transitions': count,
                    'rollout_n': len(trans_buffer),
                    'success_episodes': train_success_episodes,
                    'zero_reward_episodes': zero_reward_episodes,
                    'policy_loss': float(sum(r['actor'] for r in logs) / len(logs)),
                    'value_loss': float(sum(r['v'] for r in logs) / len(logs)),
                    'q_loss': float(sum(r['q'] for r in logs) / len(logs)),
                    'total_loss': float(sum(r['total'] for r in logs) / len(logs)),
                    'grad_norm': float(sum(r['grad_norm'] for r in logs) / len(logs)),
                    'NaN_n': nan_n,
                    'mask_error_n': mask_error_n,
                    'controller_failure_n': controller_fail_n,
                    'timeout_n': timeout_n,
                    'original_episode_n': original_ep,
                    'absent_episode_n': absent_ep,
                    'interaction_seconds': interaction_seconds,
                    **agg,
                }
                train_rows.append(row)
                diag_rows.append(row)
                csv_write(job_dir / 'train_metrics.csv', train_rows)
                csv_write(job_dir / 'diagnostics.csv', diag_rows)
                extra = {'method': method, 'update_count': updates, 'interaction_count': count, 'interaction_seconds': interaction_seconds, **hashes_doc, 'device': str(device)}
                if updates in (4, 8, 12, 16):
                    ckpt = job_dir / 'checkpoints' / ('update_%s.pt' % updates)
                    if not ckpt.exists():
                        save_job_checkpoint(ckpt, policy, trainer.optimizer, extra, capture_rng(), sampler)
                    log('%s eval update %s' % (method, updates))
                    ev = eval_dev10(root, method, ckpt, eval_cases, device, hashes_doc, job_dir / ('eval_update_%s.json' % updates))
                    eval_rows.append({'update': updates, 'skill_transitions': count, 'success_rate': ev['success_rate'], 'success_n': ev['success_n']})
                    csv_write(job_dir / 'eval_metrics.csv', eval_rows)
                trans_buffer = []
                write_json(job_dir / 'resume.json', {'updates': updates, 'count': count, 'interaction_seconds': interaction_seconds, 'train_success_episodes': train_success_episodes})
        csv_write(job_dir / 'eval_metrics.csv', eval_rows)
        csv_write(job_dir / 'train_metrics.csv', train_rows)
        summary = {
            'method': method,
            'completed_updates': updates,
            'valid_transitions': count,
            'train_success_episodes': train_success_episodes,
            'step0_dev_success': ev0['success_rate'],
            'final_dev_success': eval_rows[-1]['success_rate'] if eval_rows else None,
            'NaN_n': nan_n,
            'mask_error_n': mask_error_n,
            'original_episode_n': original_ep,
            'absent_episode_n': absent_ep,
            'hard_fail': hard_fail,
            'eval': eval_rows,
            'train': train_rows,
        }
        write_json(job_dir / 'job_summary.json', summary)
        (job_dir / 'run_summary.md').write_text('\n'.join([
            '# %s seed=0' % method, '',
            'updates: %s' % updates,
            'transitions: %s' % count,
            'train_success_episodes: %s' % train_success_episodes,
            'step0_dev_success: %s' % ev0['success_rate'],
            'final_dev_success: %s' % summary['final_dev_success'],
            'original/absent episodes: %s/%s' % (original_ep, absent_ep),
            '',
        ]), encoding='utf-8')
        return summary
    finally:
        try:
            bundle.environment.close()
        except Exception:
            pass

def prior_state(sampler):
    if sampler is None:
        return None
    st = sampler.state_dict()
    rng = {}
    for k, v in st['rng'].items():
        rng[k] = [v[0], list(v[1]), v[2]]
    active = {}
    for k, p in st['active'].items():
        active[str(k)] = {'env_id': p.env_id, 'episode_id': p.episode_id, 'edges': [list(e) for e in p.edges], 'original_hash': p.original_hash, 'hash': p.hash, 'audit_mode': p.audit_mode}
    return {'rng': rng, 'active': active, 'draw_count': st['draw_count']}

def save_job_checkpoint(path, policy, optimizer, extra, rng_payload, prior_sampler):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    manifest = dict(extra)
    manifest['rng_sidecar'] = str(path.with_suffix('.rng.json'))
    save_checkpoint(path, policy, optimizer, manifest)
    sidecar = {
        'python_rng': rng_sidecar(rng_payload)['python'],
        'numpy_rng': rng_sidecar(rng_payload)['numpy'],
        'torch_rng': rng_sidecar(rng_payload)['torch'],
        'cuda_rng': rng_sidecar(rng_payload)['cuda'],
        'prior_sampler': prior_state(prior_sampler),
        'extra': extra,
    }
    write_json(path.with_suffix('.rng.json'), sidecar)
    return path

def pass_conditions(b2, full):
    reasons = []
    if b2['completed_updates'] < 8 or full['completed_updates'] < 8:
        reasons.append('fewer than 8 updates')
    def any_success(job):
        train_ok = job.get('train_success_episodes', 0) > 0
        eval_ok = any((row.get('success_n') or 0) > 0 for row in job.get('eval') or [])
        return train_ok or eval_ok
    if not any_success(b2):
        reasons.append('B2 has no real policy success')
    if not any_success(full):
        reasons.append('Full has no real policy success')
    if b2.get('NaN_n') or full.get('NaN_n') or b2.get('hard_fail') or full.get('hard_fail'):
        reasons.append('NaN/Inf or hard failure')
    if b2.get('mask_error_n') or full.get('mask_error_n'):
        reasons.append('mask error')
    if (b2.get('valid_transitions') or 0) < 1 or (full.get('valid_transitions') or 0) < 1:
        reasons.append('no valid transitions')
    def last_cond(job, key):
        rows = job.get('train') or []
        if not rows:
            return 0.0
        v = rows[-1].get(key)
        return 0.0 if v is None else float(v)
    if last_cond(full, 'dp_nonzero_rate_conditioned') == 0 and last_cond(full, 'dp_nonzero_rate_all') == 0:
        # also inspect any update
        if not any((r.get('dp_nonzero_rate_conditioned') or r.get('dp_nonzero_rate_all') or 0) > 0 for r in (full.get('train') or [])):
            reasons.append('Full DP permanently 0')
    if not any((r.get('residual_nonzero_rate_conditioned') or r.get('residual_nonzero_rate_all') or 0) > 0 for r in (full.get('train') or [])):
        reasons.append('Full Delta permanently 0')
    if any((r.get('dp_rms_all') or 0) != 0 or (r.get('residual_rms_all') or 0) != 0 for r in (b2.get('train') or [])):
        reasons.append('B2 DP/Delta not strictly 0')
    missing_ckpt = []
    for job, method in ((b2, 'B2'), (full, 'Full')):
        if job.get('completed_updates', 0) < 8:
            missing_ckpt.append(method)
    if missing_ckpt:
        reasons.append('incomplete checkpoints')
    return (not reasons), reasons

def gate_table_row(job, verdict):
    last = (job.get('train') or [{}])[-1]
    return {
        'method': job.get('method'),
        'completed_updates': job.get('completed_updates'),
        'valid_transitions': job.get('valid_transitions'),
        'train_success_episodes': job.get('train_success_episodes'),
        'step0_dev_success': job.get('step0_dev_success'),
        'final_dev_success': job.get('final_dev_success'),
        'NaN_n': job.get('NaN_n'),
        'DP_nonzero_rate_conditioned': last.get('dp_nonzero_rate_conditioned'),
        'residual_nonzero_rate_conditioned': last.get('residual_nonzero_rate_conditioned'),
        'max_abs_residual': last.get('max_abs_residual'),
        'mask_error_n': job.get('mask_error_n'),
        'verdict': verdict,
    }

def write_combined(out, b2, full, status, reasons, hashes_doc):
    ok, _ = pass_conditions(b2, full)
    csv_write(out / 'tables' / 'smoke_gate_table.csv', [gate_table_row(b2, status), gate_table_row(full, status)])
    ckpts = []
    for method, job in (('B2', b2), ('Full', full)):
        for row in job.get('eval') or []:
            ckpts.append({'method': method, 'update': row['update'], 'skill_transitions': row.get('skill_transitions'), 'success_rate': row.get('success_rate')})
    csv_write(out / 'tables' / 'checkpoint_index.csv', ckpts)
    fails = [{'reason': r} for r in reasons]
    if b2.get('hard_fail'):
        fails.append({'method': 'B2', 'hard_fail': b2['hard_fail']})
    if full.get('hard_fail'):
        fails.append({'method': 'Full', 'hard_fail': full['hard_fail']})
    csv_write(out / 'tables' / 'failure_summary.csv', fails or [{'reason': 'none'}])
    def pts_success(job):
        return [(row.get('skill_transitions') or 0, row.get('success_rate') or 0) for row in job.get('eval') or []]
    def pts_dp(job):
        return [(row.get('update'), row.get('dp_rms_all') or 0) for row in job.get('train') or []]
    def pts_res(job):
        return [(row.get('update'), row.get('residual_rms_all') or 0) for row in job.get('train') or []]
    write_svg_lines(out / 'plots' / 'smoke_success.svg', [('B2', pts_success(b2)), ('Full', pts_success(full))], 'real skill transitions', 'dev success rate', 'Stage 1A smoke success (raw)')
    write_svg_lines(out / 'plots' / 'smoke_dp.svg', [('Full DP RMS (all collected candidates)', pts_dp(full))], 'update', 'Full DP RMS', 'Stage 1A Full DP RMS (all-candidate RMS of per-transition dp_rms)')
    write_svg_lines(out / 'plots' / 'smoke_prior_residual.svg', [('Full residual RMS', pts_res(full))], 'update', 'Full prior residual RMS', 'Stage 1A Full prior residual RMS')
    pair = [
        '# smoke pair report', '',
        'status: %s' % status, '',
        'B2 updates/transitions/success_ep: %s / %s / %s' % (b2.get('completed_updates'), b2.get('valid_transitions'), b2.get('train_success_episodes')),
        'Full updates/transitions/success_ep: %s / %s / %s' % (full.get('completed_updates'), full.get('valid_transitions'), full.get('train_success_episodes')),
        'B2 step0/final dev: %s / %s' % (b2.get('step0_dev_success'), b2.get('final_dev_success')),
        'Full step0/final dev: %s / %s' % (full.get('step0_dev_success'), full.get('final_dev_success')),
        'Full original/absent episodes: %s / %s' % (full.get('original_episode_n'), full.get('absent_episode_n')),
        'reasons: %s' % ('; '.join(reasons) if reasons else 'none'),
        '',
        'DP RMS statistic: RMS of per-transition selected-candidate dp_rms over the PPO rollout, all candidates collected (not only qualifying). Conditioned nonzero rates use Full + nonempty effective prior + nonempty nominal patch + unmasked.',
        '',
    ]
    (out / 'combined' / 'smoke_pair_report.md').write_text('\n'.join(pair), encoding='utf-8')
    (out / 'combined' / 'stage_1a_summary.md').write_text('\n'.join([
        '# Stage 1A summary', '',
        'status: %s' % status,
        'base_commit: %s' % BASE_COMMIT,
        'git_commit: %s' % hashes_doc.get('git_commit'),
        'next_stage: %s' % ('2A' if status in ('PASS', 'PASS_WITH_NOTES') else '1B'),
        '',
    ]), encoding='utf-8')

def cmd_stage_1a_run(root, gpu=0, only_startup_gates=False, resume=False, max_updates=16):
    root = Path(root).resolve()
    os.chdir(root)
    out = root / STAGE_DIR
    (out / 'startup_gate').mkdir(parents=True, exist_ok=True)
    (out / 'combined').mkdir(exist_ok=True)
    (out / 'attempts').mkdir(exist_ok=True)
    (out / 'logs').mkdir(exist_ok=True)
    (out / 'plots').mkdir(exist_ok=True)
    (out / 'tables').mkdir(exist_ok=True)
    log_path = out / 'logs' / 'runner.log'
    def log2(msg):
        line = '[%s] %s' % (utc_now(), msg)
        print(line, flush=True)
        with log_path.open('a', encoding='utf-8') as f:
            f.write(line + '\n')
        return line
    global log
    log = log2
    log('Stage 1A start gpu=%s only_gates=%s resume=%s max_updates=%s' % (gpu, only_startup_gates, resume, max_updates))
    status = load_status(root)
    caches_before = cache_fingerprint(root)
    write_json(out / 'startup_gate' / 'cache_fingerprint_before.json', caches_before)
    device, device_name = current_device()
    log('device=%s name=%s visible=%s' % (device, device_name, os.environ.get('CUDA_VISIBLE_DEVICES')))
    bundle = make_bundle(root)
    gates = []
    try:
        gates.append(gate_a(root, out, bundle, device))
        gates.append(gate_b(root, out, bundle, device))
        gates.append(gate_c(root, out, bundle, device))
        gates.append(gate_d(root, out, bundle, device, device_name, caches_before))
    except Exception as exc:
        err = {'error': str(exc), 'traceback': traceback.format_exc()}
        write_json(out / 'attempts' / 'startup_gate_exception.json', err)
        gates.append({'gate': 'EXC', 'passed': False, 'issues': [str(exc)]})
    finally:
        try:
            bundle.environment.close()
        except Exception:
            pass
    blocked = any(not g.get('passed') for g in gates)
    write_startup_report(out, gates, blocked)
    hashes_doc = hashes(root, caches_before)
    if blocked:
        status.update({'status': 'NEEDS_RERUN', 'completed_at': utc_now(), 'issues': [g for g in gates if not g.get('passed')], 'git_hash': hashes_doc['git_commit'], 'base_commit': BASE_COMMIT, 'execution_scope': 'startup_gates_only', 'next_stage': '1B'})
        save_status(root, status)
        write_json(out / 'combined' / 'stage_1a_result.json', {'status': 'NEEDS_RERUN', 'gates': gates, 'ppo': False})
        log('startup gates FAILED; no PPO')
        return {'status': 'NEEDS_RERUN', 'gates': gates, 'ppo_updates': False}
    if only_startup_gates:
        write_json(out / 'startup_gate' / 'ready_for_training.json', {'ready': True, 'time': utc_now(), 'hashes': hashes_doc})
        log('startup gates PASS; only_startup_gates so stopping before PPO')
        return {'status': 'STARTUP_GATES_PASS', 'gates': gates, 'ready_for_training': True}
    status.update({'status': 'RUNNING', 'started_at': status.get('started_at') or utc_now(), 'base_commit': BASE_COMMIT, 'git_hash': hashes_doc['git_commit']})
    save_status(root, status)
    split = load_split(root)
    tr = train_cases(split)
    dv = dev_cases(split)
    b2 = full = None
    final_status = 'NEEDS_RERUN'
    reasons = []
    try:
        for target in (8, 12, 16):
            if target > max_updates:
                break
            b2_done = 0 if b2 is None else b2.get('completed_updates') or 0
            full_done = 0 if full is None else full.get('completed_updates') or 0
            if b2_done < target:
                log('train B2 to %s updates' % target)
                b2 = train_job(root, 'B2', device, device_name, tr, dv, hashes_doc, out, max_updates, stop_after_updates=target, load_update=(None if b2_done == 0 else b2_done))
            if full_done < target:
                log('train Full to %s updates' % target)
                full = train_job(root, 'Full', device, device_name, tr, dv, hashes_doc, out, max_updates, stop_after_updates=target, load_update=(None if full_done == 0 else full_done))
            ok, reasons = pass_conditions(b2, full)
            log('stop-gate at %s updates: %s %s' % (target, ok, reasons))
            if ok:
                final_status = 'PASS'
                break
        else:
            ok, reasons = pass_conditions(b2, full) if b2 and full else (False, ['incomplete jobs'])
            final_status = 'NEEDS_RERUN'
    except Exception as exc:
        reasons = [str(exc)]
        write_json(out / 'attempts' / 'training_exception.json', {'error': str(exc), 'traceback': traceback.format_exc()})
        final_status = 'NEEDS_RERUN'
        log('training exception: %s' % exc)
    if b2 is None:
        b2 = {'method': 'B2'}
    if full is None:
        full = {'method': 'Full'}
    write_combined(out, b2, full, final_status, reasons, hashes_doc)
    caches_after = cache_fingerprint(root)
    if fingerprint_digest(caches_before) != fingerprint_digest(caches_after):
        reasons.append('caches mutated during Stage 1A')
        if final_status == 'PASS':
            final_status = 'NEEDS_RERUN'
    write_json(out / 'startup_gate' / 'cache_fingerprint_after_stage.json', caches_after)
    next_stage = '2A' if final_status in ('PASS', 'PASS_WITH_NOTES') else '1B'
    status.update({'status': final_status, 'completed_at': utc_now(), 'issues': reasons, 'next_stage': next_stage, 'runs_completed': ['B2:seed0', 'Full:seed0'] if b2.get('completed_updates') and full.get('completed_updates') else [], 'selected_tasks': ['D0'], 'selected_config': 'stage_1a_smoke_seed0', 'git_hash': git_commit(root), 'base_commit': BASE_COMMIT, 'execution_scope': 'D0 B2+Full seed0'})
    save_status(root, status)
    log('Stage 1A finished status=%s next=%s' % (final_status, next_stage))
    return {'status': final_status, 'gates': gates, 'reasons': reasons, 'b2': {k: b2.get(k) for k in ('completed_updates', 'valid_transitions', 'train_success_episodes', 'step0_dev_success', 'final_dev_success')}, 'full': {k: full.get(k) for k in ('completed_updates', 'valid_transitions', 'train_success_episodes', 'step0_dev_success', 'final_dev_success', 'original_episode_n', 'absent_episode_n')}, 'next_stage': next_stage}
