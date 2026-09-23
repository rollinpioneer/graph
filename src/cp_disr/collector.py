"""One real skill execution per transition; no nominal environment substitute."""
from .rl import Transition, interval_reward, gamma
from .adapters import EvaluationInput
from .common import DataIntegrityError, DiagnosticAbort, ClockIntegrityError
from .platforms.libero.clock import CONTROL_DT

TECHNICAL_EXITS = (
    "INTERRUPT_NAN_ACTION",
)
TECHNICAL_PREFIXES = (
    "INTERRUPT_CONTROLLER_EXCEPTION",
    "INTERRUPT_NAN_ACTION",
    "INTERRUPT_CLOCK",
)


def _raw_dict(execution, raw):
    if isinstance(raw, dict):
        return raw
    if isinstance(execution, dict):
        return execution
    return {}


def _exit_reason(execution, raw):
    data = _raw_dict(execution, raw)
    if data.get("controller_exit"):
        return str(data.get("controller_exit"))
    if isinstance(execution, dict):
        return str(execution.get("controller_exit") or "")
    return str(getattr(execution, "controller_exit", "") or "")


def _is_technical(raw, execution):
    data = _raw_dict(execution, raw)
    exit_reason = _exit_reason(execution, raw)
    if data.get("nan_blocked"):
        return True
    if any(exit_reason.startswith(p) for p in TECHNICAL_PREFIXES) or exit_reason in TECHNICAL_EXITS:
        return True
    if data.get("interrupted") and exit_reason.startswith("INTERRUPT"):
        return True
    return False


class Collector:
    def __init__(self, bundle, policy):
        self.bundle = bundle
        self.policy = policy
        self.prefixes = {}
        self.weights = {}
        self.success_seen = set()
        self.last_output = None
        self.last_execution = None

    def reset_episode(self, env_id, episode_id):
        self.prefixes[(env_id, episode_id)] = []
        self.weights[(env_id, episode_id)] = 1.

    def state_dict(self):
        return {
            'prefixes_len': {f'{a}|{b}': len(v) for (a, b), v in self.prefixes.items()},
            'weights': {f'{a}|{b}': float(w) for (a, b), w in self.weights.items()},
            'success_seen': [list(x) for x in self.success_seen],
        }

    def load_discount_success(self, payload):
        self.weights = {}
        for k, w in (payload.get('weights') or {}).items():
            a, b = k.split('|', 1)
            self.weights[(a, b)] = float(w)
        self.success_seen = {tuple(x) for x in (payload.get('success_seen') or [])}

    def _episode_elapsed(self, now):
        start = float(self.bundle.episode_start_seconds)
        return float(now) - start

    def _no_transition(self, reason, elapsed, extra=None):
        payload = {
            'terminated': True,
            'truncated': False,
            'reason': reason,
            'no_transition': True,
            'success': False,
            'elapsed': elapsed,
        }
        if extra:
            payload.update(extra)
        return None, payload

    def step(self, snapshot, deterministic=False):
        if snapshot.synthetic_unit_fixture:
            raise DataIntegrityError('Synthetic unit fixtures cannot enter real collection')
        import torch
        from .torch_rl import prefix_hidden
        key = (snapshot.env_id, snapshot.episode_id)
        prefix = self.prefixes[key]
        now = float(self.bundle.clock.now_seconds())
        deadline = float(self.bundle.evaluator.deadline)
        elapsed_now = self._episode_elapsed(now)
        remaining = deadline - elapsed_now
        if remaining <= 0:
            actual = self.bundle.evaluator.evaluate(
                EvaluationInput(self.bundle.task_id, *key, (), max(elapsed_now, deadline), now, now)
            )
            return None, actual
        if remaining < CONTROL_DT:
            return self._no_transition(
                'INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE',
                elapsed_now,
                extra={'remaining': remaining, 'truncated': True, 'terminated': True},
            )
        with torch.no_grad():
            out = self.policy(snapshot, prefix_hidden(self.policy, prefix))
        self.last_output = out
        if out.distribution is None:
            reason = self.bundle.safety.end_no_candidates(*key)
            return None, {'terminated': True, 'reason': reason, 'no_transition': True}
        candidate, index = out.select(bool(deterministic))
        observation = self.bundle.observations.observe()
        if not self.bundle.safety.can_execute(candidate, observation):
            raise DataIntegrityError('Safety/mask disagreement before execution')
        contract = next(c for c in snapshot.template.contracts if c.id == candidate)
        if not isinstance(contract.timeout_seconds, (int, float)) or contract.timeout_seconds <= 0:
            raise DataIntegrityError('Unbound controller timeout')
        start = self.bundle.clock.now_seconds()
        skill_timeout = float(contract.timeout_seconds)
        # Never expand remaining up to a control tick.
        capped = min(skill_timeout, remaining)
        execution = self.bundle.executor.execute(candidate, capped)
        self.last_execution = execution
        raw = getattr(self.bundle.executor, 'last', None)
        if isinstance(raw, dict) and raw.get('timeout') and capped < skill_timeout - 1e-9:
            if isinstance(execution, dict):
                execution['controller_exit'] = 'TASK_DEADLINE'
            else:
                object.__setattr__(execution, 'controller_exit', 'TASK_DEADLINE')
            raw['controller_exit'] = 'TASK_DEADLINE'
            raw['task_deadline_capped'] = True
        exit_reason = _exit_reason(execution, raw)
        if _is_technical(raw, execution):
            data = _raw_dict(execution, raw)
            raise DiagnosticAbort({
                'detail': 'technical_controller_failure',
                'controller_exit': exit_reason,
                'steps': data.get('steps'),
                'sim_duration': data.get('sim_duration'),
                'raw_sim_start': data.get('raw_sim_start'),
                'raw_sim_end': data.get('raw_sim_end'),
            })
        observation = self.bundle.observations.observe()
        measured = self.bundle.perception.infer(observation)
        facts = self.bundle.verifier.verify(measured, execution)
        end = self.bundle.clock.now_seconds()
        try:
            duration = self.bundle.clock.duration_seconds(start, end)
        except ClockIntegrityError as exc:
            raise DiagnosticAbort({
                'detail': 'clock_integrity_error',
                'message': str(exc),
                'start': start,
                'end': end,
            }) from exc
        data = _raw_dict(execution, raw)
        if (not isinstance(duration, (int, float))) or (not duration) or duration <= 0:
            # Zero-duration normal/safety outcomes are not PPO transitions and not DEADLINE.
            reason = exit_reason or 'ZERO_DURATION_NO_TRANSITION'
            if reason.startswith('INTERRUPT'):
                raise DiagnosticAbort({
                    'detail': 'non_positive_duration_rejected',
                    'duration': duration,
                    'start': start,
                    'end': end,
                    'controller_exit': reason,
                    'steps': data.get('steps'),
                })
            return self._no_transition(
                reason,
                self._episode_elapsed(end),
                extra={
                    'duration': duration,
                    'controller_exit': reason,
                    'steps': data.get('steps'),
                },
            )
        gamma(duration)
        elapsed = self._episode_elapsed(end)
        evidence = execution.evidence_ids if not isinstance(execution, dict) else tuple(execution.get('evidence_ids') or ())
        actual = self.bundle.evaluator.evaluate(
            EvaluationInput(self.bundle.task_id, *key, evidence, elapsed, start, end)
        )
        if actual.success and key in self.success_seen and any(r for _, r in actual.reward_events):
            raise DataIntegrityError('Repeated terminal success reward')
        if any(r for _, r in actual.reward_events) and not actual.success:
            raise DataIntegrityError('Reward without independent task success')
        if actual.reason == 'DEADLINE' and (actual.truncated or not actual.terminated or actual.success or actual.reward_events):
            raise DataIntegrityError('DEADLINE must be terminated, not truncated, with empty reward')
        if actual.success:
            self.success_seen.add(key)
        reward = interval_reward(duration, actual.reward_events)
        next_snapshot = self.bundle.snapshot_builder.build(snapshot, facts, observation, execution, end)
        with torch.no_grad():
            next_v = 0. if actual.terminated else float(self.policy(next_snapshot, out.hidden).value)
        transition = Transition(
            snapshot, next_snapshot, candidate,
            float(out.distribution.log_prob(torch.tensor(index, device=out.logits.device))),
            float(out.value), next_v, reward, duration, self.weights[key],
            actual.terminated, actual.truncated, actual.reason, tuple(prefix),
        )
        prefix.append(snapshot)
        self.weights[key] *= gamma(duration)
        return transition, actual