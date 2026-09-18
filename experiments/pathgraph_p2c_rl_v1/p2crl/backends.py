"""Training backends. Fake backend never calls learn() or optimizer.step()."""
from __future__ import annotations
from pathlib import Path
from .constants_b import GRADIENT_UPDATES_SEMANTICS
from .errors import TimestepQuantumMismatch
from .io_utils import write_text_new
from .optimizer_counter import OptimizerStepCounter, SegmentReceipt
from .release import ReleaseRejected
from .training_shape import FORMAL_SHAPE, TrainingShape, shape_for_kind

class FakePolicy:
    def __init__(self, seed=0):
        import numpy as np
        rng = np.random.RandomState(int(seed))
        self._w = rng.randn(8).astype("float64")
        self.optimizer = None
    def state_dict(self):
        import numpy as np
        return {"w": np.array(self._w)}
    def predict(self, obs, deterministic=True, action_masks=None):
        legal = [i for i, bit in enumerate(action_masks or [1]) if bit]
        return (int(legal[0] if legal else 0), None)

class FakeModel:
    def __init__(self, seed=0):
        self.policy = FakePolicy(seed)
        self.num_timesteps = 0
        self.learn_called = False
        self.gradient_updates = 0
        self.optimizer_updates = 0
        self._p2crl_optimizer_step_count = 0
        self._p2crl_rollout_iteration_count = 0
        self._p2crl_ppo_epoch_count = 0
        self._p2crl_rollout_quantum = None
    def learn(self, *args, **kwargs):
        self.learn_called = True
        raise RuntimeError("FakeModel.learn is forbidden in B0")
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_text_new(path, f"FAKE_CHECKPOINT timesteps={self.num_timesteps}\n")
    def predict(self, *args, **kwargs):
        return self.policy.predict(*args, **kwargs)

def _annotate_model(model, shape: TrainingShape):
    model.n_envs = int(shape.n_envs)
    model.n_steps = int(shape.n_steps)
    model.n_epochs = int(shape.n_epochs)
    model._p2crl_rollout_quantum = int(shape.rollout_quantum)
    model._p2crl_shape = shape
    if not hasattr(model, "_p2crl_optimizer_step_count"):
        model._p2crl_optimizer_step_count = 0
    if not hasattr(model, "_p2crl_rollout_iteration_count"):
        model._p2crl_rollout_iteration_count = 0
    if not hasattr(model, "_p2crl_ppo_epoch_count"):
        model._p2crl_ppo_epoch_count = 0
    return model

def close_model(model):
    counter = getattr(model, "_p2crl_optimizer_counter", None)
    if counter is not None:
        counter.detach()
        model._p2crl_optimizer_counter = None
    env = getattr(model, "_p2crl_venv", None)
    if env is not None:
        close = getattr(env, "close", None)
        if callable(close):
            close()
        model._p2crl_venv = None

class FakeBackend:
    name = "fake"
    def __init__(self):
        self.learn_called = False
        self.gradient_updates = 0
        self.optimizer_step_count = 0
        self.models_constructed = 0
    def construct(self, release, contracts, method, seed, n_envs, n_steps, batch_size, n_epochs, shape=None):
        if not release or release.get("status") != "ACTIVE" or release.get("training_release") is not True:
            raise ReleaseRejected("RELEASE_NOT_ACTIVE", "refused model construction")
        if self.name != "fake":
            raise RuntimeError("backend")
        if shape is None:
            shape = TrainingShape(int(n_envs), int(n_steps), int(batch_size), int(n_epochs))
        self.models_constructed += 1
        model = FakeModel(seed)
        return _annotate_model(model, shape)
    def train_segment(self, model, requested_delta, expected_after=None, shape=None):
        raise RuntimeError("fake backend must not train")
    def report(self):
        return {
            "backend": self.name,
            "learn_called": bool(self.learn_called),
            "gradient_updates": int(self.gradient_updates),
            "optimizer_step_count": int(self.optimizer_step_count),
            "gradient_updates_semantics": GRADIENT_UPDATES_SEMANTICS,
            "models_constructed": int(self.models_constructed),
        }

class RealBackend:
    name = "real"
    def __init__(self):
        self.learn_called = False
        self.gradient_updates = 0
        self.optimizer_step_count = 0
        self.models_constructed = 0
        self._counters = []
    def construct(self, release, contracts, method, seed, n_envs, n_steps, batch_size, n_epochs, shape=None):
        if not release or release.get("status") != "ACTIVE" or release.get("training_release") is not True:
            raise ReleaseRejected("RELEASE_NOT_ACTIVE", "refused model construction")
        if shape is None:
            shape = TrainingShape(int(n_envs), int(n_steps), int(batch_size), int(n_epochs))
        from .learner import make_model
        self.models_constructed += 1
        model, env = make_model(
            contracts, method, seed,
            n_envs=shape.n_envs, n_steps=shape.n_steps,
            batch_size=shape.batch_size, n_epochs=shape.n_epochs,
        )
        model._p2crl_venv = env
        _annotate_model(model, shape)
        counter = OptimizerStepCounter()
        optimizer = getattr(getattr(model, "policy", None), "optimizer", None)
        counter.attach(optimizer)
        model._p2crl_optimizer_counter = counter
        self._counters.append(counter)
        return model
    def train_segment(self, model, requested_delta, expected_after=None, shape=None):
        shape = shape or getattr(model, "_p2crl_shape", None) or FORMAL_SHAPE
        requested_delta = int(requested_delta)
        if requested_delta <= 0:
            raise TimestepQuantumMismatch("requested_delta <= 0")
        if requested_delta % int(shape.rollout_quantum) != 0:
            raise TimestepQuantumMismatch("requested_delta % rollout_quantum != 0")
        before = int(model.num_timesteps)
        if expected_after is None:
            expected_after = before + requested_delta
        expected_after = int(expected_after)
        if expected_after != before + requested_delta:
            raise TimestepQuantumMismatch("expected_after mismatch")
        counter = getattr(model, "_p2crl_optimizer_counter", None)
        before_opt = int(counter.count) if counter is not None else int(getattr(model, "_p2crl_optimizer_step_count", 0) or 0)
        model.learn(total_timesteps=requested_delta, reset_num_timesteps=False)
        self.learn_called = True
        after = int(model.num_timesteps)
        if after != expected_after:
            raise TimestepQuantumMismatch(f"num_timesteps={after} expected={expected_after}")
        after_opt = int(counter.count) if counter is not None else before_opt
        opt_delta = after_opt - before_opt
        rolls = requested_delta // int(shape.rollout_quantum)
        epoch_delta = rolls * int(shape.n_epochs)
        model._p2crl_rollout_iteration_count = int(getattr(model, "_p2crl_rollout_iteration_count", 0) or 0) + rolls
        model._p2crl_ppo_epoch_count = int(getattr(model, "_p2crl_ppo_epoch_count", 0) or 0) + epoch_delta
        model._p2crl_optimizer_step_count = after_opt
        model.optimizer_updates = after_opt
        model.gradient_updates = after_opt
        self.optimizer_step_count = after_opt
        self.gradient_updates = after_opt
        return SegmentReceipt(
            requested_delta=requested_delta,
            before=before,
            after=after,
            rollout_iterations=rolls,
            ppo_epoch_count_delta=epoch_delta,
            optimizer_step_count_delta=opt_delta,
        )
    def close(self, model=None):
        if model is not None:
            close_model(model)
            return
        for counter in self._counters:
            counter.detach()
        self._counters = []
    def report(self):
        return {
            "backend": self.name,
            "learn_called": bool(self.learn_called),
            "gradient_updates": int(self.gradient_updates),
            "optimizer_step_count": int(self.optimizer_step_count),
            "gradient_updates_semantics": GRADIENT_UPDATES_SEMANTICS,
            "models_constructed": int(self.models_constructed),
        }

def get_backend(name):
    if name == "fake":
        return FakeBackend()
    if name == "real":
        return RealBackend()
    raise ValueError(name)