"""Training backends. Fake backend never calls learn() or optimizer.step()."""
from __future__ import annotations
from pathlib import Path
from .io_utils import write_new, write_text_new
from .release import ReleaseRejected

class FakePolicy:
    def __init__(self, seed=0):
        import numpy as np
        rng = np.random.RandomState(int(seed))
        self._w = rng.randn(8).astype("float64")
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
    def learn(self, *args, **kwargs):
        self.learn_called = True
        raise RuntimeError("FakeModel.learn is forbidden in B0")
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_text_new(path, f"FAKE_CHECKPOINT timesteps={self.num_timesteps}\n")
    def predict(self, *args, **kwargs):
        return self.policy.predict(*args, **kwargs)

class FakeBackend:
    name = "fake"
    def __init__(self):
        self.learn_called = False
        self.gradient_updates = 0
        self.models_constructed = 0
    def construct(self, release, contracts, method, seed, n_envs, n_steps, batch_size, n_epochs):
        if not release or release.get("status") != "ACTIVE" or release.get("training_release") is not True:
            raise ReleaseRejected("RELEASE_NOT_ACTIVE", "refused model construction")
        if self.name != "fake":
            raise RuntimeError("backend")
        self.models_constructed += 1
        return FakeModel(seed)
    def train_segment(self, model, delta_steps):
        raise RuntimeError("fake backend must not train")
    def report(self):
        return {
            "backend": self.name,
            "learn_called": bool(self.learn_called),
            "gradient_updates": int(self.gradient_updates),
            "models_constructed": int(self.models_constructed),
        }

class RealBackend:
    name = "real"
    def __init__(self):
        self.learn_called = False
        self.gradient_updates = 0
        self.models_constructed = 0
    def construct(self, release, contracts, method, seed, n_envs, n_steps, batch_size, n_epochs):
        if not release or release.get("status") != "ACTIVE" or release.get("training_release") is not True:
            raise ReleaseRejected("RELEASE_NOT_ACTIVE", "refused model construction")
        from .learner import make_model
        self.models_constructed += 1
        model, env = make_model(contracts, method, seed, n_envs, n_steps, batch_size, n_epochs)
        model._p2crl_venv = env
        return model
    def train_segment(self, model, delta_steps):
        before = int(model.num_timesteps)
        model.learn(total_timesteps=int(delta_steps), reset_num_timesteps=False)
        self.learn_called = True
        # n_envs * n_steps env steps per update; n_epochs optimizer passes
        n_envs = int(getattr(model, "n_envs", 8) or 8)
        n_steps = int(getattr(model, "n_steps", 256) or 256)
        n_epochs = int(getattr(model, "n_epochs", 10) or 10)
        updates = max(1, int(delta_steps) // max(1, n_envs * n_steps))
        self.gradient_updates += updates * n_epochs
        model.optimizer_updates = int(getattr(model, "optimizer_updates", 0)) + updates * n_epochs
        if int(model.num_timesteps) < before + int(delta_steps):
            # SB3 may overshoot slightly; require progress
            if int(model.num_timesteps) <= before:
                raise RuntimeError("learn made no progress")
        return int(model.num_timesteps)

def get_backend(name):
    if name == "fake":
        return FakeBackend()
    if name == "real":
        return RealBackend()
    raise ValueError(name)
