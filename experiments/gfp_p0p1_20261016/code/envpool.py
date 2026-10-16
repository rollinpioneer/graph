"""Minimal pool of OGBench envs in spawned processes that can be set to an exact state."""
import multiprocessing as mp

import numpy as np


def _worker(conn, env_name):
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    import ogbench
    env = ogbench.make_env_and_datasets(env_name, env_only=True)
    env.reset()
    u = env.unwrapped
    while True:
        cmd, arg = conn.recv()
        if cmd == "reset":
            ob, _ = env.reset(seed=int(arg))
            conn.send((ob, u.data.qpos.copy(), u.data.qvel.copy()))
        elif cmd == "set":
            qpos, qvel = arg
            u.set_state(qpos, qvel)
            conn.send(u.get_ob())
        elif cmd == "step":
            ob, *_ = u.step(arg)
            conn.send(ob)
        elif cmd == "close":
            conn.close()
            return


class EnvPool:
    def __init__(self, env_name, n):
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for _ in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, env_name), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)
        self.n = n

    def reset(self, seed):
        self.conns[0].send(("reset", seed))
        return self.conns[0].recv()

    def set(self, states):
        for c, s in zip(self.conns, states):
            c.send(("set", s))
        return np.stack([c.recv() for c in self.conns[:len(states)]])

    def step(self, actions):
        for c, a in zip(self.conns, actions):
            c.send(("step", np.asarray(a, dtype=np.float64)))
        return np.stack([c.recv() for c in self.conns[:len(actions)]])

    def close(self):
        for c in self.conns:
            c.send(("close", None))
