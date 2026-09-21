"""Smoke: construct D0 env, reset, capture RGB/depth, print hidden vs public keys."""
import os
os.environ["MUJOCO_GL"] = "egl"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")
from cp_disr.platforms.libero.d0_env import CaseSpec, make_env

def main():
    case = CaseSpec("smoke", "dev", 0, (-0.12, -0.10), (0.10, -0.10), (0.18, 0.12), (-0.18, 0.12), True)
    env = make_env(case, gpu=0)
    env.last_perception = {}
    env.reset()
    obs = env.public_observation()
    print("rgb", obs["rgb"].shape, obs["rgb"].dtype, int(obs["rgb"].mean()))
    print("depth", obs["depth"].shape, float(obs["depth"].mean()))
    print("eef", obs["eef_pos"])
    print("keys_ok", all(k in obs for k in ("rgb","depth","proprio","eef_pos")))
    h = env.hidden_truth()
    print("hidden_target", h["target"])
    env.close()
    print("SMOKE_OK")

if __name__ == "__main__":
    main()
