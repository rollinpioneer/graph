"""Smoke test: can Isaac Sim + OmniGibson start headless on this GPU? Empty scene + one primitive cube. No dataset, no robot, no BEHAVIOR task."""
import json, os, sys, time, traceback
t0 = time.time()
res = {"stage": "import", "ok": False}
def dump():
    res["elapsed_s"] = round(time.time() - t0, 1)
    print("SMOKE_RESULT " + json.dumps(res), flush=True)
try:
    import torch
    res["torch"] = {"version": torch.__version__, "cuda": torch.cuda.is_available(), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    import omnigibson as og
    from omnigibson.macros import gm
    gm.HEADLESS = True
    res["stage"] = "omnigibson_imported"
    cfg = {"scene": {"type": "Scene"}, "objects": [{"type": "PrimitiveObject", "name": "cube", "primitive_type": "Cube", "scale": [0.05, 0.05, 0.05], "position": [0, 0, 1.0], "rgba": [1, 0, 0, 1]}], "render": {"viewer_width": 64, "viewer_height": 64}}
    res["stage"] = "env_create"
    env = og.Environment(configs=cfg)
    res["stage"] = "env_created"
    obj = env.scene.object_registry("name", "cube")
    p0 = obj.get_position_orientation()[0].tolist()
    res["stage"] = "stepping"
    for _ in range(60):
        env.step(env.action_space.sample() if env.action_space is not None and hasattr(env.action_space, "sample") and len(env.robots) else None) if len(env.robots) else og.sim.step()
    p1 = obj.get_position_orientation()[0].tolist()
    res["cube_z_before_after"] = [p0[2], p1[2]]
    res["physics_moved_object"] = abs(p1[2] - p0[2]) > 1e-3
    res["stage"] = "done"
    res["ok"] = True
except BaseException as e:  # noqa: BLE001
    res["error"] = "%s: %s" % (type(e).__name__, str(e)[:400])
    res["traceback_tail"] = traceback.format_exc().splitlines()[-8:]
finally:
    dump()
    try:
        import omnigibson as og
        og.shutdown()
    except BaseException:
        pass
    os._exit(0 if res.get("ok") else 3)
