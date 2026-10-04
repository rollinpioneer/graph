"""Isaac Sim only (no OmniGibson, no assets): start headless on this GPU, drop one rigid cube, step physics."""
import json, os, sys, time, traceback
t0 = time.time()
res = {"stage": "import", "ok": False}
def dump():
    res["elapsed_s"] = round(time.time() - t0, 1)
    print("ISAAC_SMOKE_RESULT " + json.dumps(res), flush=True)
app = None
try:
    from isaacsim import SimulationApp
    res["stage"] = "launch"
    app = SimulationApp({"headless": True, "width": 64, "height": 64})
    res["stage"] = "launched"
    import numpy as np
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid, GroundPlane
    world = World(stage_units_in_meters=1.0, physics_dt=1 / 60.0, rendering_dt=1 / 60.0)
    GroundPlane(prim_path="/World/ground")
    cube = world.scene.add(DynamicCuboid(prim_path="/World/cube", name="cube", position=np.array([0.0, 0.0, 1.0]), scale=np.array([0.1, 0.1, 0.1]), color=np.array([1.0, 0, 0])))
    res["stage"] = "world_reset"
    world.reset()
    z0 = float(cube.get_world_pose()[0][2])
    res["stage"] = "stepping"
    for _ in range(90):
        world.step(render=False)
    z1 = float(cube.get_world_pose()[0][2])
    res.update({"cube_z_before": z0, "cube_z_after": z1, "physics_moved_object": z1 < z0 - 0.1})
    res["stage"] = "done"
    res["ok"] = True
except BaseException as e:  # noqa: BLE001
    res["error"] = "%s: %s" % (type(e).__name__, str(e)[:400])
    res["traceback_tail"] = traceback.format_exc().splitlines()[-8:]
finally:
    dump()
    try:
        if app is not None:
            app.close()
    except BaseException:
        pass
    os._exit(0 if res.get("ok") else 3)
