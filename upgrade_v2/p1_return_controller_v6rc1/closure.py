from __future__ import annotations
import hashlib, importlib.util, sys
from pathlib import Path

PKG_LOOP = Path("/home/xushijie/PathGraph_P1_V6_Three_Issue_Execution_Agent_Package_V1.0/tools/loop_audit.py")
PRIMARY_POS = 0.001
PRIMARY_VEL = 0.01

def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()

def load_v6_closure(pkg_loop: Path | None = None):
    path = Path(pkg_loop or PKG_LOOP)
    sys.path.insert(0, str(path.parent))
    data = path.read_bytes()
    spec = importlib.util.spec_from_file_location("_v6rc1_locked_loop_audit", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.closure, {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "git_blob": git_blob_sha1(data),
        "position_m": PRIMARY_POS,
        "velocity_m_s": PRIMARY_VEL,
    }

def diagnostic_ok(p_obj, p_star, v_obj, p_eef, p_eef_star, rel_err, held, stable_s,
                  object_tol=0.005, vel_tol=0.01, eef_tol=0.01, rel_tol=0.005, stable_need=0.20):
    import math
    def n(a, b):
        return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))
    obj_err = n(p_obj, p_star)
    eef_err = n(p_eef, p_eef_star)
    vel = math.sqrt(sum(float(x) ** 2 for x in v_obj))
    passed = (obj_err <= object_tol and vel <= vel_tol and eef_err <= eef_tol
              and rel_err <= rel_tol and held and stable_s + 1e-12 >= stable_need)
    return {"diagnostic_passed": passed, "object_error_m": obj_err, "eef_error_m": eef_err,
            "object_speed": vel, "relative_error_m": rel_err, "held": held, "stable_s": stable_s,
            "replaces_primary": False}
