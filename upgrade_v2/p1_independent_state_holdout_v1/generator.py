"""Holdout trajectory generator. Forbidden: reward, potential, psi, scorer imports."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from .state_builder import EpisodeBuilder, lerp_xy, object_state, reward_relevant

# Geometry-only family table. progress_frac is a position interpolation, not a reward read.
FAMILIES = [
    {
        "family_id": "990100", "seed": 99110000, "progress_frac": 0.25,
        "A_start": (0.12, 0.11, 0.03), "A_target": (0.52, 0.11),
        "B_start": (0.13, -0.14, 0.03), "B_target": (0.49, -0.14),
        "obj_start": (0.16, 0.02, 0.03), "obj_target": (0.56, 0.02),
    },
    {
        "family_id": "990101", "seed": 99110100, "progress_frac": 0.42,
        "A_start": (0.10, 0.18, 0.03), "A_target": (0.58, 0.18),
        "B_start": (0.11, -0.20, 0.03), "B_target": (0.52, -0.20),
        "obj_start": (0.14, -0.05, 0.03), "obj_target": (0.62, -0.05),
    },
    {
        "family_id": "990102", "seed": 99110200, "progress_frac": 0.63,
        "A_start": (0.08, 0.07, 0.03), "A_target": (0.63, 0.07),
        "B_start": (0.09, -0.09, 0.03), "B_target": (0.59, -0.09),
        "obj_start": (0.12, 0.04, 0.03), "obj_target": (0.67, 0.04),
    },
    {
        "family_id": "990103", "seed": 99110300, "progress_frac": 0.35,
        "A_start": (0.15, 0.22, 0.03), "A_target": (0.48, 0.22),
        "B_start": (0.07, -0.16, 0.03), "B_target": (0.59, -0.16),
        "obj_start": (0.18, -0.08, 0.03), "obj_target": (0.51, -0.08),
    },
]

CASES = ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8"]


def _eid(family_id: str, case_id: str) -> str:
    return f"P1HOLD_F{family_id}__{case_id}"


def _loss(oid: str, idx: int, episode_id: str) -> dict:
    return {"kind": "LOSS", "object_id": oid, "loss_id": f"{episode_id}:{oid}:{idx}:1"}


def _hold(oid: str, idx: int, episode_id: str, loss_id: str) -> dict:
    return {"kind": "HOLD_REESTABLISHED", "object_id": oid, "loss_id": loss_id}


def _rec(oid: str, idx: int, episode_id: str, loss_id: str) -> dict:
    return {"kind": "RECOVERY_STARTED", "object_id": oid, "loss_id": loss_id}


def _dual_home(fam):
    return {
        "A": object_state("WAIT", fam["A_start"], fam["A_target"]),
        "B": object_state("WAIT", fam["B_start"], fam["B_target"]),
    }


def _place(obj, phase, pos=None):
    pos = pos if pos is not None else obj["pos"]
    return object_state(phase, pos, obj["target_xy"], obj["vel"])


def generate_h1(b: EpisodeBuilder, fam):
    home = _dual_home(fam)
    a_mid = lerp_xy(fam["A_start"], fam["A_target"], 0.55)
    b_mid = lerp_xy(fam["B_start"], fam["B_target"], 0.55)
    a_goal = [fam["A_target"][0], fam["A_target"][1], fam["A_start"][2]]
    b_goal = [fam["B_target"][0], fam["B_target"][1], fam["B_start"][2]]
    b.add({"A": home["A"], "B": home["B"]})
    b.add({"A": _place(home["A"], "HELD"), "B": home["B"]})
    b.add({"A": _place(home["A"], "TRANSPORT", a_mid), "B": home["B"]})
    b.add({"A": _place(home["A"], "VALID", a_goal), "B": home["B"]})
    b.add({"A": _place(home["A"], "VALID", a_goal), "B": _place(home["B"], "HELD")})
    b.add({"A": _place(home["A"], "VALID", a_goal), "B": _place(home["B"], "TRANSPORT", b_mid)})
    b.add({"A": _place(home["A"], "VALID", a_goal), "B": _place(home["B"], "VALID", b_goal)})
    b.add({"A": _place(home["A"], "VALID", a_goal), "B": _place(home["B"], "VALID", b_goal)}, success=True)


def generate_h2(b: EpisodeBuilder, fam):
    home = _dual_home(fam)
    a_mid = lerp_xy(fam["A_start"], fam["A_target"], 0.55)
    b_mid = lerp_xy(fam["B_start"], fam["B_target"], 0.55)
    a_goal = [fam["A_target"][0], fam["A_target"][1], fam["A_start"][2]]
    b_goal = [fam["B_target"][0], fam["B_target"][1], fam["B_start"][2]]
    b.add({"A": home["A"], "B": home["B"]})
    b.add({"A": home["A"], "B": _place(home["B"], "HELD")})
    b.add({"A": home["A"], "B": _place(home["B"], "TRANSPORT", b_mid)})
    b.add({"A": home["A"], "B": _place(home["B"], "VALID", b_goal)})
    b.add({"A": _place(home["A"], "HELD"), "B": _place(home["B"], "VALID", b_goal)})
    b.add({"A": _place(home["A"], "TRANSPORT", a_mid), "B": _place(home["B"], "VALID", b_goal)})
    b.add({"A": _place(home["A"], "VALID", a_goal), "B": _place(home["B"], "VALID", b_goal)})
    b.add({"A": _place(home["A"], "VALID", a_goal), "B": _place(home["B"], "VALID", b_goal)}, success=True)


def generate_h3(b: EpisodeBuilder, fam):
    home = _dual_home(fam)
    a_mid = lerp_xy(fam["A_start"], fam["A_target"], 0.50)
    b_mid = lerp_xy(fam["B_start"], fam["B_target"], 0.40)
    b_loss = lerp_xy(fam["B_start"], fam["B_target"], 0.40)
    a_goal = [fam["A_target"][0], fam["A_target"][1], fam["A_start"][2]]
    b_goal = [fam["B_target"][0], fam["B_target"][1], fam["B_start"][2]]
    b.add({"A": home["A"], "B": home["B"]})
    b.add({"A": _place(home["A"], "HELD"), "B": home["B"]})
    b.add({"A": _place(home["A"], "TRANSPORT", a_mid), "B": home["B"]})
    i_valid = b.add({"A": _place(home["A"], "VALID", a_goal), "B": home["B"]})
    b.add({"A": _place(home["A"], "VALID", a_goal), "B": _place(home["B"], "HELD")})
    i_tr = b.add({"A": _place(home["A"], "VALID", a_goal), "B": _place(home["B"], "TRANSPORT", b_mid)})
    loss_id = f"{b.episode_id}:B:{len(b.states)}:1"
    i_lost = b.add({"A": _place(home["A"], "VALID", a_goal), "B": _place(home["B"], "LOST", b_loss)},
                   events=[{"kind": "LOSS", "object_id": "B", "loss_id": loss_id}])
    i_rec = b.add({"A": _place(home["A"], "VALID", a_goal), "B": _place(home["B"], "RECOVERING", b_loss)},
                  events=[_rec("B", len(b.states), b.episode_id, loss_id)])
    b.add({"A": _place(home["A"], "VALID", a_goal), "B": _place(home["B"], "HELD", b_loss)},
          events=[_hold("B", len(b.states), b.episode_id, loss_id)])
    b.add({"A": _place(home["A"], "VALID", a_goal), "B": _place(home["B"], "TRANSPORT", b_mid)})
    b.add({"A": _place(home["A"], "VALID", a_goal), "B": _place(home["B"], "VALID", b_goal)})
    b.add({"A": _place(home["A"], "VALID", a_goal), "B": _place(home["B"], "VALID", b_goal)}, success=True)
    b.mark("alias_states", [i_valid, i_tr, i_lost, i_rec])


def generate_h4(b: EpisodeBuilder, fam):
    home = _dual_home(fam)
    a_mid = lerp_xy(fam["A_start"], fam["A_target"], 0.40)
    b_mid = lerp_xy(fam["B_start"], fam["B_target"], 0.50)
    a_loss = lerp_xy(fam["A_start"], fam["A_target"], 0.40)
    a_goal = [fam["A_target"][0], fam["A_target"][1], fam["A_start"][2]]
    b_goal = [fam["B_target"][0], fam["B_target"][1], fam["B_start"][2]]
    b.add({"A": home["A"], "B": home["B"]})
    b.add({"A": home["A"], "B": _place(home["B"], "HELD")})
    b.add({"A": home["A"], "B": _place(home["B"], "TRANSPORT", b_mid)})
    i_valid = b.add({"A": home["A"], "B": _place(home["B"], "VALID", b_goal)})
    b.add({"A": _place(home["A"], "HELD"), "B": _place(home["B"], "VALID", b_goal)})
    i_tr = b.add({"A": _place(home["A"], "TRANSPORT", a_mid), "B": _place(home["B"], "VALID", b_goal)})
    loss_id = f"{b.episode_id}:A:{len(b.states)}:1"
    i_lost = b.add({"A": _place(home["A"], "LOST", a_loss), "B": _place(home["B"], "VALID", b_goal)},
                   events=[{"kind": "LOSS", "object_id": "A", "loss_id": loss_id}])
    i_rec = b.add({"A": _place(home["A"], "RECOVERING", a_loss), "B": _place(home["B"], "VALID", b_goal)},
                  events=[_rec("A", len(b.states), b.episode_id, loss_id)])
    b.add({"A": _place(home["A"], "HELD", a_loss), "B": _place(home["B"], "VALID", b_goal)},
          events=[_hold("A", len(b.states), b.episode_id, loss_id)])
    b.add({"A": _place(home["A"], "TRANSPORT", a_mid), "B": _place(home["B"], "VALID", b_goal)})
    b.add({"A": _place(home["A"], "VALID", a_goal), "B": _place(home["B"], "VALID", b_goal)})
    b.add({"A": _place(home["A"], "VALID", a_goal), "B": _place(home["B"], "VALID", b_goal)}, success=True)
    b.mark("alias_states", [i_valid, i_tr, i_lost, i_rec])


def generate_h5(b: EpisodeBuilder, fam):
    home = _dual_home(fam)
    a_mid = lerp_xy(fam["A_start"], fam["A_target"], 0.50)
    a_goal = [fam["A_target"][0], fam["A_target"][1], fam["A_start"][2]]
    b.add({"A": home["A"], "B": home["B"]})
    b.add({"A": _place(home["A"], "HELD"), "B": home["B"]})
    b.add({"A": _place(home["A"], "TRANSPORT", a_mid), "B": home["B"]})
    i_valid = b.add({"A": _place(home["A"], "VALID", a_goal), "B": home["B"]})
    i_inv = b.add({"A": _place(home["A"], "WAIT", a_goal), "B": home["B"]})
    b.mark("invalidation_transition", [i_valid, i_inv])
    b.add({"A": _place(home["A"], "HELD", a_goal), "B": home["B"]})
    b.add({"A": _place(home["A"], "TRANSPORT", a_mid), "B": home["B"]})
    b.add({"A": _place(home["A"], "VALID", a_goal), "B": home["B"]})
    b_mid = lerp_xy(fam["B_start"], fam["B_target"], 0.55)
    b_goal = [fam["B_target"][0], fam["B_target"][1], fam["B_start"][2]]
    b.add({"A": _place(home["A"], "VALID", a_goal), "B": _place(home["B"], "HELD")})
    b.add({"A": _place(home["A"], "VALID", a_goal), "B": _place(home["B"], "TRANSPORT", b_mid)})
    b.add({"A": _place(home["A"], "VALID", a_goal), "B": _place(home["B"], "VALID", b_goal)})
    b.add({"A": _place(home["A"], "VALID", a_goal), "B": _place(home["B"], "VALID", b_goal)}, success=True)


def generate_h6(b: EpisodeBuilder, fam):
    home = _dual_home(fam)
    a_mid = lerp_xy(fam["A_start"], fam["A_target"], 0.50)
    a_goal = [fam["A_target"][0], fam["A_target"][1], fam["A_start"][2]]
    b.add({"A": home["A"], "B": home["B"]})
    b.add({"A": _place(home["A"], "HELD"), "B": home["B"]})
    b.add({"A": _place(home["A"], "TRANSPORT", a_mid), "B": home["B"]})
    start = b.add({"A": _place(home["A"], "VALID", a_goal), "B": home["B"]})
    for _ in range(4):
        b.add({"A": _place(home["A"], "VALID", a_goal), "B": home["B"]})
    end = len(b.states)
    b.mark("repeat_segment", [start, end])
    b.add({"A": _place(home["A"], "VALID", a_goal), "B": home["B"]}, terminal_failure=True)


def generate_h7(b: EpisodeBuilder, fam):
    home = object_state("WAIT", fam["obj_start"], fam["obj_target"])
    s0_pos = lerp_xy(fam["obj_start"], fam["obj_target"], fam["progress_frac"])
    prog_pos = lerp_xy(fam["obj_start"], fam["obj_target"], min(0.92, fam["progress_frac"] + 0.12))
    b.add({"obj": home})
    b.add({"obj": _place(home, "HELD", fam["obj_start"])})
    s0 = {"obj": _place(home, "TRANSPORT", s0_pos)}
    start = b.add(s0)
    cycles = []
    for k in range(3):
        c0 = len(b.states) - 1
        b.add({"obj": _place(home, "TRANSPORT", prog_pos)})
        loss_id = f"{b.episode_id}:obj:{len(b.states)}:{k+1}"
        b.add({"obj": _place(home, "LOST", prog_pos)},
              events=[{"kind": "LOSS", "object_id": "obj", "loss_id": loss_id}])
        b.add({"obj": _place(home, "RECOVERING", prog_pos)},
              events=[_rec("obj", len(b.states), b.episode_id, loss_id)])
        b.add({"obj": _place(home, "HELD", s0_pos)},
              events=[_hold("obj", len(b.states), b.episode_id, loss_id)])
        c1 = b.add(s0)
        cycles.append([c0, c1])
    b.mark("cycles", cycles)
    b.mark("s0_index", start)


def generate_h8(b: EpisodeBuilder, fam):
    home = object_state("WAIT", fam["obj_start"], fam["obj_target"])
    mid = lerp_xy(fam["obj_start"], fam["obj_target"], fam["progress_frac"])
    b.add({"obj": home})
    b.add({"obj": _place(home, "HELD", fam["obj_start"])})
    b.add({"obj": _place(home, "TRANSPORT", mid)})
    loss_id = f"{b.episode_id}:obj:{len(b.states)}:1"
    i0 = b.add({"obj": _place(home, "LOST", mid)},
               events=[{"kind": "LOSS", "object_id": "obj", "loss_id": loss_id}])
    i1 = b.add({"obj": _place(home, "RECOVERING", mid)},
               events=[_rec("obj", len(b.states), b.episode_id, loss_id)])
    b.mark("label_only_transition", [i0, i1])
    b.add({"obj": _place(home, "RECOVERING", mid)}, terminal_failure=True)


GENERATORS = {
    "H1": ("dual_order", generate_h1),
    "H2": ("dual_order", generate_h2),
    "H3": ("dual_order", generate_h3),
    "H4": ("dual_order", generate_h4),
    "H5": ("dual_order", generate_h5),
    "H6": ("dual_order", generate_h6),
    "H7": ("recovery", generate_h7),
    "H8": ("recovery", generate_h8),
}


def generate_episode(fam: dict, case_id: str) -> EpisodeBuilder:
    task, fn = GENERATORS[case_id]
    episode_id = _eid(fam["family_id"], case_id)
    builder = EpisodeBuilder(episode_id, task, fam["family_id"], case_id, fam["seed"])
    fn(builder, fam)
    if case_id == "H7":
        s0 = reward_relevant(builder.states[builder.segments["s0_index"]])
        for a, c in builder.segments["cycles"]:
            if reward_relevant(builder.states[a]) != s0 or reward_relevant(builder.states[c]) != s0:
                raise ValueError(f"{episode_id}: constructed cycle endpoints are not identical")
    return builder


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, sort_keys=True)
            f.write(line + "\n")
            h.update((line + "\n").encode("utf-8"))
    return h.hexdigest()


def generate_all(out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    raw = out_dir / "raw"
    manifest = []
    families = []
    cases = []
    sha_rows = []
    for fam in FAMILIES:
        families.append({
            "family_id": fam["family_id"], "seed": fam["seed"],
            "progress_frac": fam["progress_frac"],
            "A_start": fam["A_start"], "A_target": fam["A_target"],
            "B_start": fam["B_start"], "B_target": fam["B_target"],
            "obj_start": fam["obj_start"], "obj_target": fam["obj_target"],
        })
        for case_id in CASES:
            builder = generate_episode(fam, case_id)
            rel = Path("raw") / fam["family_id"] / case_id / "states.jsonl"
            digest = write_jsonl(out_dir / rel, builder.states)
            rec = {
                "episode_id": builder.episode_id,
                "family_id": fam["family_id"],
                "case_id": case_id,
                "task": builder.task,
                "seed": fam["seed"],
                "n_states": len(builder.states),
                "states_path": str(rel).replace("\\", "/"),
                "states_sha256": digest,
                "evidence_tier": "STATE_CONDITIONED_HOLDOUT",
                "segments": json.dumps(builder.segments, ensure_ascii=False, sort_keys=True),
            }
            manifest.append(rec)
            cases.append({**rec, "segments": builder.segments})
            sha_rows.append(f"{digest}  {rel.as_posix()}")
    (out_dir / "family_registry.json").write_text(json.dumps(families, indent=2) + "\n", encoding="utf-8")
    (out_dir / "case_registry.json").write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")
    keys = ["episode_id", "family_id", "case_id", "task", "seed", "n_states", "states_path", "states_sha256", "evidence_tier", "segments"]
    with (out_dir / "holdout_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(manifest)
    (out_dir / "states.sha256").write_text("\n".join(sha_rows) + "\n", encoding="utf-8")
    return {"episodes": len(manifest), "families": len(families)}