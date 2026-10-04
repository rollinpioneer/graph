"""Static catalog of ALFWorld pick_and_place_simple games (train + official valid splits).

The catalog separates two partitions:
  public  : what the agent may see (goal types, receptacle instances/types, static type tables)
  oracle  : hidden placement (which receptacle holds which target instance); used only by
            train-split statistics and by post-hoc evaluation, never by snapshots/priors.
"""
import glob
import json
import os
import re
import warnings
from collections import Counter, defaultdict

TASK = "pick_and_place_simple"


def _strip(name):
    return re.sub(r"type$", "", name)


def _split_dir(root, split):
    return os.path.join(root, "json_2.1.1", split)


def playable_games(root, split):
    """Game files that ALFWorld's own TextWorld env would load (solvable, has tw-pddl)."""
    out = []
    base = _split_dir(root, split)
    for d in sorted(os.listdir(base)):
        if not d.startswith(TASK + "-"):
            continue
        for t in sorted(os.listdir(os.path.join(base, d))):
            gf = os.path.join(base, d, t, "game.tw-pddl")
            if os.path.exists(gf) and os.path.exists(os.path.join(base, d, t, "traj_data.json")):
                if json.load(open(gf)).get("solvable"):
                    out.append(gf)
    return out


def scene_of(gamefile):
    task_dir = gamefile.split(os.sep)[-3] if os.sep in gamefile else gamefile.split("/")[-3]
    return int(task_dir.rsplit("-", 1)[1])


def describe_game(gamefile):
    """Run one reset and extract public/oracle partitions. Needs textworld + alfworld."""
    warnings.filterwarnings("ignore")
    import textworld
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

    infos = textworld.EnvInfos(won=True, admissible_commands=True, feedback=True, facts=True)
    env = textworld.start(gamefile, infos, wrappers=[AlfredDemangler(shuffle=False), AlfredInfos])
    state = env.reset()
    facts = list(state["facts"])
    problem = json.load(open(gamefile))["pddl_problem"]
    goal_o = re.search(r"objectType \?o (\w+)Type", problem).group(1).lower()
    goal_r = re.search(r"receptacleType \?r (\w+)Type", problem).group(1).lower()
    names = lambda f: [a.name for a in f.arguments]
    rtype = {}
    openable = set()
    r_loc = {}
    for f in facts:
        if f.name == "receptacletype":
            r, t = names(f)
            rtype[r] = _strip(t.replace(" ", ""))
        elif f.name == "openable":
            openable.add(names(f)[0])
        elif f.name == "receptacleatlocation":
            r_loc[names(f)[0]] = names(f)[1]
    otype = {}
    for f in facts:
        if f.name == "objecttype":
            o, t = names(f)
            otype[o] = _strip(t.replace(" ", ""))
    in_recep = {}
    for f in facts:
        if f.name == "inreceptacle":
            o, r = names(f)
            in_recep[o] = r
    cancontain = sorted({(_strip(names(f)[0].replace(" ", "")), _strip(names(f)[1].replace(" ", ""))) for f in facts if f.name == "cancontain"})
    targets = sorted(o for o, t in otype.items() if t == goal_o)
    first = state["feedback"]
    listed = re.search(r"you see (.*?)\.\s", first.replace("\n", " ") + " ")
    task = re.search(r"Your task is to: (.*)", first)
    return {
        "gamefile": gamefile,
        "scene": scene_of(gamefile),
        "goal_otype": goal_o,
        "goal_rtype": goal_r,
        "task_text": task.group(1).strip() if task else "",
        "receptacles": [
            {"name": r, "rtype": rtype[r], "openable": r in openable} for r in sorted(rtype)
        ],
        "initial_feedback": first,
        "oracle": {
            "targets": [{"name": o, "receptacle": in_recep.get(o)} for o in targets],
            "agent_loc": [str(f) for f in facts if f.name == "atlocation"],
        },
        "cancontain": cancontain,
    }


def build_catalog(root, out_path, workers=24):
    from multiprocessing import Pool

    games = []
    for split in ("train", "valid_seen", "valid_unseen"):
        for gf in playable_games(root, split):
            games.append((split, gf))
    with Pool(workers) as pool:
        descs = pool.map(_desc_job, [g for _, g in games], chunksize=4)
    rows = []
    for (split, _), d in zip(games, descs):
        d["split_source"] = split
        rows.append(d)
    with open(out_path, "w") as fh:
        json.dump(rows, fh)
    return rows


def _desc_job(gf):
    try:
        return describe_game(gf)
    except Exception as exc:  # recorded, never silently dropped
        return {"gamefile": gf, "error": repr(exc)}
