import json, os, subprocess, sys
from types import SimpleNamespace

from cp_disr.facts import FactStore
from cp_disr.platforms.libero.snapshot import SnapshotBuilder, CANDIDATE_FEATURE_ENCODING_VERSION


def _payload():
    contract = SimpleNamespace(name="place", bound_arguments=("red", "bin"), timeout_seconds=1.5, id="a:place:red:bin:v1", pre_pos=(), pre_neg=())
    template = SimpleNamespace(contracts=(contract,))
    class Clock:
        def now_seconds(self): return 0.25
    class Obs:
        proprioception=(1.0, 2.0, 3.0)
        frame_id="frame-0"
    snap = SnapshotBuilder(template, (("p:red", "a:place:red:bin:v1", "SUPPORTS"),), None, Clock(), 9.0).initial(FactStore(()), Obs(), "ep-0")
    return {"version": CANDIDATE_FEATURE_ENCODING_VERSION, "candidate_ids": snap.candidate_ids, "order": snap.candidate_ids, "mask": snap.mask, "features": snap.candidate_features, "base_input": snap.base_input, "prior": snap.prior_edges, "prior_hash": snap.prior_hash, "graph_node_order": [contract.id], "goal_rows": (), "nominal_patches": ()}


def test_snapshot_encoding_is_stable_across_processes_and_hashseeds():
    local_a, local_b = _payload(), _payload()
    assert local_a == local_b
    local_json = json.loads(json.dumps(local_a, sort_keys=True))
    code = "from test_stable_candidate_features import _payload; import json; print(json.dumps(_payload(), sort_keys=True))"
    outputs = []
    for seed in ("1", "2", "999"):
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=str(__import__("pathlib").Path(__file__).parent)+":"+str(__import__("pathlib").Path("src").resolve()))
        outputs.append(json.loads(subprocess.check_output([sys.executable, "-c", code], env=env, text=True)))
    assert all(item == local_json for item in outputs)
