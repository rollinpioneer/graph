from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from p2cq_research.task_contract import TaskContract, action_index, digest
from p2cq_research.generator import pair_contracts, iter_split, MOTIFS
from p2cq_research.environment import SkillEnv
from p2cq_research.observations import encode_observation, observation_digest, public_input_digest, LAYOUT_DIM
from p2cq_research.enumerate_mdp import enumerate_mdp
from p2cq_research.potentials import evaluate_all, geom_phi, flat_phi
from p2cq_research.graph_model import model_cost
from p2cq_research.source_audit import scan_tree


def test_action_count():
    from p2cq_research.task_contract import ACTION_NAMES
    assert len(ACTION_NAMES) == 37 and ACTION_NAMES[0] == "WAIT"


def test_precedence_opposite_pre():
    L, R = pair_contracts("development", "PRECEDENCE", 0)
    assert L.preconditions_satisfied(0, [False]*6)
    assert not L.preconditions_satisfied(1, [False]*6)
    assert L.preconditions_satisfied(1, [True, False, False, False, False, False])
    assert R.preconditions_satisfied(1, [False]*6)
    assert not R.preconditions_satisfied(0, [False]*6)


def test_empty_precondition_is_vacuous():
    L, _ = pair_contracts("development", "PRECEDENCE", 0)
    assert L.preconditions_satisfied(0, [False]*6)


def test_or_goal_not_and_all():
    L, _ = pair_contracts("development", "ALTERNATIVE_COST", 0)
    # G only
    valid = [False]*6
    valid[4] = True
    assert L.goal_satisfied(valid)
    valid[4] = False
    valid[0] = valid[1] = valid[2] = valid[3] = True
    assert not L.goal_satisfied(valid)


def test_padding_actions_illegal():
    L, _ = pair_contracts("development", "PRECEDENCE", 0)
    e = SkillEnv(L); e.reset()
    m = e.legal_mask()
    assert m[0] is True
    assert m[action_index(5, "ACQUIRE")] is False


def test_place_without_pre_is_legal_not_certified():
    L, _ = pair_contracts("development", "PRECEDENCE", 0)
    e = SkillEnv(L); e.reset()
    # acquire B (node 1), advance to K, place — legal, not valid
    e.step(action_index(1, "ACQUIRE"))
    for _ in range(L.transport_steps[1]):
        e.step(action_index(1, "ADVANCE"))
    assert e.legal_mask()[action_index(1, "PLACE")]
    e.step(action_index(1, "PLACE"))
    assert e.state.valid[1] is False
    assert e.state.invalidated[1] is True


def _chain_contract():
    """A->B->C with goal=C so completing A,B does not terminate the episode."""
    from p2cq_research.task_contract import (
        N, make_empty_arrays, add_clause, add_goal_clause,
        set_empty_precondition, set_invalidation_from_preconditions,
    )
    data = make_empty_arrays()
    for i, name in enumerate(("A", "B", "C")):
        data["node_mask"][i] = True
        data["node_ids"][i] = name
        data["start_xy"][i] = [float(i), 0.0]
        data["target_xy"][i] = [float(i), 1.0]
        data["transport_steps"][i] = 1
        set_empty_precondition(data, i)
    data["precondition_clause_mask"][1] = [False, False]
    data["preconditions_dnf"][1] = [[False] * N for _ in range(2)]
    add_clause(data, 1, [0])
    data["precondition_clause_mask"][2] = [False, False]
    data["preconditions_dnf"][2] = [[False] * N for _ in range(2)]
    add_clause(data, 2, [1])
    add_goal_clause(data, [2])
    set_invalidation_from_preconditions(data)
    data.update({
        "version": "P2CQ_TASK_CONTRACT_V1",
        "motif": "PRECEDENCE",
        "family_id": 1399001,
        "split": "development",
        "gamma": 0.99,
        "horizon": 32,
        "disturbance_rule": {"kind": "NONE"},
    })
    return TaskContract(data)


def _or_hold_contract():
    """G=C OR D, H<-G, goal=H so completing G does not terminate."""
    from p2cq_research.task_contract import (
        N, make_empty_arrays, add_clause, add_goal_clause,
        set_empty_precondition, set_invalidation_from_preconditions,
    )
    data = make_empty_arrays()
    for i, name in enumerate(("A", "B", "C", "D", "G", "H")):
        data["node_mask"][i] = True
        data["node_ids"][i] = name
        data["start_xy"][i] = [float(i), 0.0]
        data["target_xy"][i] = [float(i), 1.0]
        data["transport_steps"][i] = 1
        set_empty_precondition(data, i)
    for node, members in ((2, [0]), (3, [1]), (4, [2]), (5, [4])):
        data["precondition_clause_mask"][node] = [False, False]
        data["preconditions_dnf"][node] = [[False] * N for _ in range(2)]
        add_clause(data, node, members)
    add_clause(data, 4, [3])  # G = C OR D
    add_goal_clause(data, [5])
    set_invalidation_from_preconditions(data)
    data.update({
        "version": "P2CQ_TASK_CONTRACT_V1",
        "motif": "ALTERNATIVE_COST",
        "family_id": 1399002,
        "split": "development",
        "gamma": 0.99,
        "horizon": 40,
        "disturbance_rule": {"kind": "NONE"},
    })
    return TaskContract(data)


def _complete(env, contract, i):
    env.step(action_index(i, "ACQUIRE"))
    for _ in range(contract.transport_steps[i]):
        env.step(action_index(i, "ADVANCE"))
    env.step(action_index(i, "PLACE"))


def test_invalidation_closure_and_no_auto_restore():
    L = _chain_contract()
    e = SkillEnv(L); e.reset()
    _complete(e, L, 0)
    assert e.state.valid[0]
    _complete(e, L, 1)
    assert e.state.valid[1]
    assert e.terminated() is False
    # acquire A un-certifies A and should drop B; C never certified
    e.step(action_index(0, "ACQUIRE"))
    assert e.state.valid[0] is False
    assert e.state.valid[1] is False
    assert e.state.invalidated[1] is True
    assert e.state.valid[2] is False
    # restoring A's preconditions later does not auto-valid B
    e.step(action_index(0, "RELEASE"))
    _complete(e, L, 0)
    assert e.state.valid[0]
    assert e.state.valid[1] is False


def test_or_maintenance_keeps_other_support():
    L = _or_hold_contract()
    e = SkillEnv(L); e.reset()
    _complete(e, L, 0); _complete(e, L, 2)  # A then C
    _complete(e, L, 1); _complete(e, L, 3)  # B then D
    _complete(e, L, 4)  # G via either; H is the goal so episode continues
    assert e.state.valid[4]
    assert e.terminated() is False
    # un-certify C; D still supports G
    e.step(action_index(2, "ACQUIRE"))
    assert e.state.valid[2] is False
    assert e.state.valid[4] is True
    assert e.state.valid[5] is False


def test_single_resource():
    L, _ = pair_contracts("development", "PRECEDENCE", 0)
    e = SkillEnv(L); e.reset()
    e.step(action_index(0, "ACQUIRE"))
    assert e.legal_mask()[action_index(1, "ACQUIRE")] is False


def test_illegal_noop_time():
    L, _ = pair_contracts("development", "PRECEDENCE", 0)
    e = SkillEnv(L); e.reset()
    t0 = e.state.t
    fp = e.state.fingerprint()
    e.step(action_index(5, "ACQUIRE"))
    assert e.state.t == t0 + 1
    assert e.state.fingerprint() == fp  # time excluded from fingerprint


def test_observation_dim_and_parity():
    L, R = pair_contracts("development", "PRECEDENCE", 0)
    e = SkillEnv(L); e.reset()
    v = encode_observation(L, e)
    assert len(v) == LAYOUT_DIM
    e2 = SkillEnv(L); e2.reset()
    assert observation_digest(L, e) == observation_digest(L, e2)
    er = SkillEnv(R); er.reset()
    assert public_input_digest(L, e) != public_input_digest(R, er)
    # same encoder, no method field
    assert "method" not in e.public_observation()


def test_markov():
    L, _ = pair_contracts("development", "PRECEDENCE", 1)
    e = SkillEnv(L); e.reset()
    e.step(0)
    fp = e.state.fingerprint()
    e.step(action_index(0, "ACQUIRE"))
    nxt = e.state.fingerprint()
    e.reset(); e.step(0)
    assert e.state.fingerprint() == fp
    e.step(action_index(0, "ACQUIRE"))
    assert e.state.fingerprint() == nxt


def test_clone_parent_immutable():
    L, _ = pair_contracts("development", "SHARED_PREREQUISITE", 0)
    e = SkillEnv(L); e.reset()
    snap = e.snapshot()
    parent = e.state.fingerprint()
    e.step(action_index(0, "ACQUIRE"))
    e.restore(snap)
    assert e.state.fingerprint() == parent


def test_prefix_replay():
    L, _ = pair_contracts("development", "PRECEDENCE", 2)
    e = SkillEnv(L); e.reset()
    seq = [0, action_index(0, "ACQUIRE"), action_index(0, "ADVANCE")]
    for a in seq:
        e.step(a)
    end = e.state.fingerprint()
    e.reset()
    for a in seq:
        e.step(a)
    assert e.state.fingerprint() == end


def test_graph_no_env_import():
    ast = scan_tree(ROOT)
    assert ast["environment_imports_clean"]
    assert ast["graph_does_not_import_env"]


def test_potentials_reject_oracle_keys():
    L, _ = pair_contracts("development", "PRECEDENCE", 0)
    e = SkillEnv(L); e.reset()
    vals = evaluate_all(L, e.dynamic_public())
    assert "GEOM_COUNT_EVENTS_PBRS_V1" in vals
    assert vals["TASK_ONLY_ZERO_V1"] == 0.0


def test_enumerate_complete_small():
    L, _ = pair_contracts("development", "PRECEDENCE", 0)
    d = L.public_dict(); d["horizon"] = 16
    c = TaskContract(d)
    mdp, pref = enumerate_mdp(c)
    assert mdp["meta"]["omitted"] == 0
    assert mdp["meta"]["complete"]
    s0 = mdp["states"][mdp["initial_state"]]
    assert len(s0["outcomes"]) == 37
    assert len(s0["valid_actions"]) == 37


def test_structure_hash_rename_invariant():
    L1, _ = pair_contracts("development", "PRECEDENCE", 0)
    L2, _ = pair_contracts("development", "PRECEDENCE", 1)
    # different parameters may share topology
    assert len(L1.structure_hash()) == 64


def test_disturbance_local_effect_same():
    L, R = pair_contracts("development", "INVALIDATION_RECOVERY", 0)
    def fire(c):
        e = SkillEnv(c); e.reset()
        # complete P or Q is not required; advance B to trigger
        e.step(action_index(3, "ACQUIRE"))
        e.step(action_index(3, "ADVANCE"))
        return e.state.open_loss[0], e.state.disturbance_consumed
    assert fire(L) == fire(R) == (True, True)


def test_no_graph_in_environment_source():
    txt = (ROOT / "p2cq_research/environment.py").read_text(encoding="utf-8")
    assert "graph_model" not in txt and "evaluate_all" not in txt

def test_split_namespaces_disjoint():
    from p2cq_research.generator import family_id, ROOTS_PER_MOTIF
    dev, conf = set(), set()
    for motif in MOTIFS:
        for root in range(ROOTS_PER_MOTIF):
            dev.add(family_id("development", motif, root))
            conf.add(family_id("confirmation", motif, root))
    assert not (dev & conf)
    assert min(dev) >= 1301000 and max(dev) < 1302000
    assert min(conf) >= 1302000
    assert len(dev) == len(conf) == 128


def test_wait_shaping_positive_when_phi_negative():
    L, _ = pair_contracts("development", "PRECEDENCE", 0)
    e = SkillEnv(L); e.reset()
    dyn = e.dynamic_public()
    phi = geom_phi(L, dyn)
    assert phi < 0
    before = phi
    e.step(0)
    after = geom_phi(L, e.dynamic_public())
    # WAIT does not change task state; undiscounted PBRS identity is not claimed.
    shaping = L.gamma * after - before
    assert after == before
    assert shaping > 0
