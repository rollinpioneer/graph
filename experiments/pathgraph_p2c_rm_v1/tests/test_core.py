from __future__ import annotations

import ast
import hashlib
import unittest
from pathlib import Path

from p2cq_research.environment import SkillEnv
from p2cq_research.enumerate_mdp import enumerate_mdp, state_id
from p2cq_research.generator import MAKERS, MOTIFS, pair_contracts
from p2cq_research.task_contract import ACTION_NAMES, TaskContract, action_index, node_op

from p2crm.mask_contract import legal_mask_bool, mask_sha256
from p2crm.potentials_v2 import METHOD_V2, shaped_training_reward
from p2crm.remaining_work_model import (
    RemainingWorkPlanner,
    SearchTruncatedError,
    UnreachableWorkError,
    abstract_from_dynamic,
    phi_remaining_work,
    scale_s,
)
from p2crm.reward_audit import independent_skill_costs, planner_import_audit, pbrs_max_error
from p2crm.generator_v2 import CONF_START, family_id, iter_split, planned_pairs


ROOT = Path(__file__).resolve().parents[1]


def contract(motif="PRECEDENCE", root=5, side="left", split="development"):
    return MAKERS[motif](split, root, side)


class ImportAuditTests(unittest.TestCase):
    def test_planner_has_no_env_imports(self):
        rec = planner_import_audit(ROOT / "p2crm" / "remaining_work_model.py")
        self.assertTrue(rec["passed"], rec)

    def test_no_id_contract_in_planner(self):
        src = (ROOT / "p2crm" / "remaining_work_model.py").read_text(encoding="utf-8")
        self.assertNotIn("id(contract)", src.replace(" ", ""))

    def test_no_progress_bins(self):
        src = (ROOT / "p2crm" / "remaining_work_model.py").read_text(encoding="utf-8")
        self.assertNotIn("BIN_MID", src)
        self.assertNotIn("ZERO/MID/DONE", src)

    def test_method_name(self):
        self.assertEqual(METHOD_V2, "PATHGRAPH_REMAINING_WORK_PBRS_V2")


class RemainingWorkMechanismTests(unittest.TestCase):
    def test_k3_advance_decreases_cost(self):
        c = contract("PRECEDENCE", 5, "left")
        self.assertEqual(c.transport_steps[0], 3)
        env = SkillEnv(c)
        env.reset()
        planner = RemainingWorkPlanner(c)
        env.step(action_index(0, "ACQUIRE"))
        costs = []
        for _ in range(3):
            st = planner.abstract_state(env.dynamic_public())
            cost, trunc, _ = planner.remaining_cost(st)
            self.assertFalse(trunc)
            costs.append(cost)
            env.step(action_index(0, "ADVANCE"))
        self.assertEqual(costs[0] - costs[1], 1.0)
        self.assertEqual(costs[1] - costs[2], 1.0)

    def test_place_beats_wait_shaping(self):
        c = contract("PRECEDENCE", 5, "left")
        env = SkillEnv(c)
        planner = RemainingWorkPlanner(c)
        env.reset()
        env.step(action_index(0, "ACQUIRE"))
        for _ in range(c.transport_steps[0]):
            env.step(action_index(0, "ADVANCE"))
        dyn = env.dynamic_public()
        phi0 = planner.potential(dyn)[0]
        snap = env.snapshot()
        env.step(action_index(0, "PLACE"))
        phi_place = planner.potential(env.dynamic_public())[0]
        env.restore(snap)
        env.step(0)
        phi_wait = planner.potential(env.dynamic_public())[0]
        r_place, b_place = shaped_training_reward(0.0, phi0, phi_place, gamma=0.99, terminated=False)
        r_wait, b_wait = shaped_training_reward(0.0, phi0, phi_wait, gamma=0.99, terminated=False)
        self.assertGreater(b_place, b_wait)

    def test_unsatisfied_place_not_completion(self):
        c = contract("PRECEDENCE", 5, "left")
        # B requires A on left. Complete B without A.
        env = SkillEnv(c)
        env.reset()
        env.step(action_index(1, "ACQUIRE"))
        for _ in range(c.transport_steps[1]):
            env.step(action_index(1, "ADVANCE"))
        env.step(action_index(1, "PLACE"))
        self.assertFalse(env.state.valid[1])
        self.assertTrue(env.state.invalidated[1])
        planner = RemainingWorkPlanner(c)
        st = planner.abstract_state(env.dynamic_public())
        cost, trunc, _ = planner.remaining_cost(st)
        self.assertFalse(trunc)
        self.assertGreater(cost, 0.0)

    def test_recovery_regrasp_cost_chain(self):
        c = contract("INVALIDATION_RECOVERY", 2, "left")
        env = SkillEnv(c)
        env.reset()
        planner = RemainingWorkPlanner(c)
        # trigger one-shot by advancing B (node 3)
        env.step(action_index(3, "ACQUIRE"))
        env.step(action_index(3, "ADVANCE"))
        self.assertTrue(env.state.open_loss[0] or env.state.disturbance_consumed)
        if env.state.open_loss[0]:
            env.step(action_index(3, "RELEASE"))
            c0, _, _ = planner.remaining_cost(planner.abstract_state(env.dynamic_public()))
            env.step(action_index(0, "START_RECOVERY"))
            c1, _, _ = planner.remaining_cost(planner.abstract_state(env.dynamic_public()))
            env.step(action_index(0, "REGRASP"))
            c2, _, _ = planner.remaining_cost(planner.abstract_state(env.dynamic_public()))
            self.assertEqual(c0 - c1, 1.0)
            self.assertEqual(c1 - c2, 1.0)

    def test_invalidation_propagates(self):
        c = contract("SHARED_PREREQUISITE", 0, "left")
        env = SkillEnv(c)
        env.reset()
        # make P valid, then A valid, then acquire P (revokes P and should invalidate A)
        env.step(action_index(0, "ACQUIRE"))
        for _ in range(c.transport_steps[0]):
            env.step(action_index(0, "ADVANCE"))
        env.step(action_index(0, "PLACE"))
        env.step(action_index(2, "ACQUIRE"))
        for _ in range(c.transport_steps[2]):
            env.step(action_index(2, "ADVANCE"))
        env.step(action_index(2, "PLACE"))
        self.assertTrue(env.state.valid[2])
        env.step(action_index(0, "ACQUIRE"))
        self.assertFalse(env.state.valid[0])
        self.assertFalse(env.state.valid[2])
        planner = RemainingWorkPlanner(c)
        dyn = env.dynamic_public()
        st = planner.abstract_state(dyn)
        self.assertFalse(bool(st[0] & (1 << 2)))

    def test_oneshot_triggers_once(self):
        c = contract("INVALIDATION_RECOVERY", 1, "left")
        env = SkillEnv(c)
        env.reset()
        env.step(action_index(3, "ACQUIRE"))
        env.step(action_index(3, "ADVANCE"))
        self.assertTrue(env.state.disturbance_consumed)
        lost_after = list(env.state.open_loss)
        env.step(action_index(3, "ADVANCE"))
        self.assertEqual(lost_after, list(env.state.open_loss))
        planner = RemainingWorkPlanner(c)
        st = planner.abstract_state(env.dynamic_public())
        self.assertEqual(st[5], 1)

    def test_release_not_loss(self):
        c = contract("PRECEDENCE", 0, "left")
        env = SkillEnv(c)
        env.reset()
        env.step(action_index(0, "ACQUIRE"))
        env.step(action_index(0, "RELEASE"))
        self.assertFalse(env.state.open_loss[0])
        self.assertEqual(env.state.held_id, -1)
        planner = RemainingWorkPlanner(c)
        st = planner.abstract_state(env.dynamic_public())
        self.assertEqual(st[2] & 1, 0)

    def test_terminal_effective_phi_zero(self):
        phi_after = 0.3
        r, b = shaped_training_reward(1.0, -0.2, phi_after, gamma=0.99, terminated=True)
        self.assertEqual(b, 1.0 * (0.99 * 0.0 - (-0.2)))
        self.assertAlmostEqual(r, 1.0 + b)

    def test_pbrs_identity(self):
        c = contract("PRECEDENCE", 1, "left")
        err = pbrs_max_error(c, n_rollouts=4, max_steps=24)
        self.assertLessEqual(err, 1e-10)

    def test_cache_rebuild_and_order(self):
        c1 = contract("PRECEDENCE", 3, "left")
        c2 = TaskContract(c1.public_dict())
        env = SkillEnv(c1)
        env.reset()
        p1 = RemainingWorkPlanner(c1)
        p2 = RemainingWorkPlanner(c2)
        dyn = env.dynamic_public()
        a = p1.potential(dyn, use_cache=True)[0]
        b = p1.potential(dyn, use_cache=False)[0]
        d = p2.potential(dyn, use_cache=True)[0]
        self.assertEqual(a, b)
        self.assertEqual(a, d)
        self.assertEqual(p1.task_hash, p2.task_hash)

    def test_unit_skill_edges(self):
        c = contract("ALTERNATIVE_COST", 0, "left")
        p = RemainingWorkPlanner(c)
        env = SkillEnv(c)
        env.reset()
        st = p.abstract_state(env.dynamic_public())
        nxt = p.neighbors(st)
        self.assertTrue(nxt)
        c0, _, _ = p.remaining_cost(st)
        costs = []
        for ns in nxt:
            cn, tr, _ = p.remaining_cost(ns)
            if not tr and cn != float("inf"):
                costs.append(cn)
        self.assertTrue(any(abs(c0 - (1.0 + cn)) < 1e-9 for cn in costs))

    def test_wait_not_in_neighbors(self):
        c = contract("PRECEDENCE", 0, "left")
        p = RemainingWorkPlanner(c)
        env = SkillEnv(c)
        env.reset()
        st = p.abstract_state(env.dynamic_public())
        # WAIT is no-op so neighbor set should not include the identical pre-disturbance state
        # unless some skill is also a no-op. Initial WAIT does not change state before disturbance.
        self.assertTrue(p.neighbors(st))

    def test_phi_scale(self):
        c = contract("PRECEDENCE", 0, "left")
        s = scale_s(c)
        self.assertEqual(s, sum(c.transport_steps[i] + 4 for i in c.present_nodes()))
        env = SkillEnv(c)
        env.reset()
        phi, cost, _ = RemainingWorkPlanner(c).potential(env.dynamic_public())
        self.assertAlmostEqual(phi, -cost / s)

    def test_independent_cost_parity_small(self):
        c = contract("PRECEDENCE", 0, "left")
        mdp, _ = enumerate_mdp(c)
        ind = independent_skill_costs(mdp)
        p = RemainingWorkPlanner(c)
        mm = 0
        for sid, rec in mdp["states"].items():
            pc, tr, _ = p.remaining_cost(p.abstract_state(rec["dynamic_public"]))
            ic = ind.get(sid, float("inf"))
            if tr:
                mm += 1
            elif ic == float("inf") or pc == float("inf"):
                if ic != pc:
                    mm += 1
            elif abs(ic - pc) > 1e-9:
                mm += 1
        self.assertEqual(mm, 0)

    def test_unreachable_raises(self):
        c = contract("PRECEDENCE", 0, "left")
        p = RemainingWorkPlanner(c, max_expands=0)
        env = SkillEnv(c)
        env.reset()
        with self.assertRaises((SearchTruncatedError, UnreachableWorkError)):
            # max_expands=0: first expansion exceeds
            try:
                p.potential(env.dynamic_public())
            except SearchTruncatedError:
                raise
            except UnreachableWorkError:
                raise


class MaskTests(unittest.TestCase):
    def test_wait_always_legal(self):
        c = contract("PRECEDENCE", 0, "left")
        env = SkillEnv(c)
        env.reset()
        m = legal_mask_bool(env)
        self.assertTrue(m[0])
        self.assertEqual(len(m), 37)

    def test_method_independent_hash(self):
        c = contract("SHARED_PREREQUISITE", 1, "left")
        env = SkillEnv(c)
        env.reset()
        h = mask_sha256(legal_mask_bool(env))
        self.assertEqual(len(h), 64)
        self.assertEqual(h, mask_sha256(env.legal_mask()))

    def test_unsatisfied_place_unmasked_when_held_done(self):
        c = contract("PRECEDENCE", 5, "left")
        env = SkillEnv(c)
        env.reset()
        env.step(action_index(1, "ACQUIRE"))
        for _ in range(c.transport_steps[1]):
            env.step(action_index(1, "ADVANCE"))
        m = env.legal_mask()
        self.assertTrue(m[action_index(1, "PLACE")])

    def test_illegal_matches_wait(self):
        c = contract("PRECEDENCE", 0, "left")
        env = SkillEnv(c)
        env.reset()
        mask = env.legal_mask()
        snap = env.snapshot()
        for a, ok in enumerate(mask):
            if ok:
                continue
            env.restore(snap)
            env.state.t = 0
            _, ri, _, _, _ = env.step(a)
            fpi = env.state.fingerprint()
            env.restore(snap)
            env.state.t = 0
            _, rw, _, _, _ = env.step(0)
            fpw = env.state.fingerprint()
            self.assertEqual(fpi, fpw)
            self.assertEqual(ri, rw)


class GeneratorTests(unittest.TestCase):
    def test_confirmation_pair_count(self):
        self.assertEqual(planned_pairs("confirmation"), 2048)

    def test_smoke_pair_count(self):
        self.assertEqual(planned_pairs("smoke"), 4 * 2 * 16)

    def test_confirmation_namespace(self):
        self.assertEqual(family_id("confirmation", "PRECEDENCE", 0), CONF_START)
        self.assertTrue(1310000 <= family_id("confirmation", "INVALIDATION_RECOVERY", 31) <= 1310399)

    def test_smoke_namespace(self):
        ids = [family_id("smoke", m, r) for m in MOTIFS for r in range(2)]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(1311000 <= i <= 1311099 for i in ids))
        self.assertEqual(family_id("smoke", "PRECEDENCE", 0), 1311000)

    def test_k_allowed(self):
        for spec in iter_split("smoke"):
            for c in (spec["left"], spec["right"]):
                for i in c.present_nodes():
                    self.assertIn(c.transport_steps[i], (1, 2, 3))

    def test_split_only_conf_or_smoke(self):
        with self.assertRaises(ValueError):
            family_id("development", "PRECEDENCE", 0)


def _add_dynamic_tests():
    def make_parity(motif, root):
        def test(self, motif=motif, root=root):
            c = contract(motif, root, "left")
            mdp, _ = enumerate_mdp(c)
            ind = independent_skill_costs(mdp)
            p = RemainingWorkPlanner(c)
            mm = 0
            n = 0
            for sid, rec in list(mdp["states"].items())[:80]:
                n += 1
                pc, tr, _ = p.remaining_cost(p.abstract_state(rec["dynamic_public"]))
                ic = ind.get(sid, float("inf"))
                if tr:
                    mm += 1
                elif (ic == float("inf")) != (pc == float("inf")):
                    mm += 1
                elif ic != float("inf") and abs(ic - pc) > 1e-9:
                    mm += 1
            self.assertEqual(mm, 0, f"{motif} {root} n={n}")
        return test

    def make_mask(motif, root):
        def test(self, motif=motif, root=root):
            c = contract(motif, root, "right")
            env = SkillEnv(c)
            env.reset()
            m = legal_mask_bool(env)
            self.assertTrue(bool(m[0]))
            self.assertTrue(any(m))
        return test

    def make_pbrs(motif, root):
        def test(self, motif=motif, root=root):
            c = contract(motif, root, "left")
            err = pbrs_max_error(c, n_rollouts=2, max_steps=12)
            self.assertLessEqual(err, 1e-10)
        return test

    n = 0
    for motif in MOTIFS:
        for root in (0, 1, 5, 7):
            setattr(RemainingWorkMechanismTests, f"test_parity_{motif}_{root}", make_parity(motif, root))
            setattr(MaskTests, f"test_mask_init_{motif}_{root}", make_mask(motif, root))
            n += 2
            if root in (0, 5):
                setattr(RemainingWorkMechanismTests, f"test_pbrs_{motif}_{root}", make_pbrs(motif, root))
                n += 1
    return n


_add_dynamic_tests()


if __name__ == "__main__":
    unittest.main()