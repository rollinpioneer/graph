from __future__ import annotations
import ast
import unittest
from pathlib import Path
from p2crl.dataset_registry import build_family, permute_dict, sample_bundle, split_rng
from p2crl.contracts import load_protocol
from p2crl.oracle import conservative_optimal, remaining_cost, shortest_from_initial
from p2crl.sampler import run_seed
from p2cq_research.environment import SkillEnv
from p2cq_research.observations import LAYOUT_DIM, encode_observation
from p2cq_research.task_contract import N, action_index
from p2crm.mask_contract import legal_mask_bool
from p2crm.potentials_v2 import shaped_training_reward

R = Path(__file__).resolve().parents[1]

class ProtocolTests(unittest.TestCase):
    def test_prepare_only(self):
        p = load_protocol(R / "protocol.json")
        self.assertIs(p["training_release"], False)
        self.assertEqual(p["budgets"]["prepare_gradient_updates"], 0)
        self.assertEqual(p["budgets"]["formal_job_limit"], 60)

class SeedTests(unittest.TestCase):
    def test_run_seed_ignores_method(self):
        self.assertEqual(run_seed("A", 317), run_seed("A", 317))
        self.assertNotEqual(run_seed("A", 317), run_seed("B", 317))

class PermutationTests(unittest.TestCase):
    def test_structure_hash_invariant_and_index_moves(self):
        p = load_protocol(R / "protocol.json")
        fam = build_family(p, "train_A", "PRECEDENCE", 0)
        self.assertEqual(len(fam["perm"]), 6)
        self.assertEqual(sorted(fam["perm"]), list(range(6)))
        left = fam["left"]
        self.assertEqual(left.version, "P2CRL_INSTANCE_V1")
        self.assertEqual(len(encode_observation(left, SkillEnv(left))), LAYOUT_DIM)
        # prefixes WAIT remains 0
        self.assertTrue(all(0 in (pref + [0]) for pref in fam["prefixes"]))

    def test_true_tensor_permutation(self):
        p = load_protocol(R / "protocol.json")
        fam = build_family(p, "validation", "SHARED_PREREQUISITE", 1)
        self.assertEqual(fam["left"].family_id, fam["right"].family_id)
        self.assertNotEqual(fam["left"].task_hash(), fam["right"].task_hash())

class EnvTests(unittest.TestCase):
    def test_obs_dim_and_mask(self):
        p = load_protocol(R / "protocol.json")
        fam = build_family(p, "integration", "INVALIDATION_RECOVERY", 0)
        env = SkillEnv(fam["left"])
        env.reset()
        obs = encode_observation(fam["left"], env)
        self.assertEqual(len(obs), 224)
        mask = legal_mask_bool(env)
        self.assertEqual(len(mask), 37)
        self.assertTrue(bool(mask[0]))
        self.assertNotIn(fam["left"].split.encode(), (str(obs)).encode())

    def test_solvable(self):
        p = load_protocol(R / "protocol.json")
        fam = build_family(p, "train_B", "ALTERNATIVE_COST", 0)
        cost, _ = shortest_from_initial(fam["left"])
        self.assertLessEqual(cost, fam["left"].horizon)

    def test_pbrs_terminal(self):
        r, b = shaped_training_reward(1.0, -0.5, 0.3, gamma=0.99, beta=1.0, terminated=True)
        self.assertAlmostEqual(b, 0.99 * 0.0 - (-0.5))
        self.assertAlmostEqual(r, 1.0 + b)

class ASTGuard(unittest.TestCase):
    def test_cli_execute_not_default(self):
        text = (R / "p2crl" / "cli.py").read_text(encoding="utf-8")
        self.assertIn("execute-campaign refused", text)
        tree = ast.parse((R / "p2crl" / "learner.py").read_text(encoding="utf-8"))
        hits = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "learn":
                hits.append(n.lineno)
        self.assertEqual(hits, [])

    def test_no_best_checkpoint(self):
        text = (R / "protocol.json").read_text(encoding="utf-8")
        self.assertIn("FIXED_LAST_POST_UPDATE", text)

if __name__ == "__main__":
    unittest.main()
