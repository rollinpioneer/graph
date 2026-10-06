"""CP-DISR-C1-CROSS-TASK-BLOCKSWORLD-V1: exact isomorphism hashes (runbook 17.3)."""
import random

from cp_disr.blocksworld import canonical as K
from cp_disr.blocksworld import generator as G
from cp_disr.blocksworld import state as S


def _problem(n, rng, p_stack=0.55):
    colors = [rng.randint(0, 1) for _ in range(n)]
    return S.Problem(S.default_names(n), tuple(colors), G.random_initial(n, colors, rng, p_stack=p_stack), G.random_initial(n, colors, rng, p_stack=p_stack))


def test_renaming_same_colour_blocks_keeps_every_hash():
    rng = random.Random(1)
    for _ in range(30):
        p = _problem(rng.choice([3, 4, 5]), rng)
        order = list(range(p.n))
        rng.shuffle(order)
        q = K.rename(p, tuple("k%d" % i for i in order))
        assert (K.goal_iso_hash(p), K.problem_iso_hash(p), K.goal_shape_hash(p)) == (K.goal_iso_hash(q), K.problem_iso_hash(q), K.goal_shape_hash(q))
        perm = list(range(p.n))
        reds = [i for i in range(p.n) if p.colors[i] == S.RED]
        shuffled = reds[:]
        rng.shuffle(shuffled)
        for a, b in zip(reds, shuffled):
            perm[a] = b
        r = K.rename(p, p.names, perm)                      # a colour-preserving permutation of the underlying indices
        assert K.problem_iso_hash(r) == K.problem_iso_hash(p) and K.goal_iso_hash(r) == K.goal_iso_hash(p)


def test_colour_reversal_changes_the_colour_preserving_goal_hash_but_not_the_shape():
    rng = random.Random(2)
    changed = 0
    for _ in range(40):
        n = rng.choice([3, 4, 5])
        colors = [rng.randint(0, 1) for _ in range(n)]
        goal = G.make_goal(n, colors, (2, 1) if n > 2 else (2,), S.RED, rng)
        if goal is None:
            continue
        p = S.Problem(S.default_names(n), tuple(colors), goal, goal)
        flipped = S.Problem(p.names, tuple(1 - c for c in colors), goal, goal)
        assert K.goal_shape_hash(p) == K.goal_shape_hash(flipped)
        changed += K.goal_iso_hash(p) != K.goal_iso_hash(flipped)
    assert changed > 0


def test_chain_hashes_agree_with_exhaustive_enumeration():
    rng = random.Random(3)
    probs = [_problem(rng.choice([3, 4, 5]), rng) for _ in range(60)]
    for p in probs[:30]:
        for q in probs[30:]:
            if p.n == q.n:
                assert (K.goal_iso_hash(p) == K.goal_iso_hash(q)) == (K.goal_iso_hash_bruteforce(p) == K.goal_iso_hash_bruteforce(q))
    for p in probs[:6]:
        for q in probs[6:12]:
            if p.n == q.n and p.n <= 5:
                assert (K.goal_shape_hash(p) == K.goal_shape_hash(q)) == (K.goal_shape_hash_bruteforce(p) == K.goal_shape_hash_bruteforce(q))


def test_frozen_splits_respect_the_isomorphism_rules():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    p = root / "runs/final_master/c1_route_b/blocksworld_main_v1/prep/canonical_audit.json"
    if not p.is_file():
        return
    audit = json.loads(p.read_text())
    assert audit["verdict"] == "PASS"
    for key in ("a1_goal_iso_absent_from_train_dev", "a2_shape_absent_from_train_dev_a1", "a0_isomorphic_to_source", "dev_problem_classes_disjoint_from_train"):
        assert audit["checks"][key] is True


def test_a2_shapes_are_non_isomorphic_to_train_shapes():
    train_shapes = {tuple(sorted(s, reverse=True)) for n in G.TRAIN_SHAPES for s in G.TRAIN_SHAPES[n]}
    for n, shapes in G.A2_SHAPES.items():
        for s in shapes:
            assert sum(1 for h in s if h > 1) >= 2 and tuple(sorted(s, reverse=True)) not in train_shapes
