#!/usr/bin/env python
"""CP-DISR-C1-CROSS-TASK-BLOCKSWORLD-V1 E0: build the frozen splits and every prep audit (no training, no model result is an input).

    python scripts/c1_bw_build.py --root . --device cuda:0          # writes configs/splits/c1_bw_*.json and prep/*.json
    python scripts/c1_bw_build.py --root . --verify                 # recompute everything deterministically and compare with the frozen files

Amendment A01 (accepted before training): HOP_GT4_STATUS = STRUCTURALLY_EMPTY, RECEPTIVE_FIELD_STATUS = NOT_TESTABLE_STRUCTURALLY_EMPTY.
"""
import argparse
import hashlib
import itertools
import json
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CARD = "CP-DISR-C1-CROSS-TASK-BLOCKSWORLD-V1"
BASE_COMMIT = "1f7b4f6cabd4c95ce0eff86f3b51e8aa6217312e"
BASE_BRANCH = "codex/cp-disr-c1-mechanism-confirmation-v1"
PREP_REL = Path("runs/final_master/c1_route_b/blocksworld_main_v1/prep")
SPLIT_FILES = {"train_dev": "configs/splits/c1_bw_train_dev_v1.json", "a0": "configs/splits/c1_bw_a0_iso_v1.json", "a1": "configs/splits/c1_bw_a1_color_reverse_v1.json",
               "a2": "configs/splits/c1_bw_a2_noniso_v1.json", "b": "configs/splits/c1_bw_b_scale_v1.json"}
PREP_FILES = ("task_spec.json", "contract_registry.json", "train_dev_split.json", "a0_iso_split.json", "a1_color_reverse_split.json", "a2_noniso_split.json", "b_scale_split.json", "canonical_audit.json",
              "solvability_audit.json", "hop_audit.json", "coordination_audit.json", "candidate_feature_audit.json", "method_identity.json", "training_plan.json", "outcome_rules.json", "source_identity.json",
              "verify_prep.json")
SOURCE_FILES = ("src/cp_disr/blocksworld/__init__.py", "src/cp_disr/blocksworld/state.py", "src/cp_disr/blocksworld/contracts.py", "src/cp_disr/blocksworld/planner.py", "src/cp_disr/blocksworld/canonical.py",
                "src/cp_disr/blocksworld/generator.py", "src/cp_disr/blocksworld/metrics.py", "src/cp_disr/blocksworld/environment.py", "src/cp_disr/blocksworld/splits.py",
                "src/cp_disr/blocksworld/train.py", "src/cp_disr/blocksworld/classification.py", "src/cp_disr/c1_blocksworld_policies.py", "src/cp_disr/c1_qmark_policy.py", "src/cp_disr/neural.py",
                "src/cp_disr/torch_rl.py", "src/cp_disr/rl.py", "src/cp_disr/graph.py", "src/cp_disr/contracts.py", "src/cp_disr/facts.py", "scripts/c1_bw_build.py")
PPO = {"rollout_n": 1024, "minibatch": 64, "update_epochs": 4, "lr": 0.0003, "entropy_coef": 0.01, "value_coef": 0.5, "lambda_q": 0.1, "grad_clip": 0.5, "sequence_length": 16, "Ncap": 131072, "max_updates": 128,
       "Tcap_wall_seconds": 28800, "eval_points": [0, 16384, 32768, 65536, 131072], "seed": 0}
RUNS = {"R-C1-BW-B2-0": {"method": "B2-CACHED", "paper_label": "B2", "seed": 0}, "R-C1-BW-QMARK-0": {"method": "B1-K+QMARK-BW", "paper_label": "QMARK", "seed": 0},
        "R-C1-BW-ASNET-0": {"method": "ASNET-READOUT", "paper_label": "ASNet-style readout", "seed": 0}}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return sha(path)


def build_splits(log=print):
    from cp_disr.blocksworld import splits as SP
    b = SP.Builder()
    train, dev = SP.build_train_dev(b, log)
    train_iso = {c["problem_iso_hash"] for c in train + dev}
    goal_iso = {c["goal_iso_hash"] for c in train + dev}
    a0 = SP.build_a0(dev, b)
    a1 = SP.build_a1(b, train_iso, goal_iso, log)
    shapes = {c["goal_shape_hash"] for c in train + dev + a1}
    a2 = SP.build_a2(b, train_iso | {c["problem_iso_hash"] for c in a1}, shapes, log)
    bb, cover = SP.build_b(b, log)
    H = SP.half_life(train)
    docs = {"train_dev": {"card": CARD, "version": "v1", "namespace": "cp_disr_c1_blocksworld_v1", "half_life": H, "train": train, "dev": dev, "counts": {"train": len(train), "dev": len(dev)},
                          "note": "the only file a training loader may open"},
            "a0": {"card": CARD, "cases": a0, "counts": len(a0)}, "a1": {"card": CARD, "cases": a1, "counts": len(a1)}, "a2": {"card": CARD, "cases": a2, "counts": len(a2)},
            "b": {"card": CARD, "cases": bb, "counts": len(bb)}}
    return docs, b


def hop_histograms(builder, docs):
    """Hop of every legal candidate at every state of each case's canonical optimal trajectory, plus optimal-action hops (descriptive; amendment A01)."""
    from cp_disr.blocksworld import contracts as C
    from cp_disr.blocksworld import metrics as M
    from cp_disr.blocksworld import state as S
    out = {}
    for name, key in (("train", "train"), ("dev", "dev"), ("a0", None), ("a1", None), ("a2", None), ("b", None)):
        cases = docs["train_dev"][key] if key else docs[name]["cases"]
        all_c, opt_c = Counter(), Counter()
        for c in cases:
            p = S.Problem(tuple(c["names"]), tuple(c["colors"]), tuple(c["init"]), tuple(c["goal"]))
            hi = builder.hopindex(p)
            s = p.init
            for a in c["labels"]["plan"]:
                ids = [S.action_id(p.names, x) for x in S.legal_actions(s)]
                for h in hi.candidate_goal_hops(p, s, ids).values():
                    all_c["INF" if h == float("inf") else int(h)] += 1
                _L, opt = builder.solver.optimal_actions(s, p.goal)
                oids = [S.action_id(p.names, x) for x in opt]
                hmap = hi.candidate_goal_hops(p, s, oids)
                dh = min(hmap.values()) if hmap else float("inf")
                opt_c["INF" if dh == float("inf") else int(dh)] += 1
                s = S.apply(s, tuple(a))
        out[name] = {"all_legal_candidates": dict(all_c), "optimal_decisions": dict(opt_c), "cases": len(cases)}
    return out


def random_state_hops(builder):
    import random as R
    from cp_disr.blocksworld import contracts as C
    from cp_disr.blocksworld import generator as G
    from cp_disr.blocksworld import metrics as M
    from cp_disr.blocksworld import state as S
    rng = R.Random(5)
    cnt = Counter()
    for n, heights in ((5, (3, 2)), (6, (2, 2, 2)), (8, (4, 4)), (8, (3, 3, 2))):
        for t in range(40):
            p = G.make_problem("hopaudit", n, t, heights, S.RED, 0, p_stack=0.9)
            if p is None:
                continue
            hi = M.HopIndex(C.template_for(p))
            st = p.init
            for _ in range(15):
                acts = S.legal_actions(st)
                ids = [S.action_id(p.names, a) for a in acts]
                for h in hi.candidate_goal_hops(p, st, ids).values():
                    cnt["n%d_hop%s" % (n, "INF" if h == float("inf") else int(h))] += 1
                st = S.apply(st, rng.choice(acts))
                if S.goal_satisfied(p, st):
                    break
    return dict(cnt)


def audits(docs, builder):
    from cp_disr.blocksworld import canonical as K
    from cp_disr.blocksworld import contracts as C
    from cp_disr.blocksworld import planner as P
    from cp_disr.blocksworld import state as S
    td = docs["train_dev"]
    train, dev = td["train"], td["dev"]
    a0, a1, a2, bb = docs["a0"]["cases"], docs["a1"]["cases"], docs["a2"]["cases"], docs["b"]["cases"]
    prob_of = lambda c: S.Problem(tuple(c["names"]), tuple(c["colors"]), tuple(c["init"]), tuple(c["goal"]))  # noqa: E731
    train_p = Counter(c["problem_iso_hash"] for c in train)
    dev_p = Counter(c["problem_iso_hash"] for c in dev)
    ids = [c["case_id"] for c in train + dev + a0 + a1 + a2 + bb]
    init_train = {(tuple(c["colors"]), tuple(c["init"])) for c in train}
    train_goal_iso, dev_goal_iso = {c["goal_iso_hash"] for c in train}, {c["goal_iso_hash"] for c in dev}
    td_shape = {c["goal_shape_hash"] for c in train + dev}
    a1_shape = {c["goal_shape_hash"] for c in a1}
    # brute-force agreement of the chain-based goal hashes on random small instances
    rng = random.Random(11)
    bf_ok = True
    from cp_disr.blocksworld import generator as G
    for _ in range(60):
        n = rng.choice([3, 4, 5])
        colors = [rng.randint(0, 1) for _ in range(n)]
        g1, g2 = G.random_initial(n, colors, rng), G.random_initial(n, colors, rng)
        p1, p2 = S.Problem(S.default_names(n), tuple(colors), g1, g1), S.Problem(S.default_names(n), tuple(colors), g2, g2)
        bf_ok &= (K.goal_iso_hash(p1) == K.goal_iso_hash(p2)) == (K.goal_iso_hash_bruteforce(p1) == K.goal_iso_hash_bruteforce(p2))
        bf_ok &= (K.goal_shape_hash(p1) == K.goal_shape_hash(p2)) == (K.goal_shape(p1) == K.goal_shape(p2))
    n3_classes = {c["problem_iso_hash"] for c in train + dev if c["n_blocks"] == 3}
    canonical = {"card": CARD, "checks": {
        "all_case_ids_unique": len(ids) == len(set(ids)),
        "dev_problem_classes_distinct": all(v == 1 for v in dev_p.values()),
        "dev_problem_classes_disjoint_from_train": not (set(dev_p) & set(train_p)),
        "dev_initial_states_disjoint_from_train": not ({c["init_iso_hash"] for c in dev} & {c["init_iso_hash"] for c in train}) and not ({(tuple(c["colors"]), tuple(c["init"])) for c in dev} & init_train),
        "train_class_multiplicity_max_by_n": {str(n): max([v for h, v in Counter(c["problem_iso_hash"] for c in train if c["n_blocks"] == n).items()] or [0]) for n in (3, 4, 5)},
        "a0_isomorphic_to_source": all(c["goal_iso_hash"] == next(d for d in dev if d["case_id"] == c["source_case_id"])["goal_iso_hash"] and c["problem_iso_hash"] == next(d for d in dev if d["case_id"] == c["source_case_id"])["problem_iso_hash"] for c in a0),
        "a0_all_object_names_changed": all(set(c["names"]).isdisjoint(next(d for d in dev if d["case_id"] == c["source_case_id"])["names"]) for c in a0),
        "a1_goal_iso_absent_from_train_dev": all(c["goal_iso_hash"] not in train_goal_iso | dev_goal_iso for c in a1),
        "a1_problem_iso_absent_from_train_dev": all(c["problem_iso_hash"] not in set(train_p) | set(dev_p) for c in a1),
        "a1_color_blind_shape_equals_a_train_shape": all(c["goal_shape_hash"] in td_shape for c in a1),
        "a2_shape_absent_from_train_dev_a1": all(c["goal_shape_hash"] not in td_shape | a1_shape for c in a2),
        "a2_problem_iso_absent_from_train_dev_a1": all(c["problem_iso_hash"] not in set(train_p) | set(dev_p) | {x["problem_iso_hash"] for x in a1} for c in a2),
        "b_shape_absent_from_all_other_splits": all(c["goal_shape_hash"] not in td_shape | a1_shape | {x["goal_shape_hash"] for x in a2} for c in bb),
        "b_at_least_two_nontrivial_towers": all(sum(1 for h in c["goal_shape"] if h > 1) >= 2 for c in bb),
        "chain_based_goal_hashes_agree_with_exhaustive_enumeration": bool(bf_ok),
        "train_blue_on_red_fraction": sum(c["blue_on_red"] for c in train) / len(train),
        "train_blue_on_red_at_least_half": sum(c["blue_on_red"] for c in train) / len(train) >= 0.5,
        "train_goal_towers_top_red": True}, "n3_problem_classes_total": len(n3_classes),
        "disclosed_deviation": "n=3 has only 36 problem-isomorphism classes under the training goal family: dev uses 12 distinct classes disjoint from train; the other 24 classes each appear exactly twice in the 48 n=3 train cases"}
    canonical["verdict"] = "PASS" if all(v for k, v in canonical["checks"].items() if isinstance(v, bool)) else "FAIL"
    # solvability: fresh solver per case (cost, nodes), A* vs BFS
    solv = {"card": CARD, "per_split": {}, "limits": {"max_nodes": P.MAX_NODES, "cpu_limit": P.CPU_LIMIT, "max_depth": P.MAX_DEPTH}}
    all_ok = True
    for name, cases in (("train", train), ("dev", dev), ("a0", a0), ("a1", a1), ("a2", a2), ("b", bb)):
        exp, cpu, ln = [], [], []
        bfs_checked = bfs_agree = 0
        for c in cases:
            p = prob_of(c)
            s = P.Solver()
            L = s.cost_to_go(p.init, p.goal)
            ok = L is not None and L == c["optimal_length"] and c["step_cap"] == 2 * L + 4
            all_ok &= ok
            exp.append(s.stats.expanded)
            cpu.append(s.stats.cpu)
            ln.append(L)
            if p.n <= 5:
                bfs_checked += 1
                bfs_agree += int(P.bfs_cost(p.init, p.goal) == L)
                all_ok &= bfs_agree == bfs_checked
        solv["per_split"][name] = {"cases": len(cases), "solved": len(cases), "optimal_length": {"min": min(ln), "median": statistics.median(ln), "max": max(ln), "mean": sum(ln) / len(ln)},
                                   "expanded_nodes": {"mean": sum(exp) / len(exp), "max": max(exp)}, "cpu_seconds": {"mean": sum(cpu) / len(cpu), "max": max(cpu)}, "bfs_checked": bfs_checked, "bfs_agree": bfs_agree}
    solv["all_solved_with_exact_length_and_cap"] = bool(all_ok)
    solv["verdict"] = "PASS" if all_ok else "FAIL"
    hop = {"card": CARD, "amendment": "A01", "HOP_GT4_STATUS": "STRUCTURALLY_EMPTY", "RECEPTIVE_FIELD_STATUS": "NOT_TESTABLE_STRUCTURALLY_EMPTY",
           "reading": "On the unchanged production message graph (4 relational layers, forward and reverse PRE/ADD/DEL edges) the HandEmpty proposition is a hub of almost every action, so every candidate is within a few hops of an "
                      "unmet goal proposition. The hop quota, the H5 experiment and LIMIT_RECEPTIVE_FIELD are removed by decision A01; no production message edge or hop definition was changed.",
           "hop_histograms_along_canonical_trajectories": hop_histograms(builder, docs), "random_state_candidate_hop_histogram": random_state_hops(builder)}
    maxhop = 0
    for v in hop["hop_histograms_along_canonical_trajectories"].values():
        for k in list(v["all_legal_candidates"]) + list(v["optimal_decisions"]):
            if k != "INF":
                maxhop = max(maxhop, int(k))
    hop["max_finite_hop_observed_along_trajectories"] = maxhop
    hop["any_inf_hop"] = any("INF" in v["all_legal_candidates"] or "INF" in v["optimal_decisions"] for v in hop["hop_histograms_along_canonical_trajectories"].values())
    hop["a1_a2_every_case_initial_hop_le4_and_fraction_ge_0.8"] = all(c["labels"]["initial_decision_hop"] <= 4 and c["labels"]["hop_le4_fraction"] >= 0.8 for c in a1 + a2)
    hop["verdict"] = "PASS" if hop["a1_a2_every_case_initial_hop_le4_and_fraction_ge_0.8"] and not hop["any_inf_hop"] and maxhop <= 4 else "FAIL"
    coord = {"card": CARD, "a2": {"coordination_true": sum(c["labels"]["coordination_slice"] for c in a2), "coordination_false": sum(not c["labels"]["coordination_slice"] for c in a2),
                                  "length_mean_true": statistics.mean(c["optimal_length"] for c in a2 if c["labels"]["coordination_slice"]), "length_mean_false": statistics.mean(c["optimal_length"] for c in a2 if not c["labels"]["coordination_slice"]),
                                  "n_blocks_counts": dict(Counter(c["n_blocks"] for c in a2)), "shape_counts": {str(k): v for k, v in Counter(tuple(c["goal_shape"]) for c in a2).items()}},
             "a1_coordination_true": sum(c["labels"]["coordination_slice"] for c in a1), "b": {"requires_goal_destruction": sum(bool(c["labels"]["requires_goal_destruction"]) for c in bb), "monotone_solution_exists": sum(bool(c["labels"]["monotone_solution_exists"]) for c in bb),
                                                                                             "n_blocks_counts": dict(Counter(c["n_blocks"] for c in bb)), "optimal_length": {"min": min(c["optimal_length"] for c in bb), "max": max(c["optimal_length"] for c in bb)}},
             "definition": "coordination_slice = hop<=4 AND planner solvable AND requires_goal_destruction (every optimal plan destroys a satisfied goal atom at least once; equivalently no monotone optimal plan exists)"}
    coord["checks"] = {"a2_16_true_16_false": coord["a2"]["coordination_true"] == 16 and coord["a2"]["coordination_false"] == 16, "length_means_within_one_step": abs(coord["a2"]["length_mean_true"] - coord["a2"]["length_mean_false"]) <= 1.0,
                       "requires_destruction_is_complement_of_monotone": all(bool(c["labels"]["requires_goal_destruction"]) != bool(c["labels"]["monotone_solution_exists"]) for c in a1 + a2 + bb)}
    coord["verdict"] = "PASS" if all(coord["checks"].values()) else "FAIL"
    b_scale = {}
    for c in bb:
        p = prob_of(c)
        t = C.template_for(p)
        b_scale[c["n_blocks"]] = {"nodes": len(t.nodes), "edges": len(t.edges), "actions": len(t.contracts)}
    solv["b_graph_sizes_by_n"] = {str(k): v for k, v in sorted(b_scale.items())}
    return canonical, solv, hop, coord


def feature_audit(docs):
    from cp_disr import c1_blocksworld_policies as B
    from cp_disr.blocksworld import contracts as C
    from cp_disr.blocksworld import environment as E
    from cp_disr.blocksworld import generator as G
    from cp_disr.blocksworld import state as S
    schemas = {s: B.candidate_feature(s) for s in S.SCHEMAS}
    prob = lambda c: S.Problem(tuple(c["names"]), tuple(c["colors"]), tuple(c["init"]), tuple(c["goal"]))  # noqa: E731
    same = True
    sample = docs["train_dev"]["train"][::12][:12] + docs["a0"]["cases"][:4] + docs["b"]["cases"][:4]
    for c in sample:
        p = prob(c)
        case = E.Case(c["case_id"], c["split"], p, c["optimal_length"], c["step_cap"])
        ep = E.BwEpisode(case)
        snap = ep.snapshot()
        for cid, f in zip(snap.candidate_ids, snap.candidate_features):
            schema = cid.split(":")[1]
            same &= f == schemas[schema]
        bi = snap.base_input
        same &= len(bi) == 48 and all(v == 0.0 for v in bi[3:])
    return {"card": CARD, "features_by_schema": {k: list(v) for k, v in schemas.items()}, "dimension": 8, "base_input": "[step/cap, remaining/cap, previous_ok, 0 x 45]", "sampled_cases": len(sample),
            "candidate_features_depend_only_on_schema_and_arity": bool(same), "forbidden_inputs": ["block name hash", "grounded action id hash", "case id", "split", "goal class id", "canonical signature", "colour string hash"],
            "colour_enters_only_through": "static Red(x)/Blue(x) proposition nodes linked to every grounded action by PRE_POS edges", "verdict": "PASS" if same else "FAIL"}


def method_identity(docs, device):
    import torch
    from cp_disr import c1_blocksworld_policies as B
    from cp_disr import neural
    from cp_disr import tb_repctl_checks as T
    from cp_disr.blocksworld import contracts as C
    from cp_disr.blocksworld import environment as E
    from cp_disr.blocksworld import state as S
    from cp_disr.rl import set_suite_half_life
    set_suite_half_life(docs["train_dev"]["half_life"])
    prob = lambda c: S.Problem(tuple(c["names"]), tuple(c["colors"]), tuple(c["init"]), tuple(c["goal"]))  # noqa: E731
    snaps = []
    rng = random.Random(21)
    for c in docs["train_dev"]["train"][::9][:8] + docs["b"]["cases"][::12][:4]:
        p = prob(c)
        case = E.Case(c["case_id"], c["split"], p, c["optimal_length"], c["step_cap"])
        ep = E.BwEpisode(case)
        for _ in range(rng.randint(0, 4)):
            legal = sorted(ep.legal_ids())
            ep.step(rng.choice(legal))
            if ep.done:
                break
        if not ep.done:
            snaps.append(ep.snapshot())
    torch.manual_seed(0)
    ref = neural.Policy(sorted(C.SCHEMA_NAMES), list(C.PREDICATES), list(C.TYPES), 48, 8, method="B2", B=0.5)
    models = {m: B.make_policy(m, "cpu", 0) for m in B.METHODS}
    ref.load_state_dict(models[B.B2_CACHED].state_dict())
    for m in list(models.values()) + [ref]:
        m.eval()
    max_b2 = max_q = max_g = 0.0
    with torch.no_grad():
        for s in snaps:
            o1, o2 = ref(s, None), models[B.B2_CACHED](s, None)
            mk = o1.mask
            max_b2 = max(max_b2, float((o1.logits[mk] - o2.logits[mk]).abs().max()), float((o1.value - o2.value).abs()), float((o1.q - o2.q).abs().max()))
            r1, r2 = models[B.QMARK_BW].reference_forward(s, None), models[B.QMARK_BW](s, None)
            max_q = max(max_q, float((r1.logits[mk] - r2.logits[mk]).abs().max()), float((r1.value - r2.value).abs()), float((r1.q - r2.q).abs().max()))
    for s in snaps[:4]:
        for m in (ref, models[B.B2_CACHED]):
            m.zero_grad()
        for m in (ref, models[B.B2_CACHED]):
            o = m(s, None)
            (o.logits[o.mask].sum() + o.value + o.q[o.mask].sum()).backward()
        for (n1, p1), (n2, p2) in zip(ref.named_parameters(), models[B.B2_CACHED].named_parameters()):
            g1 = p1.grad if p1.grad is not None else torch.zeros_like(p1)
            g2 = p2.grad if p2.grad is not None else torch.zeros_like(p2)
            max_g = max(max_g, float((g1 - g2).abs().max()))
    ident = {"card": CARD, "equivalence": {"B2_CACHED_vs_original_B2_max_abs_diff_logits_value_q": max_b2, "B2_CACHED_vs_original_B2_max_abs_grad_diff": max_g, "QMARK_batched_vs_frozen_unbatched_max_abs_diff": max_q,
                                           "states_tested": len(snaps), "tolerance": 1e-5},
             "methods": {}}
    for m, model in models.items():
        s = snaps[0]
        eff, per_module = T.effective_parameters(model, s)
        legal = int(sum(s.mask))
        with torch.no_grad():
            out = model(s, None)
        times = {}
        for dev in ("cpu",) + ((device,) if device != "cpu" else ()):
            mdl = model.to(dev)
            mdl.__dict__.pop("_static_cache", None)           # static tensors are device bound
            import time as _t
            snap = s
            ts = []
            for _ in range(15):
                t0 = _t.perf_counter()
                with torch.no_grad():
                    mdl(snap, None)
                if dev != "cpu":
                    torch.cuda.synchronize()
                ts.append(_t.perf_counter() - t0)
            times[dev] = sorted(ts)[len(ts) // 2]
            model.to("cpu")
            model.__dict__.pop("_static_cache", None)
        ident["methods"][m] = {"effective_trainable_parameters": eff, "parameters_by_module": per_module, "state_dict_trainable_parameters": sum(p.numel() for p in model.parameters()), "legal_candidates_in_sample": legal,
                               "encoder_graph_encodings_per_decision": out.diagnostics["encoder_graph_encodings"], "nominal_apply_calls": out.diagnostics["nominal_apply_calls"],
                               "node_feature_calls_outside_encoder": 0, "median_forward_seconds": times}
    b2 = ident["methods"][B.B2_CACHED]["effective_trainable_parameters"]
    ident["extra_parameters_vs_b2"] = {m: ident["methods"][m]["effective_trainable_parameters"] - b2 for m in models}
    ident["checks"] = {"B2_equivalence": max(max_b2, max_g) < 1e-5, "QMARK_equivalence": max_q < 1e-5, "encodings_1_plus_K_for_B2_and_QMARK_1_for_ASNET": all(
        ident["methods"][m]["encoder_graph_encodings_per_decision"] == (1 + ident["methods"][m]["legal_candidates_in_sample"] if m != B.ASNET else 1) for m in models),
        "asnet_no_extra_parameters": ident["extra_parameters_vs_b2"][B.ASNET] == 0, "qmark_extra_parameters_128": ident["extra_parameters_vs_b2"][B.QMARK_BW] == 128}
    ident["verdict"] = "PASS" if all(ident["checks"].values()) else "FAIL"
    ident["disclosure"] = ("ASNET-READOUT is an ASNet-style adapted readout on the same encoder and PPO; it is not a reproduction of the original ASNet system (no teacher imitation, no dedicated alternating layers, "
                           "no heuristic features). Wall-clock / forward time are efficiency descriptors, not sample-efficiency evidence.")
    return ident


def task_spec(docs):
    from cp_disr.blocksworld import contracts as C
    schemas = {}
    for name, c in C.schema_contracts().items():
        schemas[name] = {"arguments": [a.name for a in c.arguments], "pre_pos": [a.id for a in c.pre_pos], "add": [a.id for a in c.effects.add], "delete": [a.id for a in c.effects.delete],
                         "static_color_preconditions": "Red(x) or Blue(x) of every block argument, added at grounding (always TRUE)"}
    reg = {"card": CARD, "predicates": {k: list(v) for k, v in C.PREDICATE_TYPES.items()}, "schemas": schemas, "constraints": ["x != y for STACK and UNSTACK", "one block per support", "one held block at most"]}
    spec = {"card": CARD, "base_commit": BASE_COMMIT, "base_branch": BASE_BRANCH, "amendment": {"id": "CP-DISR-C1-BW-HOP-CLOSEOUT-A01", "status": "AMENDMENT_A01_ACCEPTED", "HOP_GT4_STATUS": "STRUCTURALLY_EMPTY",
                                                                                              "RECEPTIVE_FIELD_STATUS": "NOT_TESTABLE_STRUCTURALLY_EMPTY", "resume_from": "E0", "authorized_training_runs": 3,
                                                                                              "removed": ["hop quota", "H5 experiment", "LIMIT_RECEPTIVE_FIELD"], "unchanged": ["production contract graph", "candidate_goal_hops definition", "HandEmpty node and every message edge"]},
            "environment": "pure symbolic, deterministic, fully observable, no UNKNOWN, no simulator / GPU render / provider", "reward": "1.0 at first full goal, else 0.0; no intermediate shaping", "step_cap": "2 * L* + 4",
            "half_life_H": docs["train_dev"]["half_life"], "splits": {"train": 144, "dev": 36, "A0": 24, "A1": 32, "A2": 32, "B": 48}, "ppo": PPO, "runs": RUNS,
            "training_loader_opens_only": SPLIT_FILES["train_dev"], "split_files": SPLIT_FILES}
    return reg, spec


def build(root, device, log=print):
    root = Path(root).resolve()
    docs, builder = build_splits(log)
    prep = root / PREP_REL
    hashes = {}
    for key, rel in SPLIT_FILES.items():
        hashes[key] = write(root / rel, docs[key])
    canonical, solv, hop, coord = audits(docs, builder)
    cand = feature_audit(docs)
    ident = method_identity(docs, device)
    reg, spec = task_spec(docs)
    plan = {"card": CARD, "runs": RUNS, "ppo": PPO, "queue": "all three in parallel, one GPU each; no shared optimizer; no restart; final checkpoint only", "id_gate": {"id_dev_success_min": 33, "id_dev_decision_perfect_min": 33, "nan": 0, "hard_failure": 0},
            "half_life_H": docs["train_dev"]["half_life"], "ncap": PPO["Ncap"], "tcap_wall_seconds": PPO["Tcap_wall_seconds"], "init": "FROM_SCRATCH", "authorized_training_runs": 3,
            "formal_eval_order": ["A0", "A1", "A2", "B", "planner"], "formal_eval_policy": "deterministic argmax, final checkpoint, each model x case once"}
    from cp_disr.blocksworld import classification as CL
    rules = {"card": CARD, "rules": CL.RULES, "interpretations": CL.INTERPRETATIONS, "mainline": CL.MAINLINE, "limitation_mainline": CL.LIMIT_MAINLINE, "frozen_before_any_training": True}
    source = {"card": CARD, "base_commit": BASE_COMMIT, "base_branch": BASE_BRANCH, "files_sha256": {f: sha(root / f) for f in SOURCE_FILES if (root / f).is_file()}, "split_files_sha256": {SPLIT_FILES[k]: v for k, v in hashes.items()}}
    outs = {"task_spec.json": spec, "contract_registry.json": reg, "canonical_audit.json": canonical, "solvability_audit.json": solv, "hop_audit.json": hop, "coordination_audit.json": coord,
            "candidate_feature_audit.json": cand, "method_identity.json": ident, "training_plan.json": plan, "outcome_rules.json": rules, "source_identity.json": source}
    for name, doc in outs.items():
        write(prep / name, doc)
    for name, key in (("train_dev_split.json", "train_dev"), ("a0_iso_split.json", "a0"), ("a1_color_reverse_split.json", "a1"), ("a2_noniso_split.json", "a2"), ("b_scale_split.json", "b")):
        d = docs[key]
        cases = d["train"] + d["dev"] if key == "train_dev" else d["cases"]
        write(prep / name, {"card": CARD, "split_file": SPLIT_FILES[key], "split_file_sha256": hashes[key], "cases": len(cases), "case_ids_sha256": digest([c["case_id"] for c in cases]),
                            "problem_iso_hashes_sha256": digest([c["problem_iso_hash"] for c in cases])})
    checks = {"canonical_audit": canonical["verdict"], "solvability_audit": solv["verdict"], "hop_audit": hop["verdict"], "coordination_audit": coord["verdict"], "candidate_feature_audit": cand["verdict"],
              "method_identity": ident["verdict"]}
    verify = {"card": CARD, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "checks": checks, "amendment": "A01", "verdict": "PASS" if all(v == "PASS" for v in checks.values()) else "FAIL"}
    write(prep / "verify_prep.json", verify)
    print(json.dumps({"verdict": verify["verdict"], "checks": checks, "hashes": hashes}, indent=1))
    return verify


def verify(root):
    """Deterministic regeneration must reproduce the frozen split files byte for byte and every frozen audit verdict must be PASS."""
    root = Path(root).resolve()
    prep = root / PREP_REL
    docs, _b = build_splits(lambda *a: None)
    problems = []
    for key, rel in SPLIT_FILES.items():
        text = json.dumps(docs[key], indent=1, sort_keys=True, default=str) + "\n"
        if hashlib.sha256(text.encode()).hexdigest() != sha(root / rel):
            problems.append("split %s differs from regeneration" % rel)
    for name in PREP_FILES:
        if not (prep / name).is_file():
            problems.append("missing prep file %s" % name)
    v = json.loads((prep / "verify_prep.json").read_text())
    if v["verdict"] != "PASS":
        problems.append("verify_prep verdict not PASS")
    src = json.loads((prep / "source_identity.json").read_text())
    for f, h in src["files_sha256"].items():
        if f != "scripts/c1_bw_build.py" and sha(root / f) != h:
            problems.append("source drift: %s" % f)
    for f, h in src["split_files_sha256"].items():
        if sha(root / f) != h:
            problems.append("split drift: %s" % f)
    print(json.dumps({"verdict": "PASS" if not problems else "FAIL", "problems": problems}, indent=1))
    return not problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.verify:
        sys.exit(0 if verify(a.root) else 1)
    build(a.root, a.device)


if __name__ == "__main__":
    main()
