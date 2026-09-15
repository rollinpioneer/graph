from upgrade_v2.p1_physical_mechanism_v5 import registry, scoring, graph_binding

def test_112_unique():
    fam={"families":[{"family_id":f"P1V5_F0{i}_92000{i}","family_seed":920000+i,"rollout_seed_base":92100000+100*i} for i in range(8)]}
    cases={"case_order":[{"case_id":f"C{i}","index":i,"task":"recovery","required_semantics":"X"} for i in range(14)]}
    plans=registry.planned_rollouts(fam, cases)
    assert len(plans)==112
    assert len({(p["family_id"],p["case_id"],p["rollout_seed"]) for p in plans})==112

def test_scale_table():
    assert graph_binding.REC_COST["in_transit"]==2/6
    assert abs(sum(1 for _ in graph_binding.REC_COST)-8)==0

def test_paired_once():
    seq=[dict(episode_id="e", step=0, node_before="in_transit", node_after="dropped_or_misaligned", edge_type="failure",
              cost_before=2/6, cost_after=5/6, linear_before=0.5, linear_after=0.0, unordered_before=0.5, unordered_after=0.0,
              t0_ns=0, t1_ns=1, available_at_ns=1, distance_before=0.2, distance_after=0.2, loss_episode_id="L1", recovery_completed=False),
         dict(episode_id="e", step=1, node_before="dropped_or_misaligned", node_after="recovery", edge_type="recovery",
              cost_before=5/6, cost_after=4/6, linear_before=0.0, linear_after=0.25, unordered_before=0, unordered_after=0.25,
              t0_ns=1, t1_ns=2, available_at_ns=2, distance_before=0.2, distance_after=0.2, loss_episode_id="L1", recovery_completed=False),
         dict(episode_id="e", step=2, node_before="recovery", node_after="grasped", edge_type="recovery",
              cost_before=4/6, cost_after=3/6, linear_before=0.25, linear_after=0.25, unordered_before=0.25, unordered_after=0.25,
              t0_ns=2, t1_ns=3, available_at_ns=3, distance_before=0.2, distance_after=0.2, loss_episode_id="L1", recovery_completed=True, recovered_loss_episode_id="L1")]
    det=scoring.score_episode(seq,"recovery")
    paired=[d for d in det if d["method"]=="PAIRED_EVENT_BALANCED_V1"]
    assert sum(d["reward_mu"] for d in paired)==0.0

def test_potential_domain():
    scoring.potential(0.5,0.2)
    try:
        scoring.potential(2.0,0.0); raise AssertionError
    except ValueError:
        pass