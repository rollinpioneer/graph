"""E0: nothing hidden reaches the policy side."""
import collections

import pytest

from .helpers.alfworld_fixture import FakePrior, gamepath, load_all, require_data


def test_env_started_without_fact_request(monkeypatch):
    require_data()
    import textworld
    from cp_disr.platforms.alfworld.adapter import AlfEpisode

    rows, splits, tables = load_all()
    seen = {}
    real = textworld.EnvInfos

    def spy(**kw):
        seen.update(kw)
        return real(**kw)

    monkeypatch.setattr(textworld, "EnvInfos", spy)
    ep = AlfEpisode(gamepath(splits["dev"][0]), tables)
    ep.close()
    assert not seen.get("facts") and not seen.get("extras") and not seen.get("policy_commands") and not seen.get("walkthrough")


def _pairs(rows, splits):
    by = collections.defaultdict(list)
    for r in rows:
        key = (tuple(sorted(x["name"] for x in r["receptacles"])), r["goal_otype"], r["goal_rtype"])
        by[key].append(r)
    out = []
    for key, rs in by.items():
        hold = lambda r: frozenset(t["receptacle"] for t in r["oracle"]["targets"] if t["receptacle"])
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                if hold(rs[i]) != hold(rs[j]):
                    out.append((rs[i], rs[j]))
    return out


def test_same_public_history_gives_identical_snapshots():
    """Two worlds with the same announced room/goal but different hidden placement must be
    indistinguishable to the policy until a check actually reveals a difference."""
    require_data()
    from cp_disr.platforms.alfworld.adapter import AlfEpisode
    from cp_disr.platforms.alfworld.snapshot import build_snapshot, prior_edges_for
    from cp_disr.platforms.alfworld.pddl_contracts import episode_template
    from cp_disr.baselines.alfworld_scripts import natural_key

    rows, splits, tables = load_all()
    pairs = _pairs(rows, splits)
    assert pairs, "no pair of games with equal public start and different hidden placement"
    prior = FakePrior()
    compared = 0
    for ra, rb in pairs[:40]:
        ea, eb = AlfEpisode(gamepath(ra["gamefile"]), tables), AlfEpisode(gamepath(rb["gamefile"]), tables)
        try:
            for step in range(6):
                pa, pb = ea.public(), eb.public()
                tpl = episode_template(pa.feasible, pa.goal_instance, pa.goal_otype)
                sa = build_snapshot(pa, prior_edges_for(tpl, prior.scores(pa, tables)), "e", "x", step)
                sb = build_snapshot(pb, prior_edges_for(tpl, prior.scores(pb, tables)), "e", "x", step)
                assert pa == pb
                assert sa.facts.values == sb.facts.values and sa.mask == sb.mask and sa.prior_hash == sb.prior_hash
                assert sa.base_input == sb.base_input and sa.candidate_ids == sb.candidate_ids
                compared += 1
                checks = sorted((a for a in pa.legal if a[0] == "CHECK"), key=lambda a: natural_key(a[1]))
                if not checks or ea.done or eb.done:
                    break
                ea.execute(*checks[0])
                eb.execute(*checks[0])
                if ea.checked[checks[0][1]] != eb.checked[checks[0][1]]:
                    break  # public histories diverge from here on; equality is no longer required
        finally:
            ea.close()
            eb.close()
    assert compared >= 40


def test_no_hidden_object_id_in_template_or_snapshot():
    require_data()
    from cp_disr.platforms.alfworld.adapter import AlfEpisode
    from cp_disr.platforms.alfworld.snapshot import build_snapshot

    rows, splits, tables = load_all()
    by = {r["gamefile"]: r for r in rows}
    for rel in splits["dev"][:25]:
        ep = AlfEpisode(gamepath(rel), tables)
        try:
            snap = build_snapshot(ep.public(), (), "e", "x", 0)
            blob = repr(snap.template) + repr(snap.facts) + repr(snap.candidate_ids) + repr(snap.base_input)
            for t in by[rel]["oracle"]["targets"]:
                assert t["name"] not in blob
            assert all(n.arguments == () or all(a in ep.receptacles for a in n.arguments) for n in snap.template.nodes)
        finally:
            ep.close()


def test_prior_request_contains_only_public_inputs():
    from cp_disr.platforms.alfworld.prior_provider import build_messages, prior_key

    msgs = build_messages("alarmclock", ["bed", "desk", "drawer"])
    text = " ".join(m["content"] for m in msgs)
    assert "alarmclock" in text and "desk" in text
    assert not any(ch.isdigit() for ch in text.replace("between 0 and 1", ""))  # no instance ids / numbers leak in
    assert prior_key("alarmclock", ["drawer", "bed", "desk", "bed"]) == "alarmclock|bed,desk,drawer"
