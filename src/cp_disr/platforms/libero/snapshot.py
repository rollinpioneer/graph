"""Build policy snapshots from verifier facts, cache prior, and public observations."""
from __future__ import annotations

import json, os
from pathlib import Path
from cp_disr.facts import FactStore, Truth
from cp_disr.graph import build_template, Goal
from cp_disr.contracts import precondition_value
from cp_disr.common import digest
from cp_disr.rl import Snapshot
from cp_disr.adapters import ExecutionResult


class SnapshotBuilder:
    def __init__(self, template, cache_edges, env, clock, deadline):
        self.template = template
        self.cache_edges = tuple(cache_edges)
        self.env = env
        self.clock = clock
        self.deadline = float(deadline)
        self.env_id = os.environ.get("CP_DISR_ENV_ID", "d0-env-0")
        self.episode_id = "ep-0"

    def _mask(self, facts: FactStore):
        values = dict(facts.values)
        ids = []
        mask = []
        feats = []
        for c in self.template.contracts:
            ids.append(c.id)
            legal = precondition_value(c, values) == Truth.TRUE
            mask.append(bool(legal))
            feat = [float(hash(c.name) % 97) / 97.0, float(len(c.bound_arguments)), float(c.timeout_seconds)]
            feat += [float(hash(a) % 89) / 89.0 for a in c.bound_arguments]
            feat += [0.0] * (8 - len(feat))
            feats.append(tuple(feat[:8]))
        return tuple(ids), tuple(mask), tuple(feats)

    def _base(self, observation, execution_summary, now):
        prop = list(observation.proprioception)
        remain = max(0.0, self.deadline - now)
        extra = [now, remain, float(len(execution_summary))]
        vec = prop + extra
        # pad/trim to 48
        if len(vec) < 48:
            vec = vec + [0.0] * (48 - len(vec))
        return tuple(float(x) for x in vec[:48])

    def initial(self, facts: FactStore, observation, episode_id: str):
        self.episode_id = episode_id
        ids, mask, feats = self._mask(facts)
        now = self.clock.now_seconds()
        prior = self.cache_edges
        return Snapshot(
            env_id=self.env_id,
            episode_id=episode_id,
            decision_id=0,
            template=self.template,
            facts=facts,
            candidate_ids=ids,
            mask=mask,
            prior_edges=prior,
            prior_hash=digest(prior),
            base_input=self._base(observation, (), now),
            candidate_features=feats,
            observation_ref=observation.frame_id,
            clock_seconds=now,
            execution_summary=(),
            synthetic_unit_fixture=False,
        )

    def build(self, previous_snapshot, records, observation, execution, clock_seconds):
        facts = FactStore(tuple(records))
        ids, mask, feats = self._mask(facts)
        if isinstance(execution, dict):
            execution = ExecutionResult(
                execution_id=execution["execution_id"],
                controller_exit=execution["controller_exit"],
                start_seconds=execution["start_seconds"],
                end_seconds=execution["end_seconds"],
                evidence_ids=tuple(execution.get("evidence_ids") or ()),
            )
        summary = tuple(previous_snapshot.execution_summary) + ((execution.execution_id, execution.controller_exit),)
        return Snapshot(
            env_id=previous_snapshot.env_id,
            episode_id=previous_snapshot.episode_id,
            decision_id=previous_snapshot.decision_id + 1,
            template=self.template,
            facts=facts,
            candidate_ids=ids,
            mask=mask,
            prior_edges=previous_snapshot.prior_edges,
            prior_hash=previous_snapshot.prior_hash,
            base_input=self._base(observation, summary, clock_seconds),
            candidate_features=feats,
            observation_ref=observation.frame_id,
            clock_seconds=float(clock_seconds),
            execution_summary=summary,
            synthetic_unit_fixture=False,
        )


def load_cache_edges(cache_dir: Path | None):
    if not cache_dir:
        return ()
    path = Path(cache_dir) / "final_edges.json"
    if not path.is_file():
        path = Path(cache_dir) / "accepted_relations.json"
    if not path.is_file():
        return ()
    data = json.loads(path.read_text(encoding="utf-8"))
    edges = []
    rows = data if isinstance(data, list) else data.get("edges") or data.get("relations") or []
    for r in rows:
        if isinstance(r, (list, tuple)) and len(r) >= 3:
            edges.append((str(r[0]), str(r[1]), str(r[2])))
        elif isinstance(r, dict):
            src = r.get("source") or r.get("src") or r.get("from")
            dst = r.get("target") or r.get("dst") or r.get("to")
            rel = r.get("type") or r.get("relation")
            if src and dst and rel:
                edges.append((str(src), str(dst), str(rel)))
    return tuple(edges)
