"""CHECK / TAKE / PUT macro contracts, derived from the ALFRED PDDL domain actions.

  CHECK(r) = GotoLocation(r) [+ OpenObject(r) when r is openable]; OpenObject adds checked(r),
             and the contents become observable (target presence stays UNKNOWN in the contract).
  TAKE(r)  = PickupObject(o, r): needs inReceptacle(o, r), hands empty; o:=holding, in(o,r):=F.
  PUT(g)   = GotoLocation(g) [+ OpenObject(g)] + PutObject(o, g): needs holding, canContain.

Templates only mention public objects (receptacle instances announced by the first text);
no hidden object identifier appears. TAKE/PUT refer to the target through the fixed constant
TARGET (class + instance bound at execution time), never through a hidden instance ID.
"""
from functools import lru_cache

from ...contracts import Atom, ConditionalEffect, Effects, SkillContract, TypedArgument
from ...facts import FactRecord, FactStore, Truth
from ...graph import Goal, build_template

PREDICATE_TYPES = {
    "checked": ("receptacle",),
    "target_in": ("receptacle",),
    "target_found": (),
    "holding_target": (),
    "target_placed": (),
}
GOAL_FACT = "p:target_placed"
KINDS = ("CHECK", "TAKE", "PUT")

_R = TypedArgument("r", "receptacle")
_G = TypedArgument("g", "receptacle")


def _atom(p, *a):
    return Atom(p, tuple(a))


def check_schema():
    return SkillContract(
        name="CHECK", arguments=(_R,),
        pre_pos=(), pre_neg=(_atom("checked", "r"), _atom("target_found"), _atom("holding_target")),
        effects=Effects(add=(_atom("checked", "r"),), unknown=(_atom("target_in", "r"),)),
        version="alfworld-v1", provenance="ALFRED PDDL GotoLocation+OpenObject",
        controller_ref="alfworld_macro", verifier_ref="alfworld_text", timeout_seconds=1.0,
    )


def take_schema():
    return SkillContract(
        name="TAKE", arguments=(_R,),
        pre_pos=(_atom("target_in", "r"),), pre_neg=(_atom("holding_target"),),
        effects=Effects(add=(_atom("holding_target"),), delete=(_atom("target_in", "r"),)),
        version="alfworld-v1", provenance="ALFRED PDDL PickupObject",
        controller_ref="alfworld_macro", verifier_ref="alfworld_text", timeout_seconds=1.0,
    )


def put_schema():
    return SkillContract(
        name="PUT", arguments=(_G,),
        pre_pos=(_atom("holding_target"),), pre_neg=(_atom("target_placed"),),
        effects=Effects(add=(_atom("target_placed"),), delete=(_atom("holding_target"),)),
        version="alfworld-v1", provenance="ALFRED PDDL PutObject",
        controller_ref="alfworld_macro", verifier_ref="alfworld_text", timeout_seconds=1.0,
    )


@lru_cache(maxsize=4096)
def episode_template(feasible, goal_instance):
    """Per-episode graph template over public objects only."""
    contracts = [check_schema().ground({"r": r}) for r in feasible]
    contracts += [take_schema().ground({"r": r}) for r in feasible]
    objects = {r: "receptacle" for r in feasible}
    if goal_instance is not None:
        contracts.append(put_schema().ground({"g": goal_instance}))
        objects[goal_instance] = "receptacle"
    extra = [_atom("target_found"), _atom("holding_target"), _atom("target_placed")]
    return build_template(contracts, [Goal(GOAL_FACT, 1)], PREDICATE_TYPES, objects, extra_atoms=extra)


def candidate_key(contract):
    """('CHECK', receptacle) style key for a grounded contract."""
    return contract.name, contract.bound_arguments[0]


def facts_from_public(template, pub):
    """FactStore values for the template, computed from public state only."""
    seen = dict(pub.target_in)
    checked = {r for r, _ in pub.checked}
    values = {}
    for n in template.nodes:
        if n.kind != "PROPOSITION":
            continue
        p = n.schema
        if p == "checked":
            v = Truth.TRUE if n.arguments[0] in checked else Truth.FALSE
        elif p == "target_in":
            r = n.arguments[0]
            v = (Truth.TRUE if seen[r] else Truth.FALSE) if r in checked else Truth.UNKNOWN
        elif p == "target_found":
            v = Truth.TRUE if pub.target_found else Truth.FALSE
        elif p == "holding_target":
            v = Truth.TRUE if pub.holding is not None else Truth.FALSE
        elif p == "target_placed":
            v = Truth.TRUE if pub.placed else Truth.FALSE
        else:
            raise KeyError(p)
        values[n.id] = v
    return values


def fact_store(values, clock):
    return FactStore(tuple(FactRecord(k, v, clock, clock) for k, v in values.items()))
