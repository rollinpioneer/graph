from cp_disr.baselines.b_plan import BPlanPlanner, SearchConfig
from cp_disr.contracts import Atom, Effects, SkillContract
from cp_disr.facts import FactRecord, FactStore, Truth
from cp_disr.graph import Goal, build_template


def contract(name, pre=(), pre_neg=(), add=(), delete=()):
    return SkillContract(
        name=name,
        arguments=(),
        pre_pos=tuple(pre),
        pre_neg=tuple(pre_neg),
        effects=Effects(add=tuple(add), delete=tuple(delete)),
        version="v1",
        provenance="unit-test",
        timeout_seconds=1.0,
    )


def facts(values):
    return FactStore(
        tuple(
            FactRecord(fact_id=key, value=value, capture_time=0.0, available_time=0.0)
            for key, value in sorted(values.items())
        )
    )


def template(contracts, goals):
    predicates = {
        "Ready": [],
        "Open": [],
        "Mid": [],
        "Done": [],
        "Flag": [],
    }
    return build_template(
        contracts=tuple(contracts),
        goals=tuple(goals),
        predicate_types=predicates,
        objects={},
    )


def test_b_plan_finds_canonical_chain():
    ready = Atom("Ready")
    opened = Atom("Open")
    done = Atom("Done")
    contracts = (
        contract("OPEN", pre=(ready,), add=(opened,)),
        contract("B", pre=(opened,), add=(done,)),
        contract("A", pre=(opened,), add=(done,)),
    )
    result = BPlanPlanner().plan(
        facts({"p:Ready": Truth.TRUE, "p:Open": Truth.FALSE, "p:Done": Truth.FALSE}),
        template(contracts, (Goal(done.id, 1),)),
        60.0,
    )
    assert result.status == "PLAN_FOUND"
    assert result.plan == ("a:OPEN:v1", "a:A:v1")


def test_b_plan_supports_negative_goals():
    flag = Atom("Flag")
    clear = contract("CLEAR", pre=(flag,), delete=(flag,))
    result = BPlanPlanner().plan(
        facts({"p:Flag": Truth.TRUE}),
        template((clear,), (Goal(flag.id, -1),)),
        60.0,
    )
    assert result.status == "PLAN_FOUND"
    assert result.plan == ("a:CLEAR:v1",)


def test_b_plan_enforces_depth_and_node_budgets():
    ready = Atom("Ready")
    mid = Atom("Mid")
    done = Atom("Done")
    contracts = (
        contract("A", pre=(ready,), add=(mid,)),
        contract("B", pre=(mid,), add=(done,)),
    )
    state = facts(
        {"p:Ready": Truth.TRUE, "p:Mid": Truth.FALSE, "p:Done": Truth.FALSE}
    )
    graph = template(contracts, (Goal(done.id, 1),))
    assert (
        BPlanPlanner(SearchConfig(depth_limit=1)).plan(state, graph, 60.0).status
        == "NO_PLAN"
    )
    assert (
        BPlanPlanner(SearchConfig(max_nodes=1)).plan(state, graph, 60.0).status
        == "SEARCH_TIMEOUT"
    )


def test_b_plan_does_not_execute_unknown_preconditions():
    ready = Atom("Ready")
    done = Atom("Done")
    action = contract("A", pre=(ready,), add=(done,))
    result = BPlanPlanner().plan(
        facts({"p:Ready": Truth.UNKNOWN, "p:Done": Truth.FALSE}),
        template((action,), (Goal(done.id, 1),)),
        60.0,
    )
    assert result.status == "NO_PLAN"
    assert result.plan == ()


def test_b_plan_defaults_are_frozen():
    config = SearchConfig()
    assert config.depth_limit == 6
    assert config.max_nodes == 4096
    assert config.cpu_time_limit_seconds == 2.0
    assert config.reference_skill_seconds == 4.2
