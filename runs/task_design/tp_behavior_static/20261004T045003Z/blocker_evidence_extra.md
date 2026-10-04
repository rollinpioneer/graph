# Supplementary blocker evidence (read-only greps over the pinned checkout bd049de3119acdcdf2334fe9e1ebe060fa20c108)

## files mentioning 'symbolic' outside docs/datasets/asset_pipeline
./AGENTS.md
./OmniGibson/omnigibson/action_primitives/symbolic_semantic_action_primitives.py
./OmniGibson/omnigibson/robots/robot.py
./OmniGibson/omnigibson/tasks/behavior_task.py
./OmniGibson/tests/benchmark/vector_profiling.py
./OmniGibson/tests/test_multiple_envs_behavior_task.py
./OmniGibson/tests/test_symbolic_primitives.py
./bddl3/bddl/__init__.py
./bddl3/docs/intro.md

## mock / stub / dummy environment classes
./OmniGibson/omnigibson/utils/asset_conversion_utils.py:794:    class DummyScene:
./OmniGibson/tests/test_evaluator.py:98:    class FakeEnv:

## omnigibson/__init__.py: Isaac Sim dependence
48:_og_logger.propagate = False  # prevent Isaac Sim's Carbonite handler from double-printing
51:builtins.ISAAC_LAUNCHED_FROM_JUPYTER = (
52:    os.getenv("ISAAC_JUPYTER_KERNEL") is not None
91:            Isaac Sim will be refreshed with this dt as well if running non-headless.
122:    usd_watcher = lazy.omni.kit.viewport.menubar.core.utils.usd_watch
131:    assert lazy.isaacsim.core.utils.stage.close_stage()
133:    lazy.isaacsim.core.api.SimulationContext.clear_instance()
157:        # If Isaac is running, we do the cleanup in its shutdown callback to avoid open handles.

## symbolic primitive tests (how the primitives are exercised)
import os

import pytest
import yaml

import omnigibson as og
from omnigibson import object_states
from omnigibson.action_primitives.symbolic_semantic_action_primitives import (
    SymbolicSemanticActionPrimitives,
    SymbolicSemanticActionPrimitiveSet,
)

# TODO: Using GPU dynamics causes cuda memory issues, need to investigate
# gm.USE_GPU_DYNAMICS = True
# gm.ENABLE_TRANSITION_RULES = True
current_robot_type = "R1"


def load_robot_config(robot_name):
    config_filename = os.path.join(og.example_config_path, f"{robot_name.lower()}_primitives.yaml")
    with open(config_filename, "r") as file:
        full_config = yaml.safe_load(file)
        robot_config = full_config.get("robots", {})[0]
        robot_config["disable_grasp_handling"] = True
        return robot_config


def start_env(robot_type):
    global current_robot_type
    if og.sim:

## sanity-seed activity names in this release
bag_groceries
baking_sugar_cookies
buying_groceries
buying_groceries_for_a_feast
carrying_in_groceries
delivering_groceries_to_doorstep
distributing_groceries_at_food_bank
make_lemonade
packing_grocery_bags_into_car
stock_grocery_shelves
unloading_groceries
(the card names making_lemonade and storing_the_groceries; neither exists under those names in v3.9.3-post1; they were not used as seeds and no case was hand-picked)
