# RoboCasa: RemoveCuttingBoardItems and PlaceDishesBySink (docs/metadata-only, C line)

Date 2026-10-01. Method: WebFetch/WebSearch only. Nothing installed, cloned, downloaded or run.
CAVEAT ON FIDELITY: WebFetch passes pages through a small summarizer model. Code blocks below were returned
as "verbatim" but are not byte-verified; claims marked [summarizer] are paraphrases. Re-read on GitHub before relying on any line.
No decision on eligibility/readiness/T_P is made here.

Base URL for source: https://raw.githubusercontent.com/robocasa/robocasa/main/  (branch main, package version 1.0.1 per setup.py; no commit hash obtained)

## 0. Naming / registry (both tasks)
- Registry: `REGISTERED_KITCHEN_ENVS` dict keyed by class `__name__`, auto-registered by `KitchenEnvMeta` (robocasa/environments/kitchen/kitchen.py; robocasa/environments/__init__.py exposes ALL_KITCHEN_ENVIRONMENTS). So registry id == class name. Gym id form in docs: `robocasa/<TaskName>` (https://robocasa.ai/docs/build/html/introduction/basic_usage.html, example `robocasa/PickPlaceCounterToCabinet`).
- Both are COMPOSITE tasks (source dir `robocasa/environments/kitchen/composite/<activity>/`; dataset_registry.py `COMPOSITE_TASK_DATASETS`). Neither is atomic.
- Version: dataset paths are "v1.0/pretrain/composite/..." (RoboCasa365, release 2026-02-18 per README); package version 1.0.1 (README: v1.0.1 on 5/12/2026 "updated horizon lengths"). Neither task is in the v0.2 paper's task set: v0.2 source tag has no remove_cutting_board_items.py / place_dishes_by_sink.py (404 on both), and the v0.2 paper (https://arxiv.org/html/2406.02523v1) mentions neither name. NOT verified: the exact commit that introduced them.
- Not in the RoboCasa365 paper's visible appendix tables 7-9 (https://arxiv.org/pdf/2603.04356) [summarizer; tables 8/9 may have been truncated]. Not in LeRobot RoboCasa doc, RLinf doc, or Lightwheel pages (third-party). dataset_registry.py [summarizer]: not in pretrain50/100/200/300 lists or target lists; only in COMPOSITE_TASK_DATASETS (UNVERIFIED because the summarizer may have mis-scanned a long file).
- Variants to NOT confuse (all official-class names, v0.2 and/or main):
  * `ClearingTheCuttingBoard` (chopping_food; v0.2 source verified at tag v0.2; lang "Clear the non-vegetable object off the cutting board and place the vegetables onto it.") != RemoveCuttingBoardItems. Lightwheel lists it as "Robocasa-Task-ClearingTheCuttingBoard" (third party).
  * `ScrubCuttingBoard` (same activity folder as RemoveCuttingBoardItems, sanitizing_cutting_board).
  * `DryDishes`, `StackBowls`/StackBowlsInSink (washing_dishes), `PickPlaceCounterToSink`, `PickPlaceSinkToCounter` (atomic).
  * Docstring of RemoveCuttingBoardItems says "Clear Cutting Board:" (title) but class/registry name is RemoveCuttingBoardItems.
  * Lightwheel docs (docs.lightwheel.net) use "Robocasa-Task-<Name>" ids; those pages do NOT list either of our two tasks.

## 1. RemoveCuttingBoardItems
Source: .../robocasa/environments/kitchen/composite/sanitizing_cutting_board/remove_cutting_board_items.py (raw URL above + that path). Activity: "Sanitizing Cutting Board".
- Language instruction: f"Remove the {vegetable_lang} and {meat_lang} from the cutting board to prepare it for cleaning." No novel-instruction branch in this file (ScrubCuttingBoard has one).
- Fixtures/objects: SINK fixture; COUNTER ref=sink; robot base ref = counter. Objects: `cutting_board` (obj_groups cutting_board, on counter left/right of sink, size 0.5x0.5, init_robot_here), `vegetable` and `meat` (graspable, placed ON the cutting board).
- _check_success: vegetable not in receptacle cutting_board AND meat not in receptacle cutting_board (OU.check_obj_in_receptacle) AND gripper far from both (OU.gripper_obj_far). Success does not require them to be placed anywhere specific (off-board only) -- read from code, not run.
- Horizon: dataset_registry.py `horizon=750`. Kitchen base default horizon=1000, control_freq=20, ignore_done=True (kitchen.py) -> the 750 is a registry value; whether it is enforced when you call robosuite.make is NOT verified. Whether 750 is the 1.5x-raised v1.0.1 value is NOT verified.
- Dataset entry: `pretrain=dict(human_path="v1.0/pretrain/composite/RemoveCuttingBoardItems/20250805")`; no mg_path, no target, no real.

## 2. PlaceDishesBySink
Source: .../composite/washing_dishes/place_dishes_by_sink.py. Activity: "Washing Dishes".
- Language instruction: f"Pick the {dish0_lang} and {dish1_lang} from the counter and place them on the counter next to the sink."
- Fixtures/objects: SINK, STOVE, COUNTER ref=stove (source), COUNTER ref=sink (target); base ref = stove; sink handle state set "off". Objects: dish0 (cup) and dish1 (bowl), both on the stove-side counter.
- Docstring inconsistency: class docstring says "use the pick-and-place skill ... Steps: locate the cabinet with cleaning items, open the cabinet door ..." which does not match the code (no cabinet in cfgs). Stale copy-paste; code defines behavior. The word "skill" is prose only, not an API.
- _check_success: both dishes within 0.15 m (bbox min-dist, OU.obj_fixture_bbox_min_dist) of sink AND both in contact with counter_sink (OU.check_obj_fixture_contact) AND gripper far from both.
- Horizon: dataset_registry.py `horizon=2400` (same caveats as above).
- Dataset entry: `pretrain=dict(human_path="v1.0/pretrain/composite/PlaceDishesBySink/20250717")`; no mg_path/target/real.

## 3. Embodiment / controller / action space (shared)
- Paper (RoboCasa365, https://arxiv.org/html/2603.04356v1): Franka Panda on Omron mobile base (PandaOmron), Operational Space Controller at 20 Hz; seven action dimensions (three translation, three rotation, one gripper) plus five dimensions for mobile base translation and rotation, torso height => 12-D total as described [summarizer; exact layout/ordering not verified]. kitchen.py: single robot assert; legacy "PandaMobile" renamed PandaOmron; composite controller body_part_ordering ['right','right_gripper','base','torso'] [summarizer].
- Mobile base + arm + torso; arm-space is 7-D in the paper's description, plus 5 base/torso dims. Whether these two tasks default to PandaOmron when created is not stated in what was read (kitchen.py default robot not confirmed).
- Success: sparse, reward = float(_check_success()).

## 4. Four-way separation
Legend: E=evidence found; N=no evidence found.

### RemoveCuttingBoardItems
| Item | Evidence? | Source | What is missing |
|---|---|---|---|
| 1 Env can be created | E (class exists in main; generic documented instantiation via gym.make("robocasa/<Task>", split=..., seed=...)) | source path above; https://robocasa.ai/docs/build/html/introduction/basic_usage.html | No evidence anyone (us) created it; no doc example specific to this task; unverified that gym.make id works for it (docs example is a different task); needs ~10 GB assets |
| 2 Human demos exist and replay | Partial: dataset path registered (human, pretrain, 20250805); replay tool exists | dataset_registry.py; https://robocasa.ai/docs/build/html/datasets/using_datasets.html (download_datasets, playback_dataset.py `--use-actions` open-loop replay) | Number of demos for this task and dataset size not confirmed (docs: 100 human demos/task for 300 pretrain tasks, 482 h total [summarizer]); playback script does NOT call _check_success (only checks deterministic state reproduction) [summarizer]; "replays reach success" is not documented. Replay NOT verified by us |
| 3 Model closes the loop | N for this task | RoboCasa365 paper only reports aggregates (atomic-seen 43.0% GR00T N1.5; composite-seen 9.6%; composite-unseen 4.4%) on TARGET tasks | No per-task result; task not in target lists (per summarizer, unverified); no published baseline found |
| 4 Callable atomic skill (PICK/PLACE style with pre/effects) | N | Paper: "foundational skills" (8, from Nasiriany et al. 2024) are task categories, not callable executors; MimicGen only for 60 atomic tasks | No scripted expert, motion planner, or skill API found; no mg_path for this composite task |

### PlaceDishesBySink
| Item | Evidence? | Source | What is missing |
|---|---|---|---|
| 1 Env can be created | E (class in main) | place_dishes_by_sink.py | same as above |
| 2 Human demos exist and replay | Partial: human pretrain path registered (20250717) | dataset_registry.py | same as above; horizon 2400 means long demos, size unknown |
| 3 Model closes the loop | N for this task | paper aggregates only | no per-task result found |
| 4 Callable atomic skill | N (docstring mentions "pick-and-place skill" in prose only) | place_dishes_by_sink.py | no executable skill found |

## 5. Skill / scripted-policy search (shared)
- README, docs overview/codebase overview/basic usage/datasets pages: no mention of scripted policy, skill library, or motion planning [summarizer]. robocasa/scripts has collect_demos.py, demo_teleop, playback_dataset (pure replay) [summarizer].
- MimicGen: used for 60 atomic tasks only (100 seeds -> 10k demos each); composite tasks human-only. Both our tasks are composite -> no MimicGen data.
- Not checked: full repo tree (github.com tree/search pages blocked by robots.txt; api.github.com tree 403). A hidden skill module elsewhere in the repo cannot be excluded; only "no evidence in pages read".

## 6. Integration conditions (documented)
- README/installation (https://github.com/robocasa/robocasa, https://robocasa.ai/docs/build/html/introduction/installation.html): conda Python 3.11; robosuite from ARISE-Initiative `master` branch (git clone, pip -e); robocasa `pip install -e .`; `python -m robocasa.scripts.setup_macros`; `python -m robocasa.scripts.download_kitchen_assets`; assets "around 10GB" (RLinf doc says ~5 GB; LeRobot doc downloads subset; aigen/objaverse registries ~30 GB optional).
- setup.py (main) [summarizer]: numpy==2.2.5, numba==0.61.2, scipy==1.15.3, mujoco==3.3.1, tianshou==0.4.10, lerobot==0.3.3, gymnasium, h5py, etc.
- robocasa/__init__.py [summarizer]: asserts mujoco 3.3.1, numpy 2.2.5, robosuite >=1.5.2; mimicgen optional. kitchen.py assumes robosuite >=1.5.1 controller conventions.
- Rendering: LeRobot doc says `export MUJOCO_GL=egl` for headless; official install page gives no GPU/EGL details.
- License: code MIT; assets and datasets CC BY 4.0 (README).
- Conflicts with our stack: robosuite 1.4 + LIBERO venv cannot satisfy robosuite>=1.5.2, mujoco==3.3.1, numpy==2.2.5, Python 3.11 (inference from the pins, not an official statement) => needs a SEPARATE environment; cannot be hot-swapped into the existing robosuite 1.4 venv. lerobot==0.3.3 pin reported to conflict with newer lerobot (LeRobot doc).
- RoboCasa's action space (mobile base + torso + arm, composite controller) differs from LIBERO's 7-D arm OSC; our skill-contract layer would need a different adapter.

## 7. Unknowns and minimal budget to answer them (nothing run; numbers from docs/estimates)
1. Do the classes exist at the commit we would pin and does gym.make work? -> fresh conda py3.11 venv + robosuite master + robocasa (pip, a few GB, ~10-20 min) + asset download (docs: ~10 GB) + one env-creation smoke (reset, step, render with EGL). Disk ~15-20 GB total.
2. Demo availability/size/replay: `download_datasets --tasks RemoveCuttingBoardItems PlaceDishesBySink --split pretrain --source human` (size not documented), then playback_dataset.py `--use-actions` plus an added _check_success check on the final state (the stock script has none).
3. Is there any callable skill/scripted policy? -> read the repo tree/scripts directory (blocked via web; answered by a clone or by hand-browsing the tree).
4. Which commit/tag first contains the two tasks, and whether horizons 750/2400 are pre- or post-1.0.1 values -> `git log -S` in a clone, or GitHub web UI blame.
5. Are the tasks in pretrain300/target lists, and any per-task success numbers -> read dataset_registry.py and the RoboCasa leaderboard (https://robocasa.ai/leaderboard.html, not fetched).
6. Default robot/controller when constructing these tasks, and actual arm-space action layout -> one env-creation smoke (item 1).

## 8. Pages not accessible or not fully read
- github.com tree/blob pages: blocked by robots.txt. api.github.com tree: 403. Source read via raw.githubusercontent.com through WebFetch only.
- robocasa.ai docs task tables returned only headers (65 atomic, 300 composite), no per-task rows.
- API doc pages under /docs/build/html/api/... returned 404.
- Paper tables 8/9 truncated in the fetch; leaderboard, release notes beyond README summary, HF dataset cards not read.
- kitchen.py default-robot line, setup.py/robocasa/__init__.py pins, dataset_registry.py membership claims are summarizer outputs.
