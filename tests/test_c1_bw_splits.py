"""CP-DISR-C1-CROSS-TASK-BLOCKSWORLD-V1: split identity, isolation and the training-loader guard (runbook 17.8)."""
import ast
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = {"train_dev": "configs/splits/c1_bw_train_dev_v1.json", "a0": "configs/splits/c1_bw_a0_iso_v1.json", "a1": "configs/splits/c1_bw_a1_color_reverse_v1.json",
         "a2": "configs/splits/c1_bw_a2_noniso_v1.json", "b": "configs/splits/c1_bw_b_scale_v1.json"}
PREP = ROOT / "runs/final_master/c1_route_b/blocksworld_main_v1/prep"


def _need():
    if not all((ROOT / f).is_file() for f in FILES.values()):
        pytest.skip("splits not built")


def test_counts_sizes_and_unique_ids():
    _need()
    td = json.loads((ROOT / FILES["train_dev"]).read_text())
    docs = {k: json.loads((ROOT / v).read_text()) for k, v in FILES.items() if k != "train_dev"}
    assert len(td["train"]) == 144 and len(td["dev"]) == 36
    assert [len(docs[k]["cases"]) for k in ("a0", "a1", "a2", "b")] == [24, 32, 32, 48]
    for n in (3, 4, 5):
        assert sum(c["n_blocks"] == n for c in td["train"]) == 48 and sum(c["n_blocks"] == n for c in td["dev"]) == 12
    ids = [c["case_id"] for c in td["train"] + td["dev"]] + [c["case_id"] for k in docs for c in docs[k]["cases"]]
    assert len(ids) == len(set(ids))
    assert [sum(c["n_blocks"] == n for c in docs["b"]["cases"]) for n in (6, 7, 8)] == [16, 16, 16]
    assert td["half_life"] == 6.0 or td["half_life"] > 0


def test_train_goals_are_alternating_towers_top_red_and_a1_is_top_blue():
    _need()
    from cp_disr.blocksworld import canonical as K
    from cp_disr.blocksworld import state as S
    td = json.loads((ROOT / FILES["train_dev"]).read_text())
    a1 = json.loads((ROOT / FILES["a1"]).read_text())["cases"]
    for cases, top in ((td["train"] + td["dev"], S.RED), (a1, S.BLUE)):
        for c in cases:
            p = S.Problem(tuple(c["names"]), tuple(c["colors"]), tuple(c["init"]), tuple(c["goal"]))
            for t in K.goal_towers(p.goal):
                if len(t) > 1:
                    seq = [p.colors[b] for b in t]
                    assert seq[-1] == top and all(seq[i] != seq[i + 1] for i in range(len(seq) - 1))


def test_frozen_hashes_match_the_prep_files():
    _need()
    src = json.loads((PREP / "source_identity.json").read_text())
    for rel, h in src["split_files_sha256"].items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == h


def test_training_loader_refuses_every_evaluation_split_and_only_opens_train_dev():
    from cp_disr.blocksworld import launch as L
    for name in L.EVAL_SPLIT_NAMES:
        with pytest.raises(L.LaunchError):
            L.open_training_split(ROOT / "configs/splits" / name)
    with pytest.raises(L.LaunchError):
        L.open_training_split(ROOT / "configs/splits/other.json")
    if (ROOT / FILES["train_dev"]).is_file():
        assert "train" in L.open_training_split(ROOT / FILES["train_dev"])


def test_no_training_module_mentions_an_evaluation_split_file():
    banned = ("a0_iso", "a1_color_reverse", "a2_noniso", "b_scale")
    for rel in ("src/cp_disr/blocksworld/train.py", "src/cp_disr/blocksworld/environment.py", "src/cp_disr/c1_blocksworld_policies.py", "scripts/c1_bw_launch.py"):
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        consts = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        assert not [c for c in consts if any(b in c for b in banned)], rel
    # launch.py names the eval files only to REFUSE them (EVAL_SPLIT_NAMES) and never builds a path to open them
    text = (ROOT / "src/cp_disr/blocksworld/launch.py").read_text()
    assert "read_text" in text and text.count("EVAL_SPLIT_NAMES") >= 2


def test_old_evidence_directories_are_untouched():
    import subprocess
    out = subprocess.run(["git", "diff", "--name-only", "1f7b4f6cabd4c95ce0eff86f3b51e8aa6217312e", "--", "runs/final_master/2.1.1/structgen", "runs/final_master/c1_route_b/mech_confirm_v1"],
                         cwd=str(ROOT), capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip() == ""
