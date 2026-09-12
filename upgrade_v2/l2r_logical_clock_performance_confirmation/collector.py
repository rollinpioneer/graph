from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
from typing import Any

from upgrade_v2.l2r_logical_clock_confirmation.collector import collect_rollout
from upgrade_v2.l2r_logical_clock_confirmation.io_utils import write_json

from .registry import CASES, FAMILIES


def _task(args: tuple[str, int, int, Any, str]) -> dict:
    family, family_seed, rollout_seed, case, output = args
    return collect_rollout(Path(output), family, family_seed, rollout_seed, case)


def collect(output_root: Path, workers: int = 2) -> dict:
    if workers not in (1, 2): raise ValueError("WORKERS_MUST_BE_1_OR_2")
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = []
    for family, family_seed, seed_base in FAMILIES:
        for index, case in enumerate(CASES):
            root = output_root / f"{family}__{case.case_id}"
            if (root / "termination.json").is_file() and json.loads((root / "termination.json").read_text())["status"] == "COMPLETE":
                continue
            if root.exists(): raise RuntimeError(f"INCOMPLETE_ROLLOUT_PRESENT:{root}")
            tasks.append((family, family_seed, seed_base + index, case, str(root)))
    completed, failures = [], []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        pending, iterator, stop = {}, iter(tasks), False
        for _ in range(min(workers, len(tasks))):
            task = next(iterator, None)
            if task: pending[pool.submit(_task, task)] = task
        while pending:
            done, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                task = pending.pop(future)
                try: completed.append(future.result())
                except Exception as exc:
                    failures.append({"rollout": Path(task[-1]).name, "error": f"{type(exc).__name__}: {exc}"}); stop = True
                if not stop:
                    nxt = next(iterator, None)
                    if nxt: pending[pool.submit(_task, nxt)] = nxt
    terminations = list(output_root.glob("*/termination.json"))
    passed = len(terminations) == 32 and all(json.loads(path.read_text())["status"] == "COMPLETE" for path in terminations)
    result = {"schema": "l2rar2_r21_collection_status_v1", "status": "PASS" if passed else "FAIL",
              "all_32_complete": passed, "newly_completed": len(completed), "failures": failures}
    write_json(output_root / "collection_status.json", result)
    return result
