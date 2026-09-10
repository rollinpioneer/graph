"""Run the module's pure unit tests and record the real, unedited output."""

from __future__ import annotations

import io
import json
import sys
import unittest
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        print("usage: record_tests.py <output.json>")
        return 2
    output = Path(arguments[0])
    module_root = Path(__file__).resolve().parent
    worktree_root = module_root.parents[1]
    suite = unittest.TestLoader().discover(
        str(module_root / "tests"),
        pattern="test_*.py",
        top_level_dir=str(worktree_root),
    )
    buffer = io.StringIO()
    result = unittest.TextTestRunner(stream=buffer, verbosity=2).run(suite)
    payload = {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "was_successful": result.wasSuccessful(),
        "command": "python -B -m upgrade_v2.l2r_geometry_events.record_tests <output>",
        "raw_output": buffer.getvalue(),
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("tests_run", "failures", "errors", "was_successful")}))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
