from __future__ import annotations

import argparse
import json
from pathlib import Path

from .authorization import AuthorizationDenied, protocol_sha256, validate_authorization
from .protocol import load_protocol


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="l2r_forced_drop")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, stage in (("calibrate", "R16_CALIBRATION"), ("collect-development", "R16_DEVELOPMENT")):
        command = sub.add_parser(name)
        command.add_argument("--protocol", type=Path, required=True)
        command.add_argument("--authorization", type=Path, required=True)
        command.add_argument("--output-root", type=Path, required=True)
        command.set_defaults(stage=stage)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        protocol = load_protocol(args.protocol)
        # This check runs before importing simulator.py, which is the first module
        # that can reach the MuJoCo dependency.
        auth = validate_authorization(args.authorization, expected_stage=args.stage, expected_protocol_sha256=protocol_sha256(args.protocol))
    except (AuthorizationDenied, ValueError, OSError) as exc:
        print(json.dumps({"status": "DENIED", "reason": str(exc)}, sort_keys=True))
        return 3
    args.output_root.mkdir(parents=True, exist_ok=False)
    (args.output_root / "authorization_snapshot.json").write_text(json.dumps({"stage": auth.stage, "authorized_instances": auth.authorized_instances}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "AUTHORIZED_READY", "stage": auth.stage, "output_root": str(args.output_root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
