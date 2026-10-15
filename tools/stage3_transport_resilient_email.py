#!/usr/bin/env python3
"""Resilient, privacy-preserving notifier for the detached Transport-MH pipeline."""

from __future__ import annotations

import argparse
import json
import os
import re
import smtplib
import ssl
import sys
import time
from email.message import EmailMessage
from pathlib import Path

ARTICLE_ROOT = Path("/home/__compress_data/xushijie/article")
if str(ARTICLE_ROOT) not in sys.path:
    sys.path.insert(0, str(ARTICLE_ROOT))
from original_research.pudi_route_v2.monitor_p0a_privacy_email import smtp_settings


EXPERIMENT_NAME = "CUPID Transport-MH"
FAILURE_EVENT = re.compile(
    r"\bevent=(?:pipeline_fail|analysis_child_exit|smoke_oom_retry|trak_oom_retry|"
    r"filter_ids_freezer_fail|unrecoverable|supervisor_terminated)\b"
)


def send_experiment_email(env_file: Path, event: str) -> None:
    allowed = {"completed", "possible_stall", "attention_required"}
    if event not in allowed:
        raise ValueError("unsupported notification event")
    host, port, user, password, recipient = smtp_settings(env_file)
    message = EmailMessage()
    message["Subject"] = f"[Experiment Monitor] {EXPERIMENT_NAME}"
    message["From"] = user
    message["To"] = recipient
    message.set_content(
        f"Experiment: {EXPERIMENT_NAME}\n"
        f"Event: {event}\n"
        f"Time: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
    )
    with smtplib.SMTP_SSL(
        host, port, context=ssl.create_default_context(), timeout=30
    ) as server:
        server.login(user, password)
        server.send_message(message)


def exists(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_pid(path: Path | None) -> int | None:
    if path is None:
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return None


def process_cpu_ticks(pid_file: Path | None) -> int | None:
    pid = read_pid(pid_file)
    if pid is None:
        return None
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        return int(fields[11]) + int(fields[12])
    except (FileNotFoundError, IndexError, ValueError):
        return None


def log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
        handle.flush()


def load_state(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict) and isinstance(value.get("sent"), list):
            return value
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        pass
    return {"status": "watching", "sent": []}


def write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def send_once(
    *,
    env_file: Path,
    event: str,
    fingerprint: str,
    state_file: Path,
    log_file: Path,
    retry_sec: int,
) -> None:
    state = load_state(state_file)
    if fingerprint in state["sent"]:
        log(log_file, f"duplicate suppressed fingerprint={fingerprint}")
        return
    attempts = 0
    while True:
        attempts += 1
        try:
            send_experiment_email(env_file, event)
            state = load_state(state_file)
            state["sent"].append(fingerprint)
            state["status"] = "sent"
            state["last_event"] = event
            state["last_sent_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            state["last_attempts"] = attempts
            state["smtp_payload"] = "experiment_name_event_and_time_only"
            write_state(state_file, state)
            log(log_file, f"email sent event={event} fingerprint={fingerprint}")
            return
        except Exception as exc:
            log(log_file, f"email failed event={event} error_type={type(exc).__name__}")
            time.sleep(retry_sec)


def latest_activity(roots: list[Path]) -> float:
    latest = 0.0
    patterns = ("*.log", "*.ckpt", "ep*.pkl", "*.mmap", "*.json", "*.csv")
    for root in roots:
        if not root.exists():
            continue
        paths = [root] if root.is_file() else (
            item for item in root.rglob("*") if item.is_file()
        )
        for path in paths:
            if path.suffix not in {".log", ".ckpt", ".pkl", ".mmap", ".json", ".csv"}:
                continue
            try:
                latest = max(latest, path.stat().st_mtime)
            except OSError:
                continue
    return latest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--supervisor-pid", type=int, required=True)
    parser.add_argument("--secondary-supervisor-pid-file", type=Path)
    parser.add_argument("--activity-pid-file", type=Path)
    parser.add_argument("--pass-file", type=Path, required=True)
    parser.add_argument("--secondary-pass-file", type=Path, action="append", default=[])
    parser.add_argument("--failed-file", type=Path, required=True)
    parser.add_argument("--secondary-failed-file", type=Path, action="append", default=[])
    parser.add_argument("--running-file", type=Path, required=True)
    parser.add_argument("--event-log", type=Path, action="append", default=[])
    parser.add_argument("--artifact-root", type=Path, action="append", default=[])
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--poll-sec", type=int, default=15)
    parser.add_argument("--death-grace-sec", type=int, default=15)
    parser.add_argument("--stale-sec", type=int, default=900)
    parser.add_argument("--smtp-retry-sec", type=int, default=300)
    args = parser.parse_args()

    offsets: dict[str, int] = {}
    for path in args.event_log:
        try:
            offsets[str(path)] = path.stat().st_size
        except FileNotFoundError:
            offsets[str(path)] = 0
    start_time = time.time()
    last_cpu_ticks = process_cpu_ticks(args.activity_pid_file)
    last_cpu_activity = start_time
    log(
        args.log_file,
        f"resilient notifier started supervisor_pid={args.supervisor_pid} "
        f"stale_sec={args.stale_sec} "
        "smtp_payload=experiment_name_event_and_time_only",
    )

    while True:
        current_cpu_ticks = process_cpu_ticks(args.activity_pid_file)
        if current_cpu_ticks is not None and current_cpu_ticks != last_cpu_ticks:
            last_cpu_activity = time.time()
        last_cpu_ticks = current_cpu_ticks
        primary_pass = args.pass_file.is_file()
        secondary_pass = any(path.is_file() for path in args.secondary_pass_file)
        primary_failed = args.failed_file.is_file()
        secondary_failed = any(path.is_file() for path in args.secondary_failed_file)

        if primary_pass and secondary_pass:
            send_once(
                env_file=args.env_file,
                event="completed",
                fingerprint=f"completed:{args.supervisor_pid}",
                state_file=args.state_file,
                log_file=args.log_file,
                retry_sec=args.smtp_retry_sec,
            )
            return 0

        if primary_failed or secondary_failed:
            send_once(
                env_file=args.env_file,
                event="attention_required",
                fingerprint=f"failed_marker:{args.supervisor_pid}",
                state_file=args.state_file,
                log_file=args.log_file,
                retry_sec=args.smtp_retry_sec,
            )
            return 0

        for path in args.event_log:
            key = str(path)
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(offsets.get(key, 0))
                new_lines = handle.readlines()
                offsets[key] = handle.tell()
            if any(FAILURE_EVENT.search(line) for line in new_lines):
                send_once(
                    env_file=args.env_file,
                    event="attention_required",
                    fingerprint=f"child_failure:{args.supervisor_pid}",
                    state_file=args.state_file,
                    log_file=args.log_file,
                    retry_sec=args.smtp_retry_sec,
                )
                log(args.log_file, "new child failure event observed")
                # Continue watching so an automatic retry can still complete.

        primary_alive = exists(args.supervisor_pid)
        secondary_pid = read_pid(args.secondary_supervisor_pid_file)
        secondary_running = any(
            path.exists()
            for path in (args.secondary_pass_file + args.secondary_failed_file)
        ) or args.secondary_supervisor_pid_file is not None
        if not primary_alive and not primary_pass:
            time.sleep(args.death_grace_sec)
            if not args.pass_file.is_file():
                send_once(
                    env_file=args.env_file,
                    event="attention_required",
                    fingerprint=f"supervisor_exit:{args.supervisor_pid}",
                    state_file=args.state_file,
                    log_file=args.log_file,
                    retry_sec=args.smtp_retry_sec,
                )
                return 0
        if secondary_running and not exists(secondary_pid) and not secondary_pass:
            time.sleep(args.death_grace_sec)
            if not any(path.is_file() for path in args.secondary_pass_file):
                send_once(
                    env_file=args.env_file,
                    event="attention_required",
                    fingerprint=f"secondary_supervisor_exit:{secondary_pid}",
                    state_file=args.state_file,
                    log_file=args.log_file,
                    retry_sec=args.smtp_retry_sec,
                )
                return 0

        if args.running_file.is_file() and time.time() - start_time >= args.stale_sec:
            activity = latest_activity(args.artifact_root)
            files_stale = activity and time.time() - activity >= args.stale_sec
            process_stale = time.time() - last_cpu_activity >= args.stale_sec
            if files_stale and process_stale:
                send_once(
                    env_file=args.env_file,
                    event="possible_stall",
                    fingerprint=f"stalled:{args.supervisor_pid}",
                    state_file=args.state_file,
                    log_file=args.log_file,
                    retry_sec=args.smtp_retry_sec,
                )
                log(
                    args.log_file,
                    "artifact and child CPU activity stale for at least 900 seconds",
                )

        time.sleep(args.poll_sec)


if __name__ == "__main__":
    raise SystemExit(main())
