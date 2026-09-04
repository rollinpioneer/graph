#!/usr/bin/env python3
"""Independent email notifier for the Transport-MH 50% filtering pilot."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import smtplib
import ssl
import time
from email.message import EmailMessage
from pathlib import Path


EXPERIMENT_NAME = "CUPID Transport-MH Filter50 Pilot"
ARMS = ("baseline", "cupid50", "quality50", "random50")


def read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("'\"")
    return values


def field(text: str, key: str, default: str = "unknown") -> str:
    prefix = key + "="
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :]
    return default


def pid_alive(path: Path) -> bool:
    try:
        os.kill(int(path.read_text(encoding="ascii").strip()), 0)
        return True
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
        return False


def send(env_file: Path, subject: str, body: str) -> tuple[bool, str]:
    last_error = ""
    for attempt in range(1, 6):
        try:
            values = read_env(env_file)
            user = values["SMTP_USER"]
            message = EmailMessage()
            message["Subject"] = subject
            message["From"] = user
            message["To"] = values.get("SMTP_TO", user)
            message.set_content(body)
            with smtplib.SMTP_SSL(
                values.get("SMTP_HOST", "smtp.163.com"),
                int(values.get("SMTP_PORT", "465")),
                context=ssl.create_default_context(),
                timeout=30,
            ) as server:
                server.login(user, values["SMTP_PASS"])
                server.send_message(message)
            return True, ""
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < 5:
                time.sleep(60)
    return False, last_error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--poll-sec", type=int, default=30)
    args = parser.parse_args()

    experiment_id = "transport_filter50_pilot_20260725"
    status = args.root / "status" / experiment_id
    pids = args.root / "pids" / experiment_id
    logs = args.root / "logs" / experiment_id
    state_path = status / "email_state.json"
    log_path = logs / "email_monitor.log"
    lock_path = pids / "email_monitor.lock"
    pid_path = pids / "email_monitor.pid"
    status.mkdir(parents=True, exist_ok=True)
    pids.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    lock = lock_path.open("w", encoding="ascii")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return 2
    pid_path.write_text(f"{os.getpid()}\n", encoding="ascii")
    try:
        state = json.loads(read(state_path) or "{}")
    except json.JSONDecodeError:
        state = {}

    def log(message: str) -> None:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")

    def notify(key: str, event: str, detail: str) -> bool:
        if state.get(key):
            return True
        body = "\n".join(
            [
                f"Experiment: {EXPERIMENT_NAME}",
                f"Event: {event}",
                f"Time: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
                f"Detail: {detail}",
            ]
        )
        ok, error = send(
            args.env_file,
            f"[CUPID] {EXPERIMENT_NAME}: {event}",
            body,
        )
        log(f"event={event} key={key} sent={ok} error={error or 'none'}")
        if ok:
            state[key] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            state_path.write_text(
                json.dumps(state, indent=2, ensure_ascii=True) + "\n",
                encoding="ascii",
            )
        return ok

    log("independent pilot email monitor started")
    notify(
        "started",
        "started",
        "baseline eval + cupid50/quality50/random50 training launched on GPUs 2/1/4/5",
    )

    last_attempt = {
        arm: field(read(status / arm / "training.running"), "attempt", "")
        for arm in ARMS
    }
    try:
        while True:
            overall_pass = read(status / "overall.pass")
            overall_fail = read(status / "overall.fail")
            overall_terminated = read(status / "overall.terminated")
            if overall_pass:
                decision = field(overall_pass, "decision")
                notify("completed", "completed", f"decision={decision}")
                return 0
            if overall_fail:
                notify("unrecoverable", "unrecoverable-error", overall_fail.strip())
                return 1
            if overall_terminated:
                notify("terminated", "terminated", overall_terminated.strip())
                return 1

            for arm in ARMS:
                arm_status = status / arm
                terminated = read(arm_status / "pipeline.terminated")
                unrecoverable = read(arm_status / "pipeline.unrecoverable")
                if terminated:
                    notify(f"{arm}_terminated", "terminated", f"arm={arm}\n{terminated.strip()}")
                if unrecoverable:
                    notify(
                        f"{arm}_unrecoverable",
                        "unrecoverable-error",
                        f"arm={arm}\n{unrecoverable.strip()}",
                    )

                training = read(arm_status / "training.running")
                attempt = field(training, "attempt", "")
                if attempt and last_attempt.get(arm) and attempt != last_attempt[arm]:
                    notify(
                        f"{arm}_restart_{attempt}",
                        "restarted",
                        f"arm={arm}; attempt={attempt}; checkpoint_resume=true",
                    )
                if attempt:
                    last_attempt[arm] = attempt

                running = read(arm_status / "pipeline.running")
                passed = read(arm_status / "pipeline.pass")
                if running and not passed and not pid_alive(pids / arm / "supervisor.pid"):
                    notify(
                        f"{arm}_supervisor_lost",
                        "supervisor-lost",
                        f"arm={arm}; running marker exists but supervisor PID is absent",
                    )
                    return 1

            overall_running = read(status / "overall.running")
            if overall_running and not pid_alive(pids / "finalize.pid"):
                notify(
                    "finalizer_lost",
                    "supervisor-lost",
                    "overall finalizer PID disappeared before a terminal marker",
                )
                return 1
            time.sleep(args.poll_sec)
    finally:
        pid_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
