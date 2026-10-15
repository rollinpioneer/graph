#!/usr/bin/env python3
"""Independent SMTP notifier for the detached CUPID training supervisor."""

from __future__ import annotations

import argparse
import os
import smtplib
import ssl
import time
from email.message import EmailMessage
from pathlib import Path


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("'\"")
    values.update({key: os.environ[key] for key in
                   ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASS", "SMTP_TO")
                   if key in os.environ})
    return values


def send_email(env_file: Path, subject: str, body: str, retries: int, retry_sec: int) -> bool:
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            values = read_env(env_file)
            user = values["SMTP_USER"]
            message = EmailMessage()
            message["Subject"] = subject
            message["From"] = user
            message["To"] = values.get("SMTP_TO", user)
            message.set_content(body)
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(
                values.get("SMTP_HOST", "smtp.163.com"),
                int(values.get("SMTP_PORT", "465")),
                context=context,
                timeout=30,
            ) as server:
                server.login(user, values["SMTP_PASS"])
                server.send_message(message)
            return True
        except Exception as exc:  # notifier failure must not affect training
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(retry_sec)
    print(f"email failed after {retries} attempts: {last_error}", flush=True)
    return False


def pid_alive(pid_file: Path) -> bool:
    try:
        pid = int(pid_file.read_text().strip())
    except (FileNotFoundError, ValueError):
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def state(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def field(text: str, name: str, default: str = "unknown") -> str:
    prefix = name + "="
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix):]
    return default


def body(event: str, status_text: str, root: Path) -> str:
    return "\n".join([
        "CUPID formal training event",
        f"Event: {event}",
        f"Time: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"Attempt: {field(status_text, 'attempt')}",
        f"Restart count: {field(status_text, 'restart_count')}",
        f"Physical GPU: {field(status_text, 'physical_gpu', '1')}",
        f"Checkpoint resume: {field(status_text, 'checkpoint_resume', 'true')}",
        f"CUPID root: {root}",
    ])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--poll-sec", type=int, default=30)
    parser.add_argument("--email-retries", type=int, default=5)
    parser.add_argument("--email-retry-sec", type=int, default=60)
    args = parser.parse_args()

    status_dir = args.root / "status"
    logs = args.root / "logs"
    pid_file = logs / "formal_train_supervisor.pid"
    running_file = status_dir / "formal_train_supervised.running"
    pass_file = status_dir / "formal_train_supervised.pass"
    failure_file = status_dir / "formal_train_supervised.last_failure"
    monitor_log = logs / "formal_train_email_monitor.log"

    def log(message: str) -> None:
        with monitor_log.open("a", encoding="utf-8") as handle:
            handle.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")

    started = False
    last_attempt = None
    last_failure_signature = None
    log("independent notifier started")
    while True:
        running = state(running_file)
        passed = state(pass_file)
        failed = state(failure_file)
        attempt = field(running, "attempt", "")

        if not started and running:
            ok = send_email(args.env_file, "[CUPID] formal training started",
                            body("started", running, args.root),
                            args.email_retries, args.email_retry_sec)
            log(f"started notification sent={ok}")
            started = True
            last_attempt = attempt

        if running and attempt and attempt != last_attempt:
            ok = send_email(args.env_file, "[CUPID] formal training auto-restarted",
                            body("auto-restarted", running, args.root),
                            args.email_retries, args.email_retry_sec)
            log(f"restart notification attempt={attempt} sent={ok}")
            last_attempt = attempt

        if passed:
            ok = send_email(args.env_file, "[CUPID] formal training finished",
                            body("finished", passed, args.root),
                            args.email_retries, args.email_retry_sec)
            log(f"finished notification sent={ok}")
            return 0

        if failed:
            signature = failed
            if signature != last_failure_signature:
                ok = send_email(args.env_file, "[CUPID] formal training retry pending",
                                body("retry-pending", failed, args.root),
                                args.email_retries, args.email_retry_sec)
                log(f"retry notification sent={ok}")
                last_failure_signature = signature

        if started and not running and not passed and not pid_alive(pid_file):
            ok = send_email(args.env_file, "[CUPID] formal training terminated",
                            body("terminated-or-supervisor-lost", failed or running, args.root),
                            args.email_retries, args.email_retry_sec)
            log(f"termination notification sent={ok}")
            return 1

        time.sleep(args.poll_sec)


if __name__ == "__main__":
    raise SystemExit(main())
