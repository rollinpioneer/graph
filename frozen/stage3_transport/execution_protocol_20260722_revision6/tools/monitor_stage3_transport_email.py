#!/usr/bin/env python3
"""Independent SMTP notifier for the detached Transport-MH supervisor."""

from __future__ import annotations

import argparse
import fcntl
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
    return values


def send(env_file: Path, subject: str, body: str, retries: int, retry_sec: int) -> bool:
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
            with smtplib.SMTP_SSL(
                values.get("SMTP_HOST", "smtp.163.com"),
                int(values.get("SMTP_PORT", "465")),
                context=ssl.create_default_context(), timeout=30,
            ) as server:
                server.login(user, values["SMTP_PASS"])
                server.send_message(message)
            return True
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(retry_sec)
    print(f"email failed after {retries} attempts: {last_error}", flush=True)
    return False


def read(path: Path) -> str:
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


def pid_alive(path: Path) -> bool:
    try:
        os.kill(int(path.read_text().strip()), 0)
        return True
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
        return False


def make_body(event: str, status: str) -> str:
    return "\n".join([
        "CUPID Stage 3 Transport-MH training event",
        f"Event: {event}",
        f"Time: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"Attempt: {field(status, 'attempt')}",
        f"Restart count: {field(status, 'restart_count')}",
        f"Return code: {field(status, 'return_code')}",
        f"Physical GPU: {field(status, 'physical_gpu', '1')}",
        f"Checkpoint resume: {field(status, 'checkpoint_resume', 'true')}",
        f"Train directory: {field(status, 'train_dir')}",
    ])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--poll-sec", type=int, default=30)
    parser.add_argument("--email-retries", type=int, default=5)
    parser.add_argument("--email-retry-sec", type=int, default=60)
    args = parser.parse_args()
    status_dir = args.root / "status/stage3_transport"
    pid_file = args.root / "pids/stage3_transport/train_supervisor.pid"
    log_path = args.root / "logs/stage3_transport/email_monitor.log"
    running_file = status_dir / "train.running"
    pass_file = status_dir / "train.pass"
    failure_file = status_dir / "train.last_failure"
    terminated_file = status_dir / "train.terminated"
    unrecoverable_file = status_dir / "train.unrecoverable"
    monitor_pid = args.root / "pids/stage3_transport/email_monitor.pid"
    lock_path = args.root / "pids/stage3_transport/email_monitor.lock"
    monitor_pid.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another Transport email monitor already holds the lock", flush=True)
        return 2
    monitor_pid.write_text(f"{os.getpid()}\n", encoding="utf-8")

    def log(message: str) -> None:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")

    started = False
    last_attempt = ""
    last_failure = ""
    log("independent Transport notifier started")
    try:
        while True:
            running = read(running_file)
            passed = read(pass_file)
            failed = read(failure_file)
            terminated = read(terminated_file)
            unrecoverable = read(unrecoverable_file)
            attempt = field(running, "attempt", "")
            if running and not started:
                ok = send(args.env_file, "[CUPID] Transport training started",
                          make_body("started", running), args.email_retries,
                          args.email_retry_sec)
                log(f"started notification sent={ok}")
                started, last_attempt = True, attempt
            if running and attempt and last_attempt and attempt != last_attempt:
                ok = send(args.env_file, "[CUPID] Transport training restarted",
                          make_body("auto-restarted", running), args.email_retries,
                          args.email_retry_sec)
                log(f"restart notification attempt={attempt} sent={ok}")
                last_attempt = attempt
            if passed:
                ok = send(args.env_file, "[CUPID] Transport training finished",
                          make_body("finished", passed), args.email_retries,
                          args.email_retry_sec)
                log(f"finished notification sent={ok}")
                return 0
            if terminated or unrecoverable:
                terminal = terminated or unrecoverable
                event = "terminated" if terminated else "unrecoverable failure"
                ok = send(args.env_file, f"[CUPID] Transport training {event}",
                          make_body(event, terminal), args.email_retries,
                          args.email_retry_sec)
                log(f"terminal notification event={event} sent={ok}")
                return 1
            if failed and failed != last_failure:
                ok = send(args.env_file, "[CUPID] Transport training retry pending",
                          make_body("retry-pending", failed), args.email_retries,
                          args.email_retry_sec)
                log(f"retry notification sent={ok}")
                last_failure = failed
            if started and not running and not pid_alive(pid_file):
                ok = send(args.env_file, "[CUPID] Transport supervisor lost",
                          make_body("supervisor-lost", failed), args.email_retries,
                          args.email_retry_sec)
                log(f"lost notification sent={ok}")
                return 1
            time.sleep(args.poll_sec)
    finally:
        monitor_pid.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
