#!/usr/bin/env python3
"""Email terminal events for the detached Transport-MH analysis pipeline."""

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


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("'\"")
    return values


def pid(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def pid_alive(value: int | None) -> bool:
    if value is None:
        return False
    try:
        os.kill(value, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def send(env_file: Path, subject: str, body: str) -> tuple[bool, str]:
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
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--poll-sec", type=int, default=30)
    args = parser.parse_args()

    status = args.root / "status/stage3_transport"
    pids = args.root / "pids/stage3_transport"
    logs = args.root / "logs/stage3_transport"
    state_path = status / "pipeline_email_state.json"
    done_path = status / "pipeline_email_monitor.done"
    lock_path = pids / "pipeline_email_monitor.lock"
    pid_path = pids / "pipeline_email_monitor.pid"
    log_path = logs / "pipeline_email_monitor.log"
    pids.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    lock = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return 2
    pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")

    try:
        sent = json.loads(read_text(state_path) or "{}")
    except json.JSONDecodeError:
        sent = {}

    def log(message: str) -> None:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")

    def notify(key: str, subject: str, body: str) -> bool:
        if sent.get(key):
            return True
        ok, error = send(args.env_file, subject, body)
        log(f"event={key} sent={ok} error={error or 'none'}")
        if ok:
            sent[key] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            state_path.write_text(
                json.dumps(sent, ensure_ascii=True, indent=2) + "\n",
                encoding="utf-8",
            )
        return ok

    def body(event: str, detail: str) -> str:
        return "\n".join([
            "CUPID Stage 3 Transport-MH analysis event",
            f"Event: {event}",
            f"Time: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
            "Physical GPU: 1 (CUDA_VISIBLE_DEVICES=1, process cuda:0)",
            f"Detail: {detail}",
            f"CUPID root: {args.root}",
        ])

    log("independent pipeline notifier started")
    last_analysis_pid = sent.get("last_analysis_pid")
    try:
        while True:
            analysis_running = read_text(status / "analysis_pipeline.running")
            analysis_pass = read_text(status / "analysis_pipeline.pass")
            analysis_fail = read_text(status / "analysis_pipeline.fail")
            freezer_running = read_text(status / "filter_id_freezer.running")
            freezer_pass = read_text(status / "filter_id_freezer.pass")
            freezer_skip = read_text(status / "filter_id_freezer.skip")
            freezer_fail = read_text(status / "filter_id_freezer.fail")
            analysis_pid = pid(pids / "analysis_pipeline.pid")
            freezer_pid = pid(pids / "filter_id_freezer.pid")

            if analysis_running:
                current_pid = str(analysis_pid or "")
                if not sent.get("analysis_started"):
                    if notify(
                        "analysis_started",
                        "[CUPID] Transport analysis started/resumed",
                        body("analysis-started/resumed", analysis_running.strip()),
                    ):
                        sent["last_analysis_pid"] = current_pid
                        last_analysis_pid = current_pid
                        state_path.write_text(
                            json.dumps(sent, ensure_ascii=True, indent=2) + "\n",
                            encoding="utf-8",
                        )
                elif current_pid and last_analysis_pid and current_pid != last_analysis_pid:
                    key = f"analysis_restarted_pid_{current_pid}"
                    if notify(
                        key,
                        "[CUPID] Transport analysis restarted",
                        body("analysis-restarted", analysis_running.strip()),
                    ):
                        sent["last_analysis_pid"] = current_pid
                        last_analysis_pid = current_pid
                        state_path.write_text(
                            json.dumps(sent, ensure_ascii=True, indent=2) + "\n",
                            encoding="utf-8",
                        )

            terminal_failure = analysis_fail or freezer_fail
            if terminal_failure:
                failure_pid = analysis_pid or freezer_pid or "unknown"
                if notify(
                    f"unrecoverable_failure_{failure_pid}",
                    "[CUPID] Transport analysis unrecoverable failure",
                    body("unrecoverable-failure", terminal_failure.strip()),
                ):
                    done_path.write_text("UNRECOVERABLE_FAILURE_NOTIFIED\n", encoding="utf-8")
                    return 1

            terminated = read_text(status / "analysis_pipeline.terminated")
            if terminated:
                if notify(
                    "terminated",
                    "[CUPID] Transport analysis terminated",
                    body("terminated", terminated.strip()),
                ):
                    done_path.write_text("TERMINATED_NOTIFIED\n", encoding="utf-8")
                    return 1

            if analysis_running and not pid_alive(analysis_pid):
                if notify(
                    f"analysis_supervisor_lost_{analysis_pid or 'unknown'}",
                    "[CUPID] Transport analysis supervisor lost",
                    body("supervisor-lost", analysis_running.strip()),
                ):
                    done_path.write_text("SUPERVISOR_LOST_NOTIFIED\n", encoding="utf-8")
                    return 1

            if freezer_running and not pid_alive(freezer_pid):
                if notify(
                    f"freezer_supervisor_lost_{freezer_pid or 'unknown'}",
                    "[CUPID] Transport filter decision supervisor lost",
                    body("supervisor-lost", freezer_running.strip()),
                ):
                    done_path.write_text("SUPERVISOR_LOST_NOTIFIED\n", encoding="utf-8")
                    return 1

            if analysis_pass and (freezer_pass or freezer_skip):
                decision = freezer_pass or freezer_skip
                if notify(
                    "completed",
                    "[CUPID] Transport analysis completed",
                    body("completed", analysis_pass.strip() + "\n" + decision.strip()),
                ):
                    done_path.write_text("COMPLETION_NOTIFIED\n", encoding="utf-8")
                    return 0

            time.sleep(args.poll_sec)
    finally:
        pid_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
