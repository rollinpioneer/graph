"""Four isolated, once-only phase-C continuation workers; observer never serves scores.

This draft is not an experiment invocation. On the server its interface will be:
    python orchestration_draft.py ROOT
Workers have interface `python worker.py ROOT BRANCH CPU_ID` and append JSON lines
to branches/BRANCH/model_ledger.jsonl. END records require kind, sequence, status,
cache_key, value_bits_le (for OK). START records require kind and sequence.
observed_initial.json is {state_hex: score_bits}. END cache_key is state_hex.
The full task/domain/model/protocol namespace is frozen once in preregistration;
all workers use this single namespace, so the observer needs only full state hex.

All CPU accounting uses actual process CPU, not wall time multiplied by core count.
The parent does not reap via Popen.poll(), which would lose wait4 resource usage.
"""
from __future__ import annotations

import json
import math
import os
import signal
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


BRANCHES = ("A", "B", "B_repeat", "A_repeat")
FORMAL_CPU_CAP = 30_600.0
ACTUAL_EVALUATION_CAP = 160_000
AUXILIARY_BYTES_CAP = 1 << 30
AUXILIARY_FINALIZATION_RESERVE_BYTES = 32 << 20
OBSERVE_INTERVAL = 1.0
MAX_LEDGER_LINE_BYTES = 1 << 20


def write_json_once(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def canonical_key(key: Any) -> str:
    if not isinstance(key, str) or not key or format(int(key, 16), "x") != key:
        raise ValueError("cache_key must be canonical full state hexadecimal string")
    return key


def score_bits(value: str) -> str:
    if not isinstance(value, str) or len(value) != 8:
        raise ValueError("Expected exactly four FP32 bytes as eight hexadecimal digits")
    raw = bytes.fromhex(value)
    if not math.isfinite(struct.unpack("<f", raw)[0]):
        raise ValueError("Nonfinite model value in observational ledger")
    return raw.hex()


def group(branch: str) -> str:
    return "A" if branch.startswith("A") else "B"


@dataclass
class IncrementalTail:
    path: Path
    offset: int = 0
    remainder: bytes = b""
    identity: tuple[int, int] | None = None

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        info = self.path.stat()
        inode = (info.st_dev, info.st_ino)
        if self.identity is not None and inode != self.identity:
            raise ValueError(f"Ledger replaced: {self.path}")
        self.identity = inode
        if info.st_size < self.offset:
            raise ValueError(f"Ledger truncated: {self.path}")
        result: list[dict[str, Any]] = []
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            while chunk := handle.read(1 << 20):
                self.offset += len(chunk)
                parts = (self.remainder + chunk).split(b"\n")
                self.remainder = parts.pop()
                for line in parts:
                    if len(line) > MAX_LEDGER_LINE_BYTES:
                        raise ValueError("Observer ledger line exceeds registered bound")
                    if line.strip():
                        item = json.loads(line)
                        if not isinstance(item, dict):
                            raise ValueError("Observer ledger record must be an object")
                        result.append(item)
                if len(self.remainder) > MAX_LEDGER_LINE_BYTES:
                    raise ValueError("Unterminated observer ledger line exceeds bound")
        return result


def read_proc_cpu(pid: int) -> tuple[float, int] | None:
    """Return own user+system CPU seconds and PPID; never include child counters."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        fields = text[text.rfind(")") + 2:].split()
        # `fields[0]` is field 3 (state); field 14/15 are utime/stime.
        return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK"), int(fields[1])
    except (OSError, ValueError, IndexError):
        return None


def live_tree_cpu(roots: set[int]) -> tuple[float, list[int]]:
    """Read only descendants of our still-live workers, including transient helpers."""
    table: dict[int, tuple[float, int]] = {}
    for child in Path("/proc").iterdir():
        if child.name.isdigit():
            result = read_proc_cpu(int(child.name))
            if result is not None:
                table[int(child.name)] = result
    selected = set(roots)
    while True:
        more = {pid for pid, (_, parent) in table.items() if parent in selected}
        updated = selected | more
        if updated == selected:
            break
        selected = updated
    pids = sorted(pid for pid in selected if pid in table)
    return sum(table[pid][0] for pid in pids), pids


def auxiliary_size(root: Path) -> int:
    """ROOT auxiliary data only; each worker owns its separate branch byte cap."""
    total = 0
    for dirname, dirs, files in os.walk(root):
        if Path(dirname) == root:
            dirs[:] = [name for name in dirs if name != "branches"]
        for filename in files:
            try:
                total += (Path(dirname) / filename).lstat().st_size
            except FileNotFoundError:
                pass
    return total


def physical_core(cpu_id: int) -> tuple[str, str]:
    base = Path(f"/sys/devices/system/cpu/cpu{cpu_id}/topology")
    return (base.joinpath("physical_package_id").read_text().strip(),
            base.joinpath("core_id").read_text().strip())


@dataclass
class Worker:
    branch: str
    cpu_id: int
    tail: IncrementalTail
    process: subprocess.Popen[bytes] | None = None
    stdout: Any = None
    started_at: float | None = None
    returncode: int | None = None
    wait4: dict[str, Any] | None = None
    failure: str | None = None
    starts: set[int] = field(default_factory=set)
    ends: set[int] = field(default_factory=set)


class Observer:
    def __init__(self, root: Path):
        self.root = root
        self.reg = json.loads((root / "preregistration.json").read_text(encoding="utf-8"))
        self.observed: dict[str, tuple[str, str]] = {}
        self.mismatches: list[dict[str, Any]] = []
        self.request_count = 0
        self.completed_children_cpu = 0.0
        # Include interpreter/module loading CPU in the parent, even before __init__.
        self.cpu_at_start = 0.0
        self.wall_at_start = time.perf_counter()
        self.stop_reasons: list[str] = []
        self.log = (root / "observer_ledger.jsonl").open("x", encoding="utf-8", buffering=1)
        self.workers: list[Worker] = []

    def emit(self, kind: str, **record: Any) -> None:
        self.log.write(json.dumps({"kind": kind, "elapsed_wall_seconds": time.perf_counter() - self.wall_at_start,
                                   **record}, ensure_ascii=False) + "\n")

    def stop_all(self, reason: str) -> None:
        if reason not in self.stop_reasons:
            self.stop_reasons.append(reason)
            self.emit("SOFT_STOP_ALL", reason=reason)
        p = self.root / "stop_all"
        if not p.exists():
            write_json_once(p, {"reason": reason, "issued_by": "observer", "no_restart": True})

    def stop_group(self, name: str, reason: str, evidence: Any = None) -> None:
        p = self.root / ("stop_" + name)
        if not p.exists():
            write_json_once(p, {"reason": reason, "evidence": evidence, "issued_by": "observer", "no_restart": True})
        self.emit("SOFT_STOP_GROUP", group=name, reason=reason, evidence=evidence)

    def observe(self, key: Any, value: str, source: str, branch: str | None = None) -> None:
        key = canonical_key(key)
        value = score_bits(value)
        previous = self.observed.get(key)
        if previous is not None and previous[0] != value:
            mismatch = {"cache_key": key, "previous_bits": previous[0], "previous_source": previous[1],
                        "actual_bits": value, "actual_source": source}
            self.mismatches.append(mismatch)
            affected = {group(branch)} if branch is not None else {"A", "B"}
            if previous[1].split(":")[0] in BRANCHES:
                affected.add(group(previous[1].split(":")[0]))
            if previous[1].startswith("initial"):
                # Initial entries can be present in either treatment; stop both conservatively.
                affected.update(("A", "B"))
            for name in affected:
                self.stop_group(name, "SAME_KEY_SCORE_BYTES_CHANGED", mismatch)
            self.emit("SCORE_CONSISTENCY_FAILURE", **mismatch)
        elif previous is None:
            self.observed[key] = (value, source)

    def preflight(self) -> None:
        authorization = self.reg.get("authorization", {})
        if (authorization.get("parallel") != "APPROVED_BY_USER_OVERRIDE"
                or authorization.get("parallel_user_text") != "能并行吗，能的话就都并行"):
            raise ValueError("Missing exact latest-user parallel authorization in preregistration")
        if self.reg["protocol"] != "CPU_FP32_NATIVE_ONE_STATE_DETERMINISTIC_V1":
            raise ValueError("Unexpected scoring protocol")
        context = {"task_sha256": self.reg["task"]["sha256"],
                   "domain_sha256": self.reg["task"]["domain_sha256"],
                   "model_sha256": self.reg["model"]["sha256_actual"],
                   "protocol": self.reg["protocol"]}
        if any(len(context[name]) != 64 for name in ("task_sha256", "domain_sha256", "model_sha256")):
            raise ValueError("Incomplete fixed observational namespace identity")
        cpu_ids = self.reg["cpu_ids"]
        if len(cpu_ids) != 4 or len(set(cpu_ids)) != 4 or not all(isinstance(i, int) for i in cpu_ids):
            raise ValueError("Exactly four distinct integer CPU IDs required")
        if len({physical_core(i) for i in cpu_ids}) != 4:
            raise ValueError("CPU IDs must occupy four independent physical cores")
        if not set(cpu_ids).issubset(os.sched_getaffinity(0)):
            raise ValueError("Registered CPU ID is outside this process's available affinity")
        if not (self.root / "worker.py").is_file():
            raise ValueError("Missing frozen worker.py")
        if any((self.root / name).exists() for name in ("stop_all", "stop_A", "stop_B", "runner_receipt.json")):
            raise ValueError("Refusing to restart an already used or stopped execution root")
        initial = json.loads((self.root / "observed_initial.json").read_text(encoding="utf-8"))
        if not isinstance(initial, dict):
            raise ValueError("Initial observations must be state_hex -> FP32 bytes object")
        for number, (state, value) in enumerate(initial.items()):
            self.observe(state, value, f"initial:{number}")
        if self.mismatches:
            raise ValueError("Initial frozen observations contain inconsistent score bytes")
        if auxiliary_size(self.root) >= AUXILIARY_BYTES_CAP - AUXILIARY_FINALIZATION_RESERVE_BYTES:
            raise ValueError("Auxiliary space cap already reached")
        self.workers = [Worker(branch, cpu_id, IncrementalTail(self.root / "branches" / branch / "model_ledger.jsonl"))
                        for branch, cpu_id in zip(BRANCHES, cpu_ids)]
        self.emit("PRECHECK_PASS", cpu_ids=cpu_ids, branches=list(BRANCHES), initial_keys=len(self.observed),
                  observer_is_not_score_service=True, actual_precheck_model_calls=0, fixed_namespace=context)

    def launch_once(self) -> None:
        # All four starts are attempted in the same pass, without waiting for outcomes.
        # A launch failure preserves the slot and never triggers a retry.
        for worker in self.workers:
            try:
                # worker.py exclusively creates its branch directory with exist_ok=False.
                # Keep parent-owned stdout in the auxiliary namespace instead.
                output = self.root / "worker_stdout" / (worker.branch + ".log")
                output.parent.mkdir(parents=True, exist_ok=True)
                worker.stdout = output.open("xb", buffering=0)
                worker.started_at = time.perf_counter()
                worker.process = subprocess.Popen(
                    [sys.executable, "-u", str(self.root / "worker.py"), str(self.root), worker.branch, str(worker.cpu_id)],
                    stdin=subprocess.DEVNULL, stdout=worker.stdout, stderr=subprocess.STDOUT,
                    env=os.environ.copy(), start_new_session=True,
                    preexec_fn=lambda cpu_id=worker.cpu_id: os.sched_setaffinity(0, {cpu_id}),
                )
                self.emit("WORKER_LAUNCHED", branch=worker.branch, cpu_id=worker.cpu_id, pid=worker.process.pid,
                          attempt=1, independent_process=True)
            except Exception as err:
                worker.failure = repr(err)
                self.emit("WORKER_LAUNCH_FAILED", branch=worker.branch, error=worker.failure, attempt=1)
                self.stop_group(group(worker.branch), "INFRASTRUCTURE_LAUNCH_FAILED", {"branch": worker.branch, "error": worker.failure})

    def tail_worker(self, worker: Worker) -> None:
        for record in worker.tail.read():
            kind = record.get("kind")
            if kind == "MODEL_STATE_START":
                sequence = int(record["sequence"])
                if sequence in worker.starts:
                    raise ValueError(f"Duplicate START {worker.branch}:{sequence}")
                worker.starts.add(sequence)
                self.request_count += 1
            elif kind == "MODEL_STATE_END":
                sequence = int(record["sequence"])
                if sequence not in worker.starts or sequence in worker.ends:
                    raise ValueError(f"Unpaired or duplicate END {worker.branch}:{sequence}")
                worker.ends.add(sequence)
                if record["status"] == "OK":
                    self.observe(record["cache_key"], record["value_bits_le"], f"{worker.branch}:{sequence}", worker.branch)
                elif record["status"] not in ("INTERRUPTED", "ERROR"):
                    raise ValueError("Unknown model call status")
                elif record["status"] == "ERROR":
                    self.stop_group(group(worker.branch), "MODEL_FORWARD_ERROR", record)
            elif kind == "SCORE_CONSISTENCY_FAILURE" or record.get("technical_error"):
                self.stop_group(group(worker.branch), "WORKER_TECHNICAL_FAILURE", record)

    def reap_worker(self, worker: Worker) -> None:
        process = worker.process
        if process is None or worker.wait4 is not None:
            return
        pid, status, usage = os.wait4(process.pid, os.WNOHANG)
        if not pid:
            return
        worker.returncode = os.waitstatus_to_exitcode(status)
        # Mark Popen reaped, without calling its wait/poll methods.
        process.returncode = worker.returncode
        worker.wait4 = {"pid": pid, "exit_code": worker.returncode, "user_cpu_seconds": usage.ru_utime,
                        "system_cpu_seconds": usage.ru_stime, "CPU_seconds": usage.ru_utime + usage.ru_stime,
                        "max_rss_KiB": usage.ru_maxrss, "raw_wait_status": status}
        self.completed_children_cpu += worker.wait4["CPU_seconds"]
        if worker.stdout is not None:
            worker.stdout.close()
        # Drain the final ledger suffix after wait4, when the file cannot still append.
        self.tail_worker(worker)
        if worker.tail.remainder:
            worker.failure = "INFRASTRUCTURE_INCOMPLETE_LEDGER_LINE"
        receipt = self.root / "branches" / worker.branch / "receipt.json"
        if worker.returncode != 0 or not receipt.is_file():
            worker.failure = worker.failure or "INFRASTRUCTURE_EXIT_OR_RECEIPT_MISSING"
        if worker.failure:
            self.stop_group(group(worker.branch), worker.failure, worker.wait4)
        self.emit("WORKER_EXIT", branch=worker.branch, wait4=worker.wait4, failure=worker.failure,
                  ledger_starts=len(worker.starts), ledger_ends=len(worker.ends))

    def sample_resources(self) -> None:
        alive = {w.process.pid for w in self.workers if w.process is not None and w.wait4 is None}
        running_cpu, running_pids = live_tree_cpu(alive)
        own_cpu = time.process_time() - self.cpu_at_start
        total_cpu = own_cpu + self.completed_children_cpu + running_cpu
        aux = auxiliary_size(self.root)
        self.emit("RESOURCE_SAMPLE", formal_CPU_seconds=total_cpu, orchestrator_CPU_seconds=own_cpu,
                  completed_child_CPU_seconds=self.completed_children_cpu, live_tree_CPU_seconds=running_cpu,
                  live_pids=running_pids, requested_model_states=self.request_count, auxiliary_bytes=aux)
        if total_cpu >= FORMAL_CPU_CAP:
            self.stop_all("FORMAL_CPU_LIMIT")
        if self.request_count >= ACTUAL_EVALUATION_CAP:
            self.stop_all("GLOBAL_MODEL_STATE_LIMIT")
        if aux >= AUXILIARY_BYTES_CAP - AUXILIARY_FINALIZATION_RESERVE_BYTES:
            self.stop_all("AUXILIARY_SPACE_LIMIT")

    def run(self) -> dict[str, Any]:
        failure: str | None = None
        try:
            self.preflight()
            self.launch_once()
            while any(w.process is not None and w.wait4 is None for w in self.workers):
                try:
                    for worker in self.workers:
                        self.tail_worker(worker)
                        self.reap_worker(worker)
                    self.sample_resources()
                except Exception as err:
                    failure = repr(err)
                    self.stop_all("OBSERVER_FAILURE:" + failure)
                time.sleep(OBSERVE_INTERVAL)
            for worker in self.workers:
                self.tail_worker(worker)
            self.sample_resources()
        except Exception as err:
            failure = repr(err)
            self.stop_all("ORCHESTRATOR_FAILURE:" + failure)
            # This is a soft-stop protocol. Reap owned workers; never restart slots.
            while any(w.process is not None and w.wait4 is None for w in self.workers):
                for worker in self.workers:
                    try:
                        self.reap_worker(worker)
                    except Exception as reap_err:
                        self.emit("REAP_OR_LEDGER_ERROR", branch=worker.branch, error=repr(reap_err))
                time.sleep(OBSERVE_INTERVAL)
        own_cpu = time.process_time() - self.cpu_at_start
        rows = []
        for branch in BRANCHES:
            worker = next((w for w in self.workers if w.branch == branch), None)
            rows.append({"branch": branch, "slot_retained": True, "launch_attempts": 0 if worker is None else 1,
                         "pid": None if worker is None or worker.process is None else worker.process.pid,
                         "cpu_id": None if worker is None else worker.cpu_id,
                         "wait4": None if worker is None else worker.wait4,
                         "failure": failure if worker is None else worker.failure,
                         "model_starts": 0 if worker is None else len(worker.starts),
                         "model_ends": 0 if worker is None else len(worker.ends)})
        receipt = {"card": "CP-DISR-Q2-P12-FINAL-CONTINUATION-V1", "slots": rows,
                   "fixed_launch_order": list(BRANCHES), "formal_execution_mode": "FOUR_ISOLATED_PARALLEL_PROCESSES",
                   "authorization_override": self.reg.get("authorization", {}).get("parallel_user_text"),
                   "orchestrator_CPU_seconds": own_cpu, "actual_children_wait4_CPU_seconds": self.completed_children_cpu,
                   "formal_CPU_seconds": own_cpu + self.completed_children_cpu,
                   "wall_seconds": time.perf_counter() - self.wall_at_start,
                   "actual_model_state_requests_including_interrupted": self.request_count,
                   "counts": {"actual_model_state_evaluations": self.request_count},
                   "process_cpu_seconds": own_cpu + self.completed_children_cpu,
                   "worker_cpu_seconds": self.completed_children_cpu,
                   "same_key_score_mismatches": self.mismatches, "stop_reasons": self.stop_reasons,
                   "failure": failure, "once_only_no_restart": True, "observer_never_provided_scores": True,
                   "all_spawned_workers_reaped": all(w.process is None or w.wait4 is not None for w in self.workers),
                   "GPU_seconds": 0}
        write_json_once(self.root / "runner_receipt.json", receipt)
        self.emit("RUNNER_FINISHED", formal_CPU_seconds=receipt["formal_CPU_seconds"],
                  once_only_no_restart=True, all_spawned_workers_reaped=receipt["all_spawned_workers_reaped"])
        self.log.close()
        return receipt


def main() -> None:
    if sys.platform != "linux":
        raise SystemExit("Server-only orchestration; local Windows draft is never an experiment invocation")
    if len(sys.argv) != 2:
        raise SystemExit("Usage: orchestration_draft.py ROOT")
    root = Path(sys.argv[1]).resolve()
    observer = Observer(root)
    # Preserve graceful collection when the caller interrupts the supervising process.
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda received, frame: observer.stop_all("OBSERVER_SIGNAL_" + str(received)))
    result = observer.run()
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
