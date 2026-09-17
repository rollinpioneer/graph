"""Atomic SQLite campaign ledger. No worker resets or retries."""
from __future__ import annotations
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from .constants_b import CAMPAIGN_ID, DISK_LIMIT_BYTES, FORMAL_JOB_LIMIT, FORMAL_STEP_LIMIT, SMOKE_JOB_LIMIT

SCHEMA = """
CREATE TABLE IF NOT EXISTS campaign (
  campaign_id TEXT PRIMARY KEY,
  release_sha256 TEXT NOT NULL,
  status TEXT NOT NULL,
  formal_steps_used INTEGER NOT NULL DEFAULT 0,
  formal_jobs_started INTEGER NOT NULL DEFAULT 0,
  smoke_jobs_started INTEGER NOT NULL DEFAULT 0,
  stop_new_claims INTEGER NOT NULL DEFAULT 0,
  disk_limit_bytes INTEGER NOT NULL,
  created_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
  job_id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  method TEXT,
  draw TEXT,
  policy_seed INTEGER,
  run_seed INTEGER,
  status TEXT NOT NULL,
  claim_index INTEGER,
  environment_steps INTEGER NOT NULL DEFAULT 0,
  gradient_updates INTEGER NOT NULL DEFAULT 0,
  pid INTEGER,
  directory TEXT,
  failure TEXT,
  release_sha256 TEXT
);
CREATE TABLE IF NOT EXISTS transitions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id TEXT NOT NULL,
  from_status TEXT,
  to_status TEXT NOT NULL,
  at_utc TEXT NOT NULL,
  note TEXT
);
"""

def utcnow():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class LedgerError(RuntimeError):
    pass


class CampaignLedger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)
        self.con.row_factory = sqlite3.Row
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("PRAGMA foreign_keys=ON")
        self.con.executescript(SCHEMA)

    def close(self):
        self.con.close()

    def _now(self):
        return utcnow()

    def init_campaign(self, release_sha256, jobs):
        self.con.execute("BEGIN IMMEDIATE")
        try:
            row = self.con.execute("select campaign_id from campaign where campaign_id=?", (CAMPAIGN_ID,)).fetchone()
            if row is None:
                self.con.execute(
                    "insert into campaign(campaign_id,release_sha256,status,disk_limit_bytes,created_at_utc) values (?,?,?,?,?)",
                    (CAMPAIGN_ID, release_sha256, "INITIALIZED", DISK_LIMIT_BYTES, self._now()),
                )
            else:
                existing = self.con.execute("select release_sha256 from campaign where campaign_id=?", (CAMPAIGN_ID,)).fetchone()
                if existing["release_sha256"] != release_sha256:
                    raise LedgerError("ledger bound to a different release")
            for job in jobs:
                cur = self.con.execute("select job_id from jobs where job_id=?", (job["job_id"],))
                if cur.fetchone() is None:
                    self.con.execute(
                        "insert into jobs(job_id,kind,method,draw,policy_seed,run_seed,status,environment_steps,gradient_updates,release_sha256) values (?,?,?,?,?,?,?,?,?,?)",
                        (job["job_id"], job["kind"], job.get("method"), job.get("draw"), job.get("policy_seed"),
                         job.get("run_seed"), "PENDING", 0, 0, release_sha256),
                    )
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise

    def campaign(self):
        row = self.con.execute("select * from campaign where campaign_id=?", (CAMPAIGN_ID,)).fetchone()
        return dict(row) if row else None

    def job(self, job_id):
        row = self.con.execute("select * from jobs where job_id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def jobs(self, kind=None, status=None):
        q = "select * from jobs"
        args = []
        clauses = []
        if kind:
            clauses.append("kind=?"); args.append(kind)
        if status:
            clauses.append("status=?"); args.append(status)
        if clauses:
            q += " where " + " and ".join(clauses)
        q += " order by claim_index, job_id"
        return [dict(r) for r in self.con.execute(q, args)]

    def counts(self):
        out = {}
        for r in self.con.execute("select kind,status,count(*) n, coalesce(sum(environment_steps),0) steps from jobs group by kind,status"):
            out[(r["kind"], r["status"])] = {"n": r["n"], "steps": r["steps"]}
        return out

    def claim(self, job_id, directory=None, pid=None):
        self.con.execute("BEGIN IMMEDIATE")
        try:
            camp = self.con.execute("select * from campaign where campaign_id=?", (CAMPAIGN_ID,)).fetchone()
            if camp is None:
                raise LedgerError("campaign missing")
            job = self.con.execute("select * from jobs where job_id=?", (job_id,)).fetchone()
            if job is None:
                raise LedgerError(f"unknown job {job_id}")
            if job["status"] != "PENDING":
                raise LedgerError(f"job {job_id} not PENDING")
            if camp["stop_new_claims"]:
                raise LedgerError("stop_new_claims")
            if job["kind"] == "formal" and camp["formal_jobs_started"] >= FORMAL_JOB_LIMIT:
                raise LedgerError("formal job budget")
            if job["kind"] == "smoke" and camp["smoke_jobs_started"] >= SMOKE_JOB_LIMIT:
                raise LedgerError("smoke job budget")
            nxt = self.con.execute("select coalesce(max(claim_index),0)+1 as n from jobs").fetchone()["n"]
            self.con.execute(
                "update jobs set status=?, claim_index=?, pid=?, directory=? where job_id=?",
                ("CLAIMED", nxt, pid, str(directory) if directory else None, job_id),
            )
            if job["kind"] == "formal":
                self.con.execute(
                    "update campaign set formal_jobs_started=formal_jobs_started+1 where campaign_id=?",
                    (CAMPAIGN_ID,),
                )
            else:
                self.con.execute(
                    "update campaign set smoke_jobs_started=smoke_jobs_started+1 where campaign_id=?",
                    (CAMPAIGN_ID,),
                )
            self.con.execute(
                "insert into transitions(job_id,from_status,to_status,at_utc,note) values (?,?,?,?,?)",
                (job_id, "PENDING", "CLAIMED", self._now(), "claim"),
            )
            self.con.execute("COMMIT")
            return nxt
        except Exception:
            self.con.execute("ROLLBACK")
            raise

    def _set_status(self, job_id, new_status, **fields):
        self.con.execute("BEGIN IMMEDIATE")
        try:
            job = self.con.execute("select * from jobs where job_id=?", (job_id,)).fetchone()
            if job is None:
                raise LedgerError(f"unknown job {job_id}")
            sets = ["status=?"]
            args = [new_status]
            for key in ("environment_steps", "gradient_updates", "pid", "directory", "failure"):
                if key in fields:
                    sets.append(f"{key}=?")
                    args.append(fields[key])
            args.append(job_id)
            self.con.execute(f"update jobs set {', '.join(sets)} where job_id=?", args)
            if new_status == "FAILED":
                self.con.execute("update campaign set stop_new_claims=1, status=? where campaign_id=?", ("FAILED_STOP", CAMPAIGN_ID))
            if "environment_steps" in fields and job["kind"] == "formal":
                delta = int(fields["environment_steps"]) - int(job["environment_steps"])
                if delta:
                    self.con.execute(
                        "update campaign set formal_steps_used=formal_steps_used+? where campaign_id=?",
                        (delta, CAMPAIGN_ID),
                    )
                    used = self.con.execute("select formal_steps_used from campaign where campaign_id=?", (CAMPAIGN_ID,)).fetchone()["formal_steps_used"]
                    if used > FORMAL_STEP_LIMIT:
                        raise LedgerError("formal step budget")
            self.con.execute(
                "insert into transitions(job_id,from_status,to_status,at_utc,note) values (?,?,?,?,?)",
                (job_id, job["status"], new_status, self._now(), fields.get("note")),
            )
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise

    def mark_running(self, job_id, **fields):
        self._set_status(job_id, "RUNNING", **fields)

    def mark_complete(self, job_id, **fields):
        self._set_status(job_id, "COMPLETE", **fields)

    def mark_failed(self, job_id, failure, **fields):
        fields = dict(fields)
        fields["failure"] = failure
        self._set_status(job_id, "FAILED", **fields)

    def mark_blocked(self, job_id, note=None):
        self._set_status(job_id, "BLOCKED", note=note)

    def stop_claims(self, status="STOP_NEW_CLAIMS"):
        self.con.execute("BEGIN IMMEDIATE")
        try:
            self.con.execute("update campaign set stop_new_claims=1, status=? where campaign_id=?", (status, CAMPAIGN_ID))
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise
