"""D7-A1..A8: the amended D7 check judges immutable identity and the state chain, never whole-receipt hash equality."""
from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

from cp_disr.analysis import s4_tp_sr_pilot_v2 as m
from cp_disr.analysis import s4_tp_sr_pilot_v2_diag as d

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs/final_master/s4_tp_sr_pilot_v2.yaml"
KEYS = ("wave_d_branch_attempts", "total_branch_attempts", "environment_resets")


def _j(p):
    return json.loads(Path(p).read_text())


def _w(p, o):
    Path(p).write_text(json.dumps(o))


def build_identity_fixture(tmp):
    """Four finished attempts with a full, consistent identity/state chain. The receipt is rewritten on completion, so its bytes differ from the
    reservation-time bytes that the restore receipt pinned."""
    out = tmp / "o"
    phys = out / "diagnostics"
    m.init_phys_dir(phys, 4)
    cfg = m.load_config(CFG)
    m.init_ledger(out, cfg)
    branches = [{"branch_id": f"b{i}", "attempt_id": f"b{i}", "case_id": f"c{i // 2}", "route": "direct" if i % 2 == 0 else "relocation", "candidate_id": "x", "repeat": 0,
                 "restore_seed": 1000 + i // 2} for i in range(4)]
    reg_path = phys / "diagnostic_branch_registration.json"
    _w(reg_path, {"branches": branches})
    reg_sha = m.sha256_file(reg_path)
    events, attempts = [], {}
    for i, b in enumerate(branches):
        bid, rid = b["branch_id"], f"rid{i}"
        m.charge_many(out, list(KEYS), ref=bid)
        rp = phys / "witnesses/branch_receipts" / f"{bid}.json"
        _w(rp, {"attempt_id": bid, "branch_id": bid, "cap": 4, "reservation_id": rid, "registration_sha256": reg_sha, "state": "STARTED"})
        pinned = m.sha256_file(rp)                                  # the restore receipt pins the pre-terminal bytes
        _w(rp, {"attempt_id": bid, "branch_id": bid, "cap": 4, "reservation_id": rid, "registration_sha256": reg_sha, "state": "COMPLETED",
                "execution_status": "TERMINATED", "termination_reason": "TASK_SUCCESS"})
        _w(phys / "branch_results" / f"{bid}.json", {"branch_id": bid, "execution_status": "TERMINATED", "env_counts": {"reset_calls": 1}})
        _w(phys / "witnesses/restore_receipts" / f"{bid}.json", {
            "branch_id": bid, "case_id": b["case_id"], "budget_receipt_sha256": pinned,
            "restore_receipt": {"requested_case_id": b["case_id"], "applied_case_id": b["case_id"], "requested_restore_seed": b["restore_seed"],
                                "applied_restore_seed": b["restore_seed"], "normalized_reset_config_sha256": f"cfg{i // 2}", "restore_checks": {"seed_applied": True}}})
        base = {"branch_id": bid, "event_type": "branch_attempt", "reservation_id": rid}
        events += [{**base, "status": "RESERVED"}, {**base, "status": "STARTED"}, {**base, "status": "COMPLETED", "execution_status": "TERMINATED", "termination_reason": "TASK_SUCCESS"}]
        attempts[bid] = "COMPLETED"
    (phys / "budget_events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    _w(phys / "attempt_registry.json", attempts)
    return out, cfg, {"branches": branches}


def _receipt(out, bid):
    return out / "diagnostics/witnesses/branch_receipts" / f"{bid}.json"


def _mutate(path, fn):
    o = _j(path)
    fn(o)
    _w(path, o)


def _events(out):
    p = out / "diagnostics/budget_events.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()]


def _set_events(out, evs):
    (out / "diagnostics/budget_events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs))


def _fails(out, needle):
    r = d.d7_identity_chain(out)
    assert r["pass"] is False and any(needle in p for p in r["problems"]), r["problems"]


def test_D7_A1_receipt_updated_to_completed_hash_differs_identity_consistent_passes(tmp_path):
    out, _, _ = build_identity_fixture(tmp_path)
    for i in range(4):
        rr = _j(out / "diagnostics/witnesses/restore_receipts" / f"b{i}.json")
        assert rr["budget_receipt_sha256"] != m.sha256_file(_receipt(out, f"b{i}"))      # whole-file hashes really differ
    r = d.d7_identity_chain(out)
    assert r["pass"] is True, r["problems"]
    assert r["per_branch"]["b0"]["pre_terminal_receipt_bytes"] == "PRE_TERMINAL_RECEIPT_BYTES_NOT_RETAINED"


def test_D7_A2_branch_id_mismatch_fails(tmp_path):
    out, _, _ = build_identity_fixture(tmp_path)
    _mutate(_receipt(out, "b1"), lambda o: o.update(branch_id="other"))
    _fails(out, "RECEIPT_BRANCH_ID_MISMATCH")


@pytest.mark.parametrize("where", ["receipt_attempt", "receipt_reservation", "event_reservation"])
def test_D7_A3_reservation_or_attempt_id_mismatch_fails(tmp_path, where):
    out, _, _ = build_identity_fixture(tmp_path)
    if where == "receipt_attempt":
        _mutate(_receipt(out, "b2"), lambda o: o.update(attempt_id="zzz"))
        _fails(out, "ATTEMPT_ID_MISMATCH")
    elif where == "receipt_reservation":
        _mutate(_receipt(out, "b2"), lambda o: o.update(reservation_id="zzz"))
        _fails(out, "RESERVATION_ID_MISMATCH")
    else:
        evs = _events(out)
        evs[4]["reservation_id"] = "zzz"
        _set_events(out, evs)
        _fails(out, "RESERVATION_ID_MISMATCH")


@pytest.mark.parametrize("how", ["receipt_field", "registration_file_edited"])
def test_D7_A4_registration_sha256_mismatch_fails(tmp_path, how):
    out, _, _ = build_identity_fixture(tmp_path)
    if how == "receipt_field":
        _mutate(_receipt(out, "b0"), lambda o: o.update(registration_sha256="0" * 64))
    else:
        _mutate(out / "diagnostics/diagnostic_branch_registration.json", lambda o: o["branches"][0].update(route="tampered"))
    _fails(out, "REGISTRATION_SHA256_MISMATCH")


@pytest.mark.parametrize("missing", ["RESERVED", "STARTED", "COMPLETED"])
def test_D7_A5_missing_reserved_started_or_terminal_event_fails(tmp_path, missing):
    out, _, _ = build_identity_fixture(tmp_path)
    evs = [e for e in _events(out) if not (e["branch_id"] == "b3" and e["status"] == missing)]
    _set_events(out, evs)
    _fails(out, "EVENT_SEQUENCE_NOT_RESERVED_STARTED_TERMINAL")


@pytest.mark.parametrize("what", ["second_reset", "second_attempt"])
def test_D7_A6_two_resets_or_two_attempts_for_one_branch_fails(tmp_path, what):
    out, _, _ = build_identity_fixture(tmp_path)
    if what == "second_reset":
        with (out / "budget_events.jsonl").open("a") as f:
            f.write(json.dumps({"event": "charge", "key": "environment_resets", "used": 5, "cap": 20, "ref": "b0"}) + "\n")
        _fails(out, "CHARGE_COUNT_NOT_1:environment_resets")
    else:
        evs = _events(out)
        evs.insert(1, dict(evs[0]))                                         # a second RESERVED for b0
        _set_events(out, evs)
        r = d.d7_identity_chain(out)
        assert r["pass"] is False and any("EVENT_SEQUENCE" in p or "ATTEMPT_CONSUMED" in p for p in r["problems"])


def test_D7_A7_registry_and_terminal_receipt_state_disagree_fails(tmp_path):
    out, _, _ = build_identity_fixture(tmp_path)
    _mutate(out / "diagnostics/attempt_registry.json", lambda o: o.update(b2="FAILED"))
    _fails(out, "TERMINAL_STATE_DISAGREEMENT")


def test_D7_A8_legacy_whole_receipt_hash_logic_is_rejected(tmp_path):
    out, _, _ = build_identity_fixture(tmp_path)

    def legacy(o):                                                          # the retired check: compare pinned hash with the current whole file
        phys = Path(o) / "diagnostics"
        return all(_j(phys / "witnesses/restore_receipts" / f"b{i}.json")["budget_receipt_sha256"] == m.sha256_file(_receipt(Path(o), f"b{i}")) for i in range(4))

    assert legacy(out) is False and d.d7_identity_chain(out)["pass"] is True     # legacy logic would wrongly fail a healthy run
    for fn in (d.d7_identity_chain, d.wave_d_gate):                            # and neither gate function may compare receipt hashes again
        tree = ast.parse(inspect.getsource(fn))
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                assert "budget_receipt_sha256" not in ast.dump(node) and "sha256_file" not in ast.dump(node)
    assert "d7_identity_chain(" in inspect.getsource(d.wave_d_gate)


def test_original_gate_file_is_never_overwritten_by_plain_classification(tmp_path):
    out = tmp_path / "o"
    (out / "diagnostics").mkdir(parents=True)
    (out / "diagnostics/wave_d_gate.json").write_text('{"wave_d_status": "FAIL"}')
    with pytest.raises(m.StopRun):
        d.classify_diagnostics(ROOT, CFG, out)
    assert (out / "diagnostics/wave_d_gate.json").read_text() == '{"wave_d_status": "FAIL"}'
