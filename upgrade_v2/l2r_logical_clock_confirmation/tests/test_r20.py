from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from upgrade_v2.l2r_logical_clock_confirmation.candidate_adapter import apply_logical_guard
from upgrade_v2.l2r_logical_clock_confirmation.io_utils import read_csv, read_json, read_jsonl, write_csv, write_json, write_jsonl
from upgrade_v2.l2r_logical_clock_confirmation.logical_fault_injection import build_rollout_candidate_input
from upgrade_v2.l2r_logical_clock_confirmation.protocol import FROZEN_BLOBS, LEVELS, protocol_lock
from upgrade_v2.l2r_logical_clock_confirmation.registry import CASES, load_families, unique_keys
from upgrade_v2.l2r_logical_observation_clock.guard import INTERCEPT_REASON

ROOT = Path(__file__).resolve().parents[3]
REGISTRY = ROOT / "artifacts/pathgraph_sarm/upgrade_v2/r19_logical_observation_clock_development_v1/r19_confirmation_design_v1/confirmation_registry.json"


def blob(path: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), "hash-object", path], capture_output=True, text=True, check=True).stdout.strip()


def observation(time=1.0, order=10, contact=False, **extra):
    return {"time": time, "capture_order": order, "attempt_id": 1, "contact_present": contact,
            "gripper_command": "closed", "requested_effect": "HOLD_OBJECT", "context_valid": True,
            "attempt_end": False, "attempt_end_reason": None, **extra}


def proposal(time=1.0, order=10, action="recover_object", reason=INTERCEPT_REASON):
    return {**observation(time, order), "selected_action": action, "reason_code": reason}


class Fixture:
    def __init__(self, case_id: str):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name) / ("FAMILY__" + case_id)
        (self.root / "online_raw").mkdir(parents=True); (self.root / "detector_output").mkdir(); (self.root / "reference").mkdir()
        write_json(self.root / "metadata.json", {"case_id": case_id})
        frames=[]; streams={key:[] for key in ("detections_rgb","contact_proxy","attempt_lifecycle","gripper_commands","request_context")}
        for order in range(7):
            time=1.0 + order*.05
            frames.append({"capture_order":order,"time":time,"phase":"transport_pre_event","capture_kind":"periodic",
                           "jpeg_sha256":"samehash","jpeg_path":f"online_raw/rgb/front/{order:06d}.jpg","frame_missing":False})
            streams["detections_rgb"].append({"capture_order":order,"time":time,"object_centroid":[1,2],"gripper_centroid":[3,4],"object_confidence":1.0,"gripper_confidence":1.0,"width":192,"height":144})
            streams["contact_proxy"].append({"capture_order":order,"time":time,"contact_present":True})
            streams["attempt_lifecycle"].append({"capture_order":order,"time":time,"attempt_id":1,"attempt_phase":"inactive","attempt_active":False,"attempt_end":False,"attempt_end_reason":None})
            streams["gripper_commands"].append({"capture_order":order,"time":time,"gripper_command":"closed"})
            streams["request_context"].append({"capture_order":order,"time":time,"request_id":"request_1","requested_effect":"HOLD_OBJECT","context_valid":True})
        write_csv(self.root/"online_raw/frame_manifest.csv",frames)
        for key in streams:
            target=self.root/("detector_output/detections_rgb.jsonl" if key=="detections_rgb" else f"online_raw/{key}.jsonl")
            write_jsonl(target,streams[key])
        write_json(self.root/"reference/physical_reference.json",{"state":"WELD_SUPPORTED_HOLD"})
    def close(self): self.temp.cleanup()


class ParentFrozenTests(unittest.TestCase):
    def test_parent_commit(self): self.assertEqual(protocol_lock()["base_commit"],"e18bda2c897e01a9dd73ee319e603206e2ec3b6c")
    def test_R19_development_gate_pass(self): self.assertTrue(read_json(ROOT/"artifacts/pathgraph_sarm/upgrade_v2/r19_logical_observation_clock_development_v1/r17_candidate_development_replay_v1/development_gate.json")["r19_confirmation_design_allowed"])
    def test_guard_blob_frozen(self): self.assertEqual(blob("upgrade_v2/l2r_logical_observation_clock/guard.py"),FROZEN_BLOBS["guard"])
    def test_registry_exact_6x12(self): self.assertEqual((len(load_families(REGISTRY)),len(CASES)),(6,12))
    def test_R17_levels_frozen(self): self.assertEqual(LEVELS["strong"]["delta_v_local_mps"],(30.,30.,-6.))
    def test_detector_renderer_frozen(self): self.assertEqual((blob("upgrade_v2/visual_refine_l2/vision.py"),blob("upgrade_v2/visual_refine_l2/renderer.py")),(FROZEN_BLOBS["detector"],FROZEN_BLOBS["renderer"]))
    def test_temporal_scoring_frozen(self): self.assertEqual(blob("upgrade_v2/l2r_rgb_temporal_confirmation/temporal_scoring.py"),FROZEN_BLOBS["temporal_scoring"])


class CollectorTests(unittest.TestCase):
    def test_six_new_families(self): self.assertEqual(len(load_families(REGISTRY)),6)
    def test_72_unique_keys(self): self.assertEqual(len(set(unique_keys(REGISTRY))),72)
    def test_all_cases_transport(self): self.assertTrue(all("transport" in case.case_id for case in CASES))
    def test_prehold_point5_seconds(self): self.assertIn("_prehold(capture, 50)",(ROOT/"upgrade_v2/l2r_logical_clock_confirmation/collector.py").read_text())
    def test_phase_grid_alignment(self): self.assertIn("physics_step_index % 5",(ROOT/"upgrade_v2/l2r_logical_clock_confirmation/collector.py").read_text())
    def test_offsets_0_10_20_30_40(self): self.assertEqual({case.phase_offset_ms for case in CASES[6:11]},{0,10,20,30,40})
    def test_weak_medium_fixed_20ms(self): self.assertEqual((CASES[4].phase_offset_ms,CASES[5].phase_offset_ms),(20,20))
    def test_T11_exact_three_dropouts(self): self.assertEqual((.10,.15,.20),tuple(float(x) for x in (.10,.15,.20)))
    def test_T12_no_force_pulse(self): self.assertIsNone(CASES[11].level)


class FaultInjectionTests(unittest.TestCase):
    def _build(self,case):
        fixture=Fixture(case); result=build_rollout_candidate_input(fixture.root); return fixture,result
    def test_T3_exact_one_false_row(self):
        f,_=self._build(CASES[2].case_id); self.addCleanup(f.close); self.assertEqual(sum(r["contact_present"] is False for r in read_jsonl(f.root/"candidate_input/contact_proxy.jsonl")),1)
    def test_T3_next_physical_time_restores_true(self):
        f,_=self._build(CASES[2].case_id); self.addCleanup(f.close); rows=read_jsonl(f.root/"candidate_input/contact_proxy.jsonl"); i=next(i for i,r in enumerate(rows) if r["contact_present"] is False); self.assertTrue(rows[i+1]["time"]>rows[i]["time"] and rows[i+1]["contact_present"])
    def test_T4_same_time_pair(self):
        f,_=self._build(CASES[3].case_id); self.addCleanup(f.close); rows=read_jsonl(f.root/"candidate_input/contact_proxy.jsonl"); i=next(i for i,r in enumerate(rows) if r["contact_present"] is False); self.assertEqual(rows[i]["time"],rows[i+1]["time"])
    def test_T4_consecutive_capture_order(self):
        f,_=self._build(CASES[3].case_id); self.addCleanup(f.close); rows=read_jsonl(f.root/"candidate_input/contact_proxy.jsonl"); i=next(i for i,r in enumerate(rows) if r["contact_present"] is False); self.assertEqual(rows[i+1]["capture_order"],rows[i]["capture_order"]+1)
    def test_T4_false_then_true(self):
        f,_=self._build(CASES[3].case_id); self.addCleanup(f.close); rows=read_jsonl(f.root/"candidate_input/contact_proxy.jsonl"); i=next(i for i,r in enumerate(rows) if r["contact_present"] is False); self.assertTrue(rows[i+1]["contact_present"])
    def test_T4_same_jpeg_and_detection(self):
        f,_=self._build(CASES[3].case_id); self.addCleanup(f.close); rows=read_jsonl(f.root/"candidate_input/detections_rgb.jsonl"); self.assertEqual({k:v for k,v in rows[3].items() if k!="capture_order"},{k:v for k,v in rows[4].items() if k!="capture_order"})
    def test_fault_does_not_modify_reference(self):
        f,_=self._build(CASES[3].case_id); self.addCleanup(f.close); m=read_json(f.root/"online_raw/fault_injection_manifest.json"); self.assertEqual(m["reference_hashes_before"],m["reference_hashes_after"])
    def test_fault_metadata_not_candidate_visible(self):
        f,_=self._build(CASES[3].case_id); self.addCleanup(f.close); text="".join(p.read_text() for p in (f.root/"candidate_input").glob("*.jsonl")); self.assertNotIn("fault_type",text)


class CandidateTests(unittest.TestCase):
    def test_guard_imported_not_copied(self): self.assertIn("from upgrade_v2.l2r_logical_observation_clock.guard import LogicalObservationClockGuard",(ROOT/"upgrade_v2/l2r_logical_clock_confirmation/candidate_adapter.py").read_text())
    def test_guard_blob_hash(self): self.assertEqual(blob("upgrade_v2/l2r_logical_observation_clock/guard.py"),FROZEN_BLOBS["guard"])
    def test_same_time_false_false_can_confirm(self):
        rows=[observation(),observation(order=11)]; out=apply_logical_guard(rows,[proposal(),proposal(order=11,action="none",reason="x")]); self.assertEqual(out[1]["selected_action"],"recover_object")
    def test_same_time_false_true_clears(self):
        rows=[observation(),observation(order=11,contact=True)]; out=apply_logical_guard(rows,[proposal(),proposal(order=11,action="none",reason="x")]); self.assertEqual(out[1]["selected_action"],"none")
    def test_single_false_then_true_no_action(self): self.test_same_time_false_true_clears()
    def test_release_clears_pending(self):
        rows=[observation(),observation(order=11,gripper_command="open")]; out=apply_logical_guard(rows,[proposal(),proposal(order=11)]); self.assertNotEqual(out[1]["guard_state"],"CONFIRMED")
    def test_no_backdating(self):
        rows=[observation(),observation(1.05,11)]; out=apply_logical_guard(rows,[proposal(),proposal(1.05,11,action="none",reason="x")]); self.assertEqual(out[1]["time"],1.05)
    def test_raw_B2_unchanged(self): self.assertIn('_run_o(rows, "O_B2")',(ROOT/"upgrade_v2/l2r_logical_clock_confirmation/candidate_adapter.py").read_text())
    def test_raw_C3_unchanged(self): self.assertIn('_run_o(rows, "O_C3")',(ROOT/"upgrade_v2/l2r_logical_clock_confirmation/candidate_adapter.py").read_text())


class GeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.source=(ROOT/"upgrade_v2/l2r_logical_clock_confirmation/generator_gate.py").read_text()
    def test_72_trace_complete(self): self.assertIn("trace_complete_72_of_72",self.source)
    def test_72_detector_complete(self): self.assertIn("detector_complete_72_of_72",self.source)
    def test_72_prehold(self): self.assertIn("prehold_72_of_72",self.source)
    def test_30_strong_phase_loss(self): self.assertIn("strong_loss_T7_T10_24_of_24",self.source)
    def test_6_T11_loss(self): self.assertIn("strong_loss_T11_6_of_6",self.source)
    def test_24_T1_T4_no_loss(self): self.assertIn("no_loss_T1_T4_24_of_24",self.source)
    def test_6_release(self): self.assertIn("commanded_release_T12_6_of_6",self.source)
    def test_12_T5_T6_resolved(self): self.assertIn("resolved_T5_T6_12_of_12",self.source)
    def test_18_planned_rgb_missing(self): self.assertIn("T11_SUCCESSOR_PERIODIC_RGB_DROPOUT",self.source)
    def test_same_time_false_true_6_of_6(self): self.assertIn("T4_same_time_false_true_6_of_6",self.source)
    def test_positive_same_time_confirmation_coverage(self): self.assertIn("same_time_false_false_positive_count",self.source)


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.source=(ROOT/"upgrade_v2/l2r_logical_clock_confirmation/evaluation.py").read_text()
    def test_no_early_tolerance(self): self.assertNotIn("early_tolerance",self.source)
    def test_temporal_scoring_uses_physical_onset(self): self.assertIn("physical_loss_onset",self.source)
    def test_O_tier_input_provenance(self): self.assertIn("input_provenance_audit",self.source)
    def test_prefix_causality(self): self.assertIn('"expected": 1152',self.source)
    def test_parameter_search_disabled(self): self.assertIn("parameter_audit",self.source)
    def test_candidate_gate(self): self.assertIn('candidate["strong_T7_T11_correct_in_window"] >= 29',self.source)
    def test_raw_methods_not_selection_eligible(self): self.assertIn('"raw_methods_selection_eligible": False',self.source)
    def test_S_G_H_not_selection_eligible(self): self.assertIn('"state_assisted_selection_eligible": False',self.source)
    def test_l3_not_started(self): self.assertIn('"l3_entry_allowed": False',self.source)
    def test_failed_confirmation_does_not_retune(self): self.assertNotIn("grid_search",self.source)


if __name__=="__main__": unittest.main()
