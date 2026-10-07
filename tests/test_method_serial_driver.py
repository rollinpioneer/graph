"""Subprocess fixtures only. No repository, model, GPU or SSH access."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from cp_disr.blocksworld.method_serial.schedule import run_manifest, check_new_root, order_pending

class DriverTests(unittest.TestCase):
    def make_fixture(self,tmp, statuses):
        storage=Path(tmp)/"xushijie3";storage.mkdir()
        repo=storage/"repo";repo.mkdir()
        run=storage/"run";run.mkdir()
        tasks=[]
        for i,status in enumerate(statuses):
            tid=f"stage{i}"
            receipt=run/f"{tid}.json"
            artifact=run/f"{tid}.txt"
            count=run/f"{tid}.count"
            code=("import pathlib,json,hashlib,os,sys;"
                  f"a=pathlib.Path({str(artifact)!r});a.write_text('fixture');"
                  f"c=pathlib.Path({str(count)!r});c.write_text(str(int(c.read_text())+1) if c.exists() else '1');"
                  f"r={{'status':{status!r},'input_fingerprint':os.environ['TASK_FINGERPRINT'],"
                  "'outputs':[{'path':str(a),'sha256':hashlib.sha256(a.read_bytes()).hexdigest()}]};"
                  f"pathlib.Path({str(receipt)!r}).write_text(json.dumps(r));"
                  f"sys.exit({0 if status=='DONE' else 1})")
            tasks.append({"id":tid,"argv":[sys.executable,"-S","-c",code],
                          "input_hashes":{"fixture":"v1"},"receipt":str(receipt),
                          "failure_scope":"local"})
        manifest=storage/"manifest.json"
        manifest.write_text(json.dumps({"execution_authorized":True,"storage_root":str(storage),
                                        "repo_root":str(repo),"run_root":str(run),
                                        "max_new_training_runs":9,"tasks":tasks}))
        return manifest,run,storage

    def test_01_completed_stages_are_not_rerun(self):
        with tempfile.TemporaryDirectory() as d:
            m,r,_=self.make_fixture(d,["DONE","DONE"])
            self.assertEqual(run_manifest(m),0)
            self.assertEqual(run_manifest(m),0)
            self.assertEqual((r/"stage0.count").read_text(),"1")
            self.assertEqual((r/"stage1.count").read_text(),"1")

    def test_02_local_failure_continues(self):
        with tempfile.TemporaryDirectory() as d:
            m,r,_=self.make_fixture(d,["TECHNICAL_INCOMPLETE","DONE"])
            self.assertEqual(run_manifest(m),1)
            receipt=json.loads((r/"driver_receipt.json").read_text())
            self.assertEqual(receipt["stages"]["stage1"]["status"],"DONE")
            self.assertEqual(receipt["state"],"C1_METHOD_SERIAL_SUITE_PARTIAL_TECHNICAL")

    def test_03_global_failure_stops(self):
        with tempfile.TemporaryDirectory() as d:
            m,r,_=self.make_fixture(d,["DATA_LEAKAGE","DONE"])
            self.assertEqual(run_manifest(m),2)
            self.assertFalse((r/"stage1.count").exists())

    def test_04_legacy_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            old=Path(d)/"xushijie2";old.mkdir()
            alias=Path(d)/"xushijie3";alias.symlink_to(old,target_is_directory=True)
            with self.assertRaises(ValueError):
                check_new_root(str(alias))

    def test_05_priority_reorders_both_groups(self):
        tasks=[{"id":"sg","group":"event"},
               {"id":"cal1","group":"calibration"},{"id":"cal2","group":"calibration"},
               {"id":"rec1","group":"recurrent"},{"id":"rec2","group":"recurrent"},
               {"id":"att","group":"attention"}]
        got=order_pending(tasks,"RECURRENT_THEN_CALIBRATION")
        self.assertEqual([t["id"] for t in got],
                         ["sg","rec1","rec2","cal1","cal2","att"])

    def test_06_no_implicit_authorization(self):
        with tempfile.TemporaryDirectory() as d:
            m,_,_=self.make_fixture(d,["DONE"])
            data=json.loads(m.read_text());data["execution_authorized"]=False
            m.write_text(json.dumps(data))
            with self.assertRaises(PermissionError):
                run_manifest(m)

    def test_07_bad_executable_continues_local(self):
        with tempfile.TemporaryDirectory() as d:
            m,r,_=self.make_fixture(d,["DONE","DONE"])
            data=json.loads(m.read_text());data["tasks"][0]["argv"]=["/no/such/fixture-executable"]
            m.write_text(json.dumps(data))
            self.assertEqual(run_manifest(m),1)
            self.assertEqual(json.loads((r/"driver_receipt.json").read_text())["stages"]["stage1"]["status"],"DONE")

if __name__=="__main__":
    unittest.main()

