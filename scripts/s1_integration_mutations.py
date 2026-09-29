#!/usr/bin/env python3
"""Run the eight S1-REV1 integration mutations in isolated source copies."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

TESTS = [
    "tests/test_s1_dispatch_integration.py",
    "tests/test_s1_restore_receipt_integration.py",
    "tests/test_s1_source_binding_integration.py",
    "tests/test_s1_finalize_loader_integration.py",
]
MUTATIONS = [
    ("L-M01", "src/cp_disr/analysis/s1_revision_resume.py",
     "if not receipt_path.is_file() and ledger_path.is_file():", "if ledger_path.is_file():",
     "reject a reserved last slot before claim"),
    ("L-M02", "src/cp_disr/analysis/s1_revision_resume.py",
     "receipt = claim_branch_attempt(output_dir, branch_id)", "receipt = _json(receipt_path)",
     "skip the one-time receipt claim"),
    ("L-M03", "src/cp_disr/analysis/s1_integration.py",
     "if branch.get(\"authorized\") is not True or branch.get(\"execute_now\") is not True:",
     "if False:", "allow an unauthorized branch reservation"),
    ("L-M04", "src/cp_disr/analysis/s1_integration.py",
     "if not isinstance(receipt, dict):", "if False:",
     "default missing restore receipt to acceptance"),
    ("L-M05", "src/cp_disr/analysis/s1_integration.py",
     "if int(receipt[\"applied_restore_seed\"]) != int(branch.get(\"restore_seed\")):",
     "if False:", "ignore the applied restore seed"),
    ("L-M06", "src/cp_disr/runtime.py",
     "if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=spec['sha256']:",
     "if False:", "bypass the RuntimeFactory source hash"),
    ("L-M07", "src/cp_disr/analysis/s1_revision_resume.py",
     "bundle = load_gate_evidence_bundle(root, manifest_path)", "bundle = {}",
     "finalize without loading the six evidence tables"),
    ("L-M08", "src/cp_disr/analysis/s1_revision_resume.py",
     "elif all_pass and bundle.get(\"scientific_admissible\") is True:", "elif False:",
     "force the eligibility decision to NOT_ELIGIBLE"),
]

def sha(path):
    h=hashlib.sha256(); h.update(Path(path).read_bytes()); return h.hexdigest()

def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2)+"\n", encoding="utf-8")

def run_one(root, out, item):
    ident, rel, old, new, rationale = item
    work=Path(tempfile.mkdtemp(prefix=f"s1_integration_{ident}_", dir="/tmp"))
    result={"id":ident,"path":rel,"rationale":rationale,"isolated_copy":str(work)}
    try:
        shutil.copytree(root/"src", work/"src")
        (work/"tests").mkdir(); (work/"tests/__init__.py").write_text(""); (work/"scripts").mkdir(); (work/"scripts/__init__.py").write_text(""); shutil.copy2(root/"scripts/s1_integration_closeout.py", work/"scripts/s1_integration_closeout.py")
        shutil.copytree(root/"tests/helpers", work/"tests/helpers")
        for test in TESTS:
            target=work/test; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(root/test,target)
        target=work/rel; original=target.read_text(encoding="utf-8")
        result["original_sha256"]=sha(target); result["occurrences"]=original.count(old)
        if result["occurrences"] != 1:
            result.update(status="NOT_APPLIED", reason="expected one mutation site")
            write(out/f"mutations/{ident}/result.json", result); return result
        mutated=original.replace(old,new,1); target.write_text(mutated,encoding="utf-8")
        result["mutated_sha256"]=sha(target)
        env=os.environ.copy(); env["PYTHONPATH"]=str(work/"src")+os.pathsep+str(work)
        env.update({"CUDA_VISIBLE_DEVICES":"","CP_DISR_AUDIT_ONLY":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"})
        proc=subprocess.run([sys.executable,"-m","pytest","-q","-p","no:cacheprovider",*TESTS],cwd=work,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
        log=out/f"mutations/{ident}/pytest.log"; log.parent.mkdir(parents=True,exist_ok=True); log.write_text(proc.stdout,encoding="utf-8")
        invalid=any(token in proc.stdout for token in ("ERROR collecting","SyntaxError","ImportError","ModuleNotFoundError"))
        result.update(returncode=proc.returncode,log=str(log.relative_to(out)),status=("INVALID_MUTATION" if invalid else ("KILLED" if proc.returncode else "SURVIVED")))
        write(out/f"mutations/{ident}/result.json",result); return result
    except subprocess.TimeoutExpired as exc:
        result.update(status="INVALID_MUTATION",returncode="TIMEOUT",error=str(exc))
        write(out/f"mutations/{ident}/result.json",result); return result
    finally:
        shutil.rmtree(work,ignore_errors=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",required=True); ap.add_argument("--output",required=True)
    a=ap.parse_args(); root=Path(a.root).resolve(); out=Path(a.output).resolve()
    results=[run_one(root,out,item) for item in MUTATIONS]
    summary={"status":"PASS" if all(r["status"]=="KILLED" for r in results) else "NOT_READY","mutation_count":len(results),"killed":sum(r["status"]=="KILLED" for r in results),"survived":sum(r["status"]=="SURVIVED" for r in results),"invalid":sum(r["status"]=="INVALID_MUTATION" for r in results),"not_applied":sum(r["status"]=="NOT_APPLIED" for r in results),"results":results,"real_provider_calls":0,"real_environment_constructions":0,"real_resets":0,"real_captures":0,"real_skill_executions":0,"real_physical_episodes":0,"rl_transitions":0,"optimizer_steps":0}
    write(out/"mutations/summary.json",summary); print(json.dumps(summary,ensure_ascii=False))
if __name__=="__main__": main()
