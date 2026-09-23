"""Offline validations are torch/API independent; execution fails closed on bindings."""
import argparse, json, subprocess, sys
from pathlib import Path
from .common import BindingError, ContractError, canonical
from .runtime import require_runtime, require_cache_configuration

def read(path):
    path = Path(path)
    if path.suffix == ".json":
        return json.loads(path.read_text())
    import yaml
    return yaml.safe_load(path.read_text())

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path.cwd())
    sub = p.add_subparsers(dest="command", required=True)
    core = (
        "validate-manifests",
        "validate-contracts",
        "build-graph-fixture",
        "train",
        "evaluate",
        "generate-cache",
        "validate-runtime",
        "validate-controller",
        "validate-verifier",
        "validate-evaluator",
        "build-d0-split",
        "validate-d0-cache",
        "stage-1a-preflight",
        "stage-1a-run",
        "stage-1a-v11-run",
        "stage-2a-p0", "stage-2a-startup-gate", "stage-2a-run",
    )
    for name in core:
        sp = sub.add_parser(name)
        if name in ("validate-controller", "validate-verifier", "validate-evaluator", "stage-1a-preflight", "stage-1a-run", "stage-1a-v11-run", "stage-2a-p0", "stage-2a-startup-gate", "stage-2a-run"):
            sp.add_argument("--gpu", type=int, default=0)
        if name == "stage-1a-preflight":
            sp.add_argument("--all-p0", action="store_true")
            sp.add_argument("--resume-p0", action="store_true")

        if name == "stage-2a-run":
            sp.add_argument("--task", required=True, choices=["T_A", "T_C"])
            sp.add_argument("--method", required=True, choices=["B0", "B1", "B2", "Full"])
            sp.add_argument("--seed", type=int, required=True)
            sp.add_argument("--max-updates", type=int, default=64)
            sp.add_argument("--stop-after-updates", type=int, default=None)
            sp.add_argument("--resume", action="store_true")
        if name == "stage-1a-run":
            sp.add_argument("--only-startup-gates", action="store_true")
            sp.add_argument("--resume", action="store_true")
            sp.add_argument("--max-updates", type=int, default=16)
        if name == "stage-1a-v11-run":
            sp.add_argument("--only-startup-gates", action="store_true")
            sp.add_argument("--resume", action="store_true")
            sp.add_argument("--max-updates", type=int, default=16)
            sp.add_argument("--stop-after-updates", type=int, default=None)
            sp.add_argument("--method", choices=["B2", "Full"], default=None)
            sp.add_argument("--stamp", default=None)
            sp.add_argument("--configsha", default=None)
            sp.add_argument("--skip-startup-gates", action="store_true")
            sp.add_argument("--num-envs", type=int, default=1)
    cache = sub.add_parser("validate-relation-cache")
    cache.add_argument("--cache-directory", type=Path)
    t = sub.add_parser("run-unit-tests")
    t.add_argument("--scope", choices=["pure", "full"], required=True)
    a = p.parse_args(argv)
    root = a.root.resolve()
    try:
        if a.command == "validate-manifests":
            from .validation import validate_repository
            result = validate_repository(root)
            print(canonical(result))
            return 0 if result["passed"] else 2
        if a.command == "validate-contracts":
            from .contracts import from_dict, Registry
            doc = read(root / "configs/contracts/skills.yaml")
            reg = Registry(doc["predicate_types"])
            for c in doc["contracts"]:
                reg.register(from_dict(c))
            from jsonschema import Draft202012Validator
            validator = Draft202012Validator(read(root / "schemas/skill_contract.schema.json"))
            for c in doc["contracts"]:
                validator.validate(c)
            print(canonical({"status": "PASS_TEMPLATE_SCHEMA_ONLY", "count": len(doc["contracts"]), "runtime_bound": False}))
            return 0
        if a.command == "build-graph-fixture":
            from .fixtures import build_fixture
            f = build_fixture(root / "tests/fixtures")
            print(canonical({"synthetic_unit_fixture": True, "paper_performance_eligible": False, "template": f["template"]}))
            return 0
        if a.command == "validate-relation-cache":
            from .fixtures import build_fixture
            from .vlm import validate_relations, load_cache
            if a.cache_directory:
                manifest, edges = load_cache(a.cache_directory)
                print(canonical({"cache_identity_valid": True, "edges": len(edges), "semantic_binding_validated": False, "note": "Supply registered task template/ID/effect context to validate_relations for semantic validation"}))
                return 0
            f = build_fixture(root / "tests/fixtures")
            result = validate_relations(f["relations"], f["template"])
            print(canonical({"synthetic_unit_fixture": True, "paper_performance_eligible": False, "result": result}))
            return 0
        if a.command == "run-unit-tests":
            args = [sys.executable, "-m", "pytest", str(root / "tests"), "-m", "pure" if a.scope == "pure" else "pure or torch_runtime", "-ra"]
            return subprocess.call(args, cwd=root)
        if a.command in ("train", "evaluate"):
            require_runtime(read(root / "experiments/manifests/runtime_manifest.yaml"))
            config = root / "configs/run_resolved.json"
            if not config.exists():
                raise BindingError("MUST_BIND: configs/run_resolved.json (model dimensions, task cases, budget, trusted runtime factory)")
            from .execution import execute
            return execute(a.command, root, read(config))
        if a.command == "generate-cache":
            from .stage0c import run
            print(canonical(run(root)))
            return 0
        if a.command == "build-d0-split":
            import importlib.util
            spec = importlib.util.spec_from_file_location("build_d0_split", root / "scripts/stage_1a/build_d0_split.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            path = root / "configs/splits/D0_stage_1a.json"
            payload = mod.build_split(path)
            print(canonical({"status": "PASS", "path": str(path), "train": payload["train_count"], "dev": payload["dev_count"]}))
            return 0
        if a.command == "validate-d0-cache":
            from . import stage1a
            print(canonical(stage1a.cmd_validate_d0_cache(root)))
            return 0
        if a.command == "stage-1a-run":
            from . import stage1a_smoke
            print(canonical(stage1a_smoke.cmd_stage_1a_run(root, gpu=getattr(a, "gpu", 0), only_startup_gates=bool(getattr(a, "only_startup_gates", False)), resume=bool(getattr(a, "resume", False)), max_updates=int(getattr(a, "max_updates", 16)))))
            return 0
        if a.command == "stage-1a-v11-run":
            from . import stage1a_v11
            print(canonical(stage1a_v11.cmd_stage_1a_v11_run(root, gpu=getattr(a, "gpu", 0), only_startup_gates=bool(getattr(a, "only_startup_gates", False)), resume=bool(getattr(a, "resume", False)), max_updates=int(getattr(a, "max_updates", 16)), stop_after_updates=getattr(a, "stop_after_updates", None), method=getattr(a, "method", None), stamp=getattr(a, "stamp", None), configsha=getattr(a, "configsha", None), skip_startup_gates=bool(getattr(a, "skip_startup_gates", False)), num_envs=int(getattr(a, "num_envs", 1)))))
            return 0
        from . import stage1a
        gpu = getattr(a, "gpu", 0)
        if a.command == "validate-runtime":
            print(canonical(stage1a.cmd_validate_runtime(root)))
            return 0
        if a.command == "validate-controller":
            print(canonical(stage1a.cmd_validate_controller(root, gpu=gpu)))
            return 0
        if a.command == "validate-verifier":
            print(canonical(stage1a.cmd_validate_verifier(root, gpu=gpu)))
            return 0
        if a.command == "validate-evaluator":
            print(canonical(stage1a.cmd_validate_evaluator(root, gpu=gpu)))
            return 0

        if a.command == "stage-2a-startup-gate":
            from . import stage2a_startup_gate
            print(canonical(stage2a_startup_gate.cmd_stage_2a_startup_gate(root, gpu=gpu)))
            return 0
        if a.command == "stage-2a-run":
            from . import stage2a_explore
            print(canonical(stage2a_explore.cmd_stage_2a_run(root, a.task, a.method, a.seed, gpu=gpu, max_updates=getattr(a, "max_updates", 64), stop_after_updates=getattr(a, "stop_after_updates", None), resume=bool(getattr(a, "resume", False)))))
            return 0
        if a.command == "stage-2a-p0":
            from . import stage2a_p0
            print(canonical(stage2a_p0.cmd_stage_2a_p0(root, gpu=gpu)))
            return 0
        if a.command == "stage-1a-preflight":
            print(canonical(stage1a.cmd_stage_1a_preflight(root, gpu=gpu, resume=bool(getattr(a, "resume_p0", False)))))
            return 0
    except (BindingError, ContractError, ValueError, FileNotFoundError) as error:
        print(canonical({"status": "BLOCKED", "error": str(error)}))
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
