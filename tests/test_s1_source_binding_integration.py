import json, shutil
from argparse import Namespace
from pathlib import Path
import pytest
from cp_disr.runtime import BindingError, load_runtime, require_runtime
from scripts.s1_integration_closeout import prepare_binding

def make_binding(tmp_path):
    root=Path("/home/xushijie2/graph_cp_disr_final_s1_rev1")
    source=Path("runs/final_master/S1/20260928T154311Z_bea4bf0d")
    out=tmp_path/"out"; out.mkdir()
    prepare_binding(Namespace(root=str(root),source=str(source),output=str(out)))
    return root,out

def test_prepare_binding_and_real_loader_source_gate(tmp_path, monkeypatch):
    root,out=make_binding(tmp_path)
    manifest=json.loads((out/"derived/runtime_manifest.current.json").read_text())
    spec=require_runtime(manifest)
    assert spec["source_path"].endswith("src/cp_disr/platforms/libero/runtime_factory.py")
    calls=[]
    class Bundle:
        environment=executor=observations=perception=verifier=evaluator=safety=clock=snapshot_builder=object()
    import cp_disr.platforms.libero.runtime_factory as rf
    monkeypatch.setattr(rf,"create_stage_2a_runtime",lambda m: calls.append(True) or Bundle())
    loaded=load_runtime(manifest)
    assert calls==[True] and loaded.environment is not None
    assert json.loads((out/"inventory/source_binding.json").read_text())["real_runtime_instantiated"] is False

def test_hash_mismatch_rejects_before_factory(tmp_path, monkeypatch):
    root,out=make_binding(tmp_path)
    manifest=json.loads((out/"derived/runtime_manifest.current.json").read_text())
    manifest["runtime_factory"]["sha256"]="0"*64
    calls=[]
    import cp_disr.platforms.libero.runtime_factory as rf
    monkeypatch.setattr(rf,"create_stage_2a_runtime",lambda m: calls.append(True))
    with pytest.raises(BindingError):
        load_runtime(manifest)
    assert calls==[]

def test_module_realpath_mismatch_rejects_before_factory(tmp_path, monkeypatch):
    root,out=make_binding(tmp_path)
    manifest=json.loads((out/"derived/runtime_manifest.current.json").read_text())
    copied=tmp_path/"runtime_factory_copy.py"
    shutil.copy2(manifest["runtime_factory"]["source_path"],copied)
    manifest["runtime_factory"]["source_path"]=str(copied)
    from scripts.s1_integration_closeout import sha
    manifest["runtime_factory"]["sha256"]=sha(copied)
    calls=[]
    import cp_disr.platforms.libero.runtime_factory as rf
    monkeypatch.setattr(rf,"create_stage_2a_runtime",lambda m: calls.append(True))
    with pytest.raises(BindingError):
        load_runtime(manifest)
    assert calls==[]
