"""Fail closed on missing Kinect data; this does not generate supervision.

The structure pass checks every required filename. --content additionally opens
every required image/meta file and validates all supervision arrays. Neither a
directory nor a partial sample can pass the complete-data gate.
"""
import argparse
import json
from pathlib import Path


def audit(root, content=False, supervision=True, scene_start=0, scene_end=190, asset_checks=True, required_object_ids=None):
    import numpy as np
    from PIL import Image
    from scipy.io import loadmat

    errors, scenes = [], []
    required_objects = set(required_object_ids or [])
    image_count = 0

    def problem(path, message):
        errors.append({"path": str(path), "error": str(message)})

    def exists(path):
        if not path.is_file() or path.stat().st_size == 0:
            problem(path, "missing or empty")
            return False
        return True

    for scene in range(scene_start, scene_end):
        scene_dir = root / "scenes" / f"scene_{scene:04d}"
        camera = scene_dir / "kinect"
        if not camera.is_dir():
            problem(camera, "missing Kinect scene")
            continue
        scenes.append(scene)
        object_file = scene_dir / "object_id_list.txt"
        if exists(object_file) and content:
            try:
                ids = [int(x) for x in object_file.read_text().split()]
                assert ids and all(0 <= i < 88 for i in ids)
                if scene < 100:
                    required_objects.update(ids)
            except Exception as exc:
                problem(object_file, exc)
        for name, shape in [("camK.npy", (3, 3)),
                            ("camera_poses.npy", (256, 4, 4)),
                            ("cam0_wrt_table.npy", (4, 4))]:
            path = camera / name
            if exists(path) and content:
                try:
                    arr = np.load(path, allow_pickle=False)
                    assert arr.shape == shape and np.isfinite(arr).all()
                except Exception as exc:
                    problem(path, exc)
        for frame in range(256):
            frame_name = f"{frame:04d}"
            for kind in ("rgb", "depth", "label"):
                path = camera / kind / (frame_name + ".png")
                if exists(path):
                    image_count += 1
                    if content:
                        try:
                            with Image.open(path) as image:
                                image.load()
                                assert image.size == (1280, 720)
                                pixels = np.asarray(image)
                                assert np.isfinite(pixels).all()
                                if kind == "depth":
                                    assert pixels.ndim == 2 and pixels.max() > 0
                                elif kind == "label":
                                    assert pixels.ndim == 2 and 0 <= pixels.min() <= pixels.max() <= 88
                                else:
                                    assert pixels.shape == (720, 1280, 3)
                        except Exception as exc:
                            problem(path, exc)
            path = camera / "meta" / (frame_name + ".mat")
            if exists(path) and content:
                try:
                    meta = loadmat(path)
                    for key in ("intrinsic_matrix", "factor_depth", "poses", "cls_indexes"):
                        assert key in meta and np.isfinite(meta[key]).all(), key
                    assert meta["intrinsic_matrix"].shape == (3, 3)
                    assert np.asarray(meta["factor_depth"]).item() > 0
                    ids = meta["cls_indexes"].reshape(-1)
                    assert len(ids) and np.all((ids >= 1) & (ids <= 88))
                    assert meta["poses"].shape == (3, 4, len(ids))
                except Exception as exc:
                    problem(path, exc)
            if scene < 100:
                exists(camera / "annotations" / (frame_name + ".xml"))
                if supervision:
                    path = root / "graspness" / scene_dir.name / "kinect" / (frame_name + ".npy")
                    if exists(path) and content:
                        try:
                            arr = np.load(path, allow_pickle=False)
                            assert arr.ndim in (1, 2) and arr.size > 0
                            assert arr.ndim == 1 or arr.shape[1] == 1
                            assert np.isfinite(arr).all() and 0 <= arr.min() <= arr.max() <= 1
                        except Exception as exc:
                            problem(path, exc)
        if scene < 100:
            path = root / "collision_label" / scene_dir.name / "collision_labels.npz"
            if exists(path) and content:
                try:
                    with np.load(path, allow_pickle=False) as labels:
                        assert labels.files
                        for key in labels.files:
                            arr = labels[key]
                            assert arr.ndim == 4 and arr.shape[1:] == (300, 12, 4)
                            assert arr.size and np.isin(arr, [0, 1]).all()
                except Exception as exc:
                    problem(path, exc)
            if supervision:
                path = root / "economic_grasp_label_300views" / (scene_dir.name + "_labels.npz")
                if exists(path) and content:
                    try:
                        with np.load(path, allow_pickle=False) as labels:
                            required = {"points", "rotations", "depth", "scores", "widths", "pointid", "vgraspness", "topview"}
                            assert required.issubset(labels.files)
                            count = len(labels["points"])
                            assert count > 0 and labels["points"].shape == (count, 3)
                            for key in required:
                                arr = labels[key]
                                assert len(arr) == count and np.isfinite(arr).all(), key
                            for key in required - {"points", "pointid"}:
                                assert labels[key].shape == (count, 300), key
                            assert np.all(labels["rotations"] < 12) and np.all(labels["depth"] < 4)
                            assert np.all(labels["topview"] < 300)
                            assert np.all(labels["scores"] <= 11)
                            assert np.all(labels["widths"][labels["scores"] > 0] <= 100)
                            assert np.all((labels["vgraspness"] >= 0) & (labels["vgraspness"] <= 1))
                    except Exception as exc:
                        problem(path, exc)
    if asset_checks:
        for obj in range(88):
            exists(root / "models" / f"{obj:03d}" / "nontextured.ply")
        if not required_objects:
            problem(root / "grasp_label", "training object identities not established from complete data")
        for obj in sorted(required_objects):
            path = root / "grasp_label" / f"{obj:03d}_labels.npz"
            if exists(path) and content:
                try:
                    with np.load(path, allow_pickle=False) as labels:
                        count = len(labels["points"])
                        assert labels["points"].shape == (count, 3) and count > 0
                        assert labels["scores"].shape == (count, 300, 12, 4)
                        assert labels["offsets"].shape == (count, 300, 12, 4, 3)
                        for key in ("points", "scores", "offsets"):
                            assert np.isfinite(labels[key]).all(), key
                except Exception as exc:
                    problem(path, exc)
    return {"root": str(root), "scene_count": len(scenes),
            "training_scene_count": sum(s < 100 for s in scenes),
            "test_scene_count": sum(s >= 100 for s in scenes),
            "image_files_present": image_count, "content_checked": content,
            "complete": content and not errors, "supervision_checked": supervision, "asset_checks": asset_checks, "scene_range": [scene_start, scene_end], "required_object_ids": sorted(required_objects), "errors": errors,
            "note": "This gate does not replace the real DataLoader/model/backward smoke test or prove per-point geometric correspondence."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--content", action="store_true")
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, args.content, supervision=not args.raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "errors"}))
    print("error_count", len(result["errors"]))
    raise SystemExit(0 if result["complete"] else 2)
