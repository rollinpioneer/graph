"""Read archived models through their matching archived source, without renaming them."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


def load_strict_legacy_graph_model(source: Path, checkpoint: Path, device: str = "cpu") -> Any:
    """Load a checkpoint only if the archived model definition matches exactly."""
    import torch
    spec = importlib.util.spec_from_file_location("upgrade_v2_archived_graph_model", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load archived model source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.GraphStateModel()
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload.get("model", payload), strict=True)
    return model.to(device).eval()
