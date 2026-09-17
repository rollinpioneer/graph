"""Protocol constants and helpers. Does not generate data or train."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from . import METHODS, MOTIFS, SCHEMA
from .io_utils import load_json, hash_json

IDENTITY_FIELDS = ("family_id", "split", "version", "node_ids", "motif")

def load_protocol(path):
    p = load_json(path)
    if p.get("schema") != SCHEMA:
        raise ValueError("protocol schema")
    if p.get("default_mode") != "PREPARE_ONLY" or p.get("training_release") is not False:
        raise ValueError("protocol is prepare-only")
    if p.get("methods") != METHODS or p.get("motifs") != MOTIFS:
        raise ValueError("method/motif list")
    return p

def roots_for_split(protocol, split):
    d = protocol["data"]
    if split.startswith("train_"):
        return int(d["train_roots_per_motif_per_draw"])
    if split == "validation":
        return int(d["validation_roots_per_motif"])
    if split == "test":
        return int(d["test_roots_per_motif"])
    if split == "integration":
        return 2
    raise ValueError(split)

def family_id(protocol, split, motif, root):
    ns = protocol["data"]["namespaces"][split]
    fid = int(ns) + 100 * MOTIFS.index(motif) + int(root)
    prefix = str(ns)[:4]
    if prefix in protocol["data"]["forbidden_prior_prefixes"]:
        raise ValueError("forbidden namespace")
    return fid

def strip_identity(contract_dict):
    d = deepcopy(contract_dict)
    for k in IDENTITY_FIELDS:
        d.pop(k, None)
    return d
