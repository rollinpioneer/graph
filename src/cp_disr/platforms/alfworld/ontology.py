"""Fixed semantic class vocabulary (ALFRED ontology + UNK). Never built from results or from test games."""
import json
import os

from .observation import type_of

_ONT = json.load(open(os.path.join(os.path.dirname(__file__), "ontology.json")))
OTYPES = tuple(_ONT["otypes"])
RTYPES = tuple(_ONT["rtypes"])
UNK = "UNK"
OTYPE_INDEX = {c: i for i, c in enumerate(OTYPES + (UNK,))}
RTYPE_INDEX = {c: i for i, c in enumerate(RTYPES + (UNK,))}
# node-type vocabulary handed to the graph encoder (argument types of nodes)
NODE_TYPES = tuple("rtype:" + c for c in RTYPES + (UNK,)) + tuple("otype:" + c for c in OTYPES + (UNK,))


def rclass(instance_or_class):
    c = type_of(instance_or_class) if instance_or_class[-1:].isdigit() else instance_or_class
    return "rtype:" + (c if c in RTYPE_INDEX else UNK)


def oclass(c):
    return "otype:" + (c if c in OTYPE_INDEX else UNK)


def rclass_onehot(instance):
    v = [0.0] * len(RTYPE_INDEX)
    c = type_of(instance)
    v[RTYPE_INDEX.get(c, RTYPE_INDEX[UNK])] = 1.0
    return v


def oclass_onehot(otype):
    v = [0.0] * len(OTYPE_INDEX)
    v[OTYPE_INDEX.get(otype, OTYPE_INDEX[UNK])] = 1.0
    return v
