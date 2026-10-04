"""Freeze the ALFRED object / receptacle class ontology from the installed alfworld package.

Source: alfworld.gen.constants (fixed ALFRED vocabulary, independent of any split or result).
Output: src/cp_disr/platforms/alfworld/ontology.json (sorted lowercase class names, UNK is added at use time).
"""
import json
import os

import alfworld.gen.constants as C

otypes = sorted({k for k in C.OBJECTS_LOWER_TO_UPPER})
rtypes = sorted({x.lower() for x in (set(C.RECEPTACLES) | set(C.STATIC_RECEPTACLES) | set(C.MOVABLE_RECEPTACLES) | set(C.OPENABLE_CLASS_SET))})
out = os.path.join(os.path.dirname(__file__), "..", "src", "cp_disr", "platforms", "alfworld", "ontology.json")
json.dump({"source": "alfworld.gen.constants", "alfworld_version": "0.4.2", "otypes": otypes, "rtypes": rtypes}, open(out, "w"), indent=1)
print(len(otypes), "object classes,", len(rtypes), "receptacle classes ->", os.path.abspath(out))
