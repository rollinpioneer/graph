"""Clarify existing A physical-reference labels without repeating measurements."""
import csv
from pathlib import Path
p=Path(__file__).resolve().parent/'results/E3_A_VERIFIED_TARGETS.csv'
with open(p) as f:reader=csv.DictReader(f);rows=list(reader);fields=reader.fieldnames
aliases={'actor_near_reached':'source_A_actor_near_reached','teacher_near_reached':'source_A_teacher_near_reached'}
if any(k in fields for k in aliases):
    fields=[aliases.get(k,k) for k in fields]
    rows=[{aliases.get(k,k):v for k,v in row.items()} for row in rows]
    with open(p,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
print('A_PHYSICAL_REFERENCE_LABELS_EXPLICIT',flush=True)
