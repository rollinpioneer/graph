import json,shutil
from pathlib import Path
R=Path(__file__).resolve().parent;OLD=R.parent/'eg_n0_nvblox_20261024'
p=json.loads((OLD/'protocol.json').read_text())
p.update(primary='N_ref continuous projective SDF',comparator='official nvblox v0.0.10, independently calibrated',
 amendment='User explicitly promotes continuous NumPy projection to primary and authorizes completion. Prior 99% continuous-vs-voxel equality gate is not a gate for this new experiment; old HOLD remains unchanged.',
 calibration_candidates='All original final-pool candidates, 12 angles each; official collision and empty labels only. No force closure.',
 P2_candidates='All original final-pool candidates, 12 angles each; report clean/nonempty false filtering, not FC-feasible false filtering.',
 main_kappa='Choose kappa at OP1 training recall; freeze the same kappa for OP2. Calibrate threshold separately for each OP. nvblox separately follows same rule.',
 simple_rules=['R1: full-cloud EG global occupancy ratio','R2: max component point count under EG geometry','R3: point count in official static solids','R4: negative minimum distance from official solid surfaces to full observed cloud'],
 simple_choice='Per OP choose highest training scene-equal collision recall; ties lower false filter then R1/R2/R3/R4 order.',
 dense_audit='C3 all variants: original full-resolution object model points plus 2.5mm table; robust = dense point at least 1.25mm inside official static solid; original 8mm-collision without robust confirmation called boundary-sensitive/unconfirmed, not false.',
 metrics='Scene-equal. C3 clean-nonempty and FC-feasible FPR separately; training/P2 clean-nonempty FPR. Stratified scene bootstrap 10000 draws seed20261024.',
 postprocessing='All 1024 original raw candidates, changed angles only for existing 1229. NMS 3cm/pi6, object top10, top50. New top10 unknowns direct evaluated, lower ranks remain unknown.',
 gates='Original GO-N gates applied to N_ref; require actual C3 feasible FPR <= EG +1.5pp at OP1. No post-C3 retuning.',
 no_training=True,no_C4_C5=True)
(R/'protocol.json').write_text(json.dumps(p,indent=2),encoding='utf-8')
src=(OLD/'infer_frozen.py').read_text().replace("ROOT=B/'eg_n0_nvblox_20261024'","ROOT=B/'eg_n0_nref_main_20261024'")
(R/'infer_frozen.py').write_text(src,encoding='utf-8')
shutil.copy2(OLD/'proj_sdf_ref.py',R/'proj_sdf_ref.py')
