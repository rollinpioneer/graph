"""CPU execution adapter; runs the pinned generator with the verified KNN only."""
import sys,runpy,time,json,hashlib,os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import cpu_knn_probe as cpu
source=cpu.S/'dataset/generate_graspness.py';sys.argv[0]=str(source);started=time.time();status='FAIL';cfg=None
try:
 context=runpy.run_path(str(source),run_name='__main__');cfg=context['cfgs'];status='PASS'
finally:
 key='unknown' if cfg is None else f'{cfg.scene_start}_{cfg.scene_end}_{cfg.scene_stride}'
 out={'status':status,'backend':'CPU KD-tree shortlist + CUDA-equivalent float32 FMA/tie distance','stats':cpu.stats,'wall_seconds':time.time()-started,'generator_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'backend_sha256':hashlib.sha256(Path(cpu.__file__).read_bytes()).hexdigest(),'cuda_context_required':False}
 (cpu.R/'receipts'/f'cpu_graspness_backend_{key}.json').write_text(json.dumps(out,indent=2))
