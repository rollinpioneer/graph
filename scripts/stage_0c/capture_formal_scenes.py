"""Real read-only MuJoCo scene capture; never executes a skill or calls VLM."""
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np
from PIL import Image

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--task-id',required=True);p.add_argument('--scene-id',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(); root=a.root.resolve(); out=a.output.resolve()
 if a.task_id not in {'D0','T_A','T_C'}: raise SystemExit('BLOCKED: task not in formal matrix')
 if out.exists(): raise SystemExit('BLOCKED: refusing overwrite of formal scene')
 task=root/'configs/tasks/resolved'/f'{a.task_id}.yaml'
 if not task.is_file(): raise SystemExit('BLOCKED: missing resolved task')
 import yaml
 td=yaml.safe_load(task.read_text())
 if td.get('stage_0c_capture_ready') is not True: raise SystemExit('BLOCKED: resolved task is not capture-ready')
 cfg=json.loads(a.config.read_text())
 if cfg.get('task_id')!=a.task_id or cfg.get('scene_id')!=a.scene_id or int(cfg.get('seed'))!=a.seed: raise SystemExit('BLOCKED: config identity mismatch')
 try:
  from cp_disr.platforms.libero.scene_capture import SceneCaptureAdapter
 except Exception as e: raise SystemExit('BLOCKED: platform capture environment unavailable: '+type(e).__name__)
 out.mkdir(parents=True)
 adapter=SceneCaptureAdapter(root/'assets/cp_disr/scene_template.xml',128,128); adapter.reset_scene(a.task_id,cfg,a.seed); rgb=adapter.capture_rgb(); depth=adapter.capture_depth(); adapter.close()
 Image.fromarray(rgb).save(out/'rgb.png'); np.save(out/'depth.npy',depth.astype('float32'))
 manifest={'task_id':a.task_id,'scene_id':a.scene_id,'seed':a.seed,'platform_type':'simulator','platform_ref':'LIBERO_CP_DISR_CLEAN','rgb_sha256':sha(out/'rgb.png'),'depth_sha256':sha(out/'depth.npy'),'actions_executed':0,'hidden_truth_used_for_prompt':False}
 (out/'capture_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n'); print(json.dumps(manifest))
if __name__=='__main__':main()
