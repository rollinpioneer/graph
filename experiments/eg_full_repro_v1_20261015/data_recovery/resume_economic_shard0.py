from resource_jobs import run,R
import sys
S=R/'runtime/EconomicGrasp'
run('economic_full_shard_0_attempt2',[sys.executable,str(S/'dataset/generate_economic.py'),'--dataset_root',str(R/'data/graspnet'),'--camera_type','kinect','--scene_start','0','--scene_end','100','--scene_stride','2','--resume_verified'],cwd=S)
