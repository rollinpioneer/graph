"""V3 fixed-frame online/cache equivalence; no new training or labels."""
import os, sys, json, time, platform
from pathlib import Path
import numpy as np
from scoring import FrozenScorer, cache_inputs, sha256
from infer import EconomicExtractor

G = Path(os.environ['GRASP_ROOT'])
R = Path(os.environ['LQR_ROOT'])
cfg = json.loads((R/'frozen_config.json').read_text())
N = G/'eg_nqs_pilot_v2_20261018'
V = G/'eg_local_delta_global_score_v2_20261019'
old = G/'eg_sub25_v1_20261015'
scorers = {h['seed']: FrozenScorer(G/f'eg_pose_select_pilot_v1_20261018/heads/{h["seed"]}/step1000.pt', h['sha256']) for h in cfg['heads']}
start = time.time()
extractor = EconomicExtractor(old/'runtime/EconomicGrasp', old/'data/graspnet', old/'runs/EG_SUB25_V1/train/economicgrasp_epoch10.tar', cfg['backbone_sha256'], 'cuda:0')
rows = []
for scene, frame in cfg['p0_frames']:
    source = N/f'exports/subset/scene_{scene:04d}/{frame:04d}.npz'
    pred, feature, action, identity = cache_inputs(source)
    sample_path = N/f'inputs/scene_{scene:04d}/{frame:04d}.npz'
    sample = extractor.sample(scene, frame, sample_path)
    actual, shared, original_action, seconds = extractor.extract(sample, verify_hook=True)
    checks = dict(prediction_exact=np.array_equal(actual,pred), feature_exact=np.array_equal(shared,feature), original_action_exact=np.array_equal(original_action,action))
    assert all(checks.values()), dict(scene=scene,frame=frame,checks=checks,pred_max_abs=float(np.max(np.abs(actual-pred))),feature_max_abs=float(np.max(np.abs(shared-feature))),action_max_abs=float(np.max(np.abs(original_action-action))))
    with np.load(V/f'evaluation/development/{scene:04d}/{frame:04d}.npz') as previous:
        for seed, scorer in scorers.items():
            cache_q = scorer.quality(feature,action)
            online_q = scorer.quality(shared,original_action)
            assert np.array_equal(cache_q, online_q)
            assert np.array_equal(cache_q,previous[f'Q_{seed}'][:,0])
            bundle = R/f'preflight/head_{seed}.pt'
            if not bundle.exists():
                scorer.save(bundle)
            restored = FrozenScorer(bundle)
            assert np.array_equal(cache_q,restored.quality(feature,action))
            for family in ['R0','S','F05']:
                updated,q,info=scorer.apply(pred,feature,action,family)
                assert np.array_equal(updated[:,1:],pred[:,1:])
            rows.append(dict(scene=scene,frame=frame,seed=seed,input_sha256=sha256(sample_path),export_sha256=sha256(source),prediction_exact=True,feature_exact=True,action_exact=True,old_Q_exact=True,online_cache_Q_exact=True,save_load_Q_exact=True,geometry_exact=True,EG_extract_seconds=seconds))
    print('P0_FRAME_PASS',scene,frame,flush=True)
import torch,MinkowskiEngine,open3d
result=dict(status='PASS_ONLINE_CACHE',frames=cfg['p0_frames'],heads=3,records=rows,wall_seconds=time.time()-start,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),gpu_peak_bytes=torch.cuda.max_memory_allocated(),versions=dict(torch=torch.__version__,CUDA=torch.version.cuda,MinkowskiEngine=MinkowskiEngine.__version__,open3d=open3d.__version__,libc=platform.libc_ver()),native_evaluation='PENDING_SEPARATE_CHECK',new_trainings=0)
(R/'results/preflight_online.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
