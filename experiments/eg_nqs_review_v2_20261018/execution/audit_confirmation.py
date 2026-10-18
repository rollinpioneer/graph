"""Before confirmation inference: known exposure and raw observation identity."""
import csv,re,sys
from common import *
import scipy.io as sio
out=R/'results';confirmation=sum(SPLIT['official_confirmation'].values(),[])
sources=[OLD/'runs/EG_SUB25_V1/evaluation/per_scene_results.csv',G/'eg_quality_validation_v1_20261018/config.json',G/'eg_full_repro_v1_20261015/runs/EG_FULL_V1/evaluation_results.json']
prior=set(int(r['scene']) for r in csv.DictReader(open(sources[0])))
q=json.loads(sources[1].read_text());prior.update(q['dev_scenes']+q['seen']+q['novel'])
oldfull=json.loads(sources[2].read_text());assert oldfull['official_test_frames_completed']==0
# Existing derivative-result job lists and directories provide a cross-check.
jobs=[]
for p in (OLD/'d2d4_20261017').glob('jobs*.txt'):
    jobs.append(dict(file=str(p.relative_to(G)),sha256=sha(p),content=p.read_text()))
    prior.update(int(line.split()[-1]) for line in p.read_text().splitlines() if line.strip())
result_dirs=[p.name for p in (G/'eg_quality_validation_v1_20261018/results').iterdir() if p.is_dir()]
for s in result_dirs:
    m=re.search(r'_(\d{4})$',s)
    if m:prior.add(int(m.group(1)))
overlap=sorted(set(confirmation)&prior)
assert not any((R/f'exports/{role}/scene_{s:04d}').exists() for role in ['subset','official'] for s in confirmation),'audit must precede confirmation predictions'
audit=dict(status='NO_OVERLAP_IN_KNOWN_MIGRATED_EVIDENCE' if not overlap else 'EXPOSED_DOWNGRADE_CONFIRMATION_NOT_BLIND',confirmation_scenes=confirmation,prior_evaluated_scenes=sorted(prior),overlap=overlap,sources=[dict(file=str(p.relative_to(G)),sha256=sha(p)) for p in sources],derivative_job_lists=jobs,result_directories=result_dirs,limitation='Checks available migrated experiment records; no assertion about unknown or unavailable external histories. Scene list never changed according to scores.')
write_json(out/'confirmation_exposure_audit.json',audit)
records=[];groups={};fullgroups={};start=time.time()
for split,scenes in SPLIT['official_confirmation'].items():
    for s in scenes:
        root=DATA/f'scenes/scene_{s:04d}/kinect';poses=np.load(root/'camera_poses.npy');align=np.load(root/'cam0_wrt_table.npy')
        for f in SPLIT['confirmation_frames']:
            meta_path=root/f'meta/{f:04d}.mat';meta=sio.loadmat(meta_path)
            geometric=hashlib.sha256()
            for k in ['intrinsic_matrix','cls_indexes','poses','factor_depth']:
                arr=np.ascontiguousarray(meta[k]);geometric.update(k.encode());geometric.update(str(arr.dtype).encode());geometric.update(str(arr.shape).encode());geometric.update(arr.tobytes())
            geometric.update(np.ascontiguousarray(poses[f]).tobytes());geometric.update(np.ascontiguousarray(align).tobytes())
            rgb=sha(root/f'rgb/{f:04d}.png');depth=sha(root/f'depth/{f:04d}.png');obs=hashlib.sha256((rgb+depth).encode()).hexdigest();full=hashlib.sha256((obs+geometric.hexdigest()).encode()).hexdigest()
            row=dict(split=split,scene=s,frame=f,rgb_sha256=rgb,depth_sha256=depth,meta_file_sha256=sha(meta_path),canonical_geometry_sha256=geometric.hexdigest(),observation_key=obs,full_identity_key=full)
            records.append(row);groups.setdefault(obs,[]).append([s,f]);fullgroups.setdefault(full,[]).append([s,f])
        print(json.dumps(dict(scene=s,frames=256,elapsed=time.time()-start)),flush=True)
with open(out/'confirmation_raw_identity.csv','w',newline='') as h:
    w=csv.DictWriter(h,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
dup=[v for v in groups.values() if len(v)>1];full_dup=[v for v in fullgroups.values() if len(v)>1]
canonical={k:min(v) for k,v in groups.items()}
write_json(out/'confirmation_duplicate_audit.json',dict(status='COMPLETE_BEFORE_PREDICTIONS',frames=len(records),unique_rgb_depth_observations=len(groups),unique_full_identities=len(fullgroups),duplicate_observation_groups=dup,duplicate_full_identity_groups=full_dup,sensitivity_policy='One smallest(scene,frame) representative per exact RGB+depth byte-identity group; recompute scene/stratum means without changing native 256-frame primary AP. Any empty scene is reported, not silently imputed.',canonical_observations=canonical,wall_seconds=time.time()-start))
