"""Frozen-weight R2: integer-bin, GT-free local-coordinate reencoding.

Only existing 256 train queries receive analytic outcomes, by their unchanged
48-action grid. No evaluator, optimizer, C3/C4/C5 input, or source edits.
"""
import sys, json, hashlib, random, time, math
from pathlib import Path
import numpy as np
import torch

mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
B = Path.home() / 'work/grasp'
OLD = B / 'eg_pose_quality_diagnosis_v1_20261022'
PRIOR = B / 'eg_module_controlled_diagnosis_20261023'
ROOT = B / 'eg_causal_mechanism_v2_20261023/r2'
SRC = B / 'eg_full_repro_v1_20261015/runtime/EconomicGrasp'
D = B / 'eg_sub25_v1_20261015/data/graspnet'
sys.path[:0] = [str(SRC), str(SRC / 'libs/pointnet2')]
sys.argv = ['r2', '--dataset_root', str(D), '--camera', 'kinect', '--batch_size', '1']
from models.economicgrasp import economicgrasp, pred_decode
import libs.pointnet2.pointnet2_utils as PU

P = json.loads((PRIOR / 'protocol.json').read_text())
K = [0, 1, 3, 6]
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
def cpu(x): return x.detach().cpu().numpy()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rx(a):
    a = np.asarray(a); r = np.zeros(a.shape + (3, 3), dtype=float)
    r[..., 0, 0] = 1; r[..., 1, 1] = r[..., 2, 2] = np.cos(a)
    r[..., 2, 1] = np.sin(a); r[..., 1, 2] = -np.sin(a)
    return r
def softmax(x, axis=0):
    y = np.exp(x.astype(float) - x.max(axis=axis, keepdims=True))
    return y / y.sum(axis=axis, keepdims=True)
def symerr(a, b):
    direct = np.max(abs(a-b), axis=(-1,-2))
    swapped = np.max(abs(a-b@np.diag([1.,-1.,-1.])), axis=(-1,-2))
    return float(np.minimum(direct,swapped).max())
def state_hash(net):
    h = hashlib.sha256()
    for n,v in net.state_dict().items(): h.update(n.encode()); h.update(cpu(v).tobytes())
    return h.hexdigest()

ROOT.mkdir(parents=True, exist_ok=True)
ck = B / 'eg_full_repro_v1_20261015/downloads/economicgrasp_kinect.tar'
assert sha(ck) == P['checkpoint_sha256']
net = economicgrasp(seed_feat_dim=512, is_training=False).cuda()
net.load_state_dict(torch.load(ck, map_location='cpu', weights_only=True)['model_state_dict'], strict=True)
net.eval(); initial = state_hash(net)
cy, head = net.cy_group, net.grasp_head
capt = {}
def pre(m,args): capt['cy_args'] = args
def grouped(m,args,out): capt['grouped'] = out
cy.register_forward_pre_hook(pre)
cy.grouper.register_forward_hook(grouped)
orig_query = PU.cylinder_query
def query(*a, **kw):
    out = orig_query(*a, **kw); capt['neighbors'] = out; return out
PU.cylinder_query = query
def pack(end):
    return dict(pred=cpu(pred_decode(end)[0]),
        angle_logits=cpu(end['grasp_angle_pred'])[0],
        depth_logits=cpu(end['grasp_depth_pred'])[0],
        quality_logits=cpu(end['grasp_score_pred'])[0],
        raw_width=cpu(end['grasp_width_pred'])[0].reshape(-1))

pairs = [(s,f) for s in P['train_probe_scenes'] for f in P['train_probe_frames']]
if mode == 'pilot': pairs = pairs[:1]
for s,f in pairs:
    tag = f'train_{s:04d}_{f:04d}'
    dest = ROOT / (tag + '.npz')
    if dest.exists():
        assert dest.with_suffix('.json').exists(), 'incomplete previous frame'
        continue
    start = time.time()
    with np.load(OLD / f'cache/{tag}.npz') as zz: a = {k:zz[k] for k in zz.files}
    with np.load(PRIOR / f'cache/{tag}.npz') as z: ids = z['grid_ids']; oldpred = z['H0_pred']
    with np.load(PRIOR / f'evaluation/{tag}.npz') as z:
        assert np.array_equal(ids, z['grid_ids'])
        grid = {k:z['grid_'+k].reshape(len(ids),12,4) for k in ['mu','collision','empty']}
    assert len(ids) == 16
    seed = int(a['seed']); np.random.seed(seed); random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    points = a['input_points'].astype(np.float32)
    data = {'point_clouds':torch.from_numpy(points[None]).cuda(),
            'coordinates_for_voxel':torch.from_numpy((points / .005)[None]).cuda()}
    with torch.no_grad():
        end = net(data); base = pack(end)
        assert np.array_equal(base['pred'], oldpred), (tag, 'historical replay', float(abs(base['pred']-oldpred).max()))
        assert np.array_equal(cpu(end['grasp_top_view_inds'])[0], a['view_index'])
        args = tuple(t.detach().clone() for t in capt['cy_args'])
        group0 = capt['grouped'].detach().clone(); neighbor0 = cpu(capt['neighbors']).copy()
        rb = a['view_rotation'].astype(float)
        assert np.array_equal(cpu(args[2])[0], a['view_rotation'])
        a0 = base['angle_logits'][:12].argmax(0); d0 = base['depth_logits'][:4].argmax(0)
        out = dict(candidate_id=ids, scene=np.array(s), frame=np.array(f),
                   baseline_pred=base['pred'][ids], baseline_a=a0[ids], baseline_d=d0[ids],
                   valid_mask=a['batch_valid_mask'][ids], q_ref=a['batch_grasp_score'][ids],
                   matched_object_id=a['matched_object_id'][ids])
        out.update({'grid_'+k:v for k,v in grid.items()})
        receipts = []
        for k in K:
            phi = k*np.pi/12; T = rx(phi); rbnew = rb@T
            # Every a,d has the same center, approach, width and physical pose,
            # modulo the parallel-jaw pi symmetry. Mapping: old_a=(new_a+k)%12.
            angles = np.arange(12); physical = (angles+k)%12
            err = symerr(rbnew[:,None]@rx(angles*np.pi/12),rb[:,None]@rx(physical*np.pi/12))
            assert err < 2e-6, ('physical_action_set',k,err)
            control_a = (a0-k)%12
            identity_pose_err = symerr(rbnew@rx(control_a*np.pi/12),rb@rx(a0*np.pi/12))
            assert identity_pose_err < 2e-6
            # Pure cached-output relabeling; do not re-softmax a permutation.
            cached_prob = softmax(base['angle_logits'][:12])
            relabeled = np.roll(cached_prob, -k, axis=0)
            assert np.array_equal(np.roll(relabeled,k,axis=0),cached_prob)
            assert np.array_equal((control_a+k)%12,a0)
            transform = group0.clone()
            if k:
                xyz = cpu(group0[:,:3]).transpose(0,2,3,1)
                transform[:,:3] = torch.from_numpy(np.einsum('bnsi,ij->bnsj',xyz,T).transpose(0,3,1,2)).to(group0)
            assert torch.equal(transform[:,3:], group0[:,3:])
            h = cy.grouper.register_forward_hook(lambda m,aa,oo:transform)
            try: feat = cy(*args)
            finally: h.remove()
            assert np.array_equal(neighbor0,cpu(capt['neighbors']))
            z = pack(head(feat,dict(end)))
            if k == 0:
                for name in base: assert np.array_equal(z[name],base[name]), ('k0',name)
            for name in ['angle_logits','depth_logits','quality_logits']:
                out[f'k{k}_{name}'] = z[name][:,ids]
            out[f'k{k}_raw_width'] = z['raw_width'][ids]
            out[f'k{k}_score'] = z['pred'][ids,0]
            out[f'k{k}_a_physical'] = (z['angle_logits'][:12].argmax(0)[ids]+k)%12
            out[f'k{k}_d'] = z['depth_logits'][:4].argmax(0)[ids]
            out[f'k{k}_decoded_width_secondary_only'] = z['pred'][ids,1]
            receipts.append(dict(k=k,physical_48_action_set_max_abs=err,
                cached_output_relabel_identity_pose_max_abs=identity_pose_err,
                cached_output_relabel_prob_exact=True,neighbors_exact=True,
                upstream_features_exact=True,query_center_exact=True,baseline_width_frozen=True))
        # Audit actual train-label vs inference bases on these fixed inference
        # queries. Their approaches are not exactly equal, so this is not pure roll.
        trainrb = a['top_grasp_views_rot'].astype(float)
        approach_deg = np.degrees(np.arccos(np.clip((trainrb[:,:,0]*rb[:,:,0]).sum(1),-1,1)))
        roll = np.arctan2((trainrb[:,:,1]*rb[:,:,2]).sum(1),(trainrb[:,:,1]*rb[:,:,1]).sum(1))
        roll = (roll+np.pi/2)%np.pi-np.pi/2
        offgrid_deg = np.degrees(abs((roll+np.pi/24)%(np.pi/12)-np.pi/24))
        out['basis_train_infer_approach_deg_all'] = approach_deg
        out['basis_train_infer_roll_deg_all'] = np.degrees(roll)
        out['basis_train_infer_offgrid_deg_all'] = offgrid_deg
        out['basis_all_valid_mask'] = a['batch_valid_mask']
    assert state_hash(net) == initial, 'parameters or buffers changed'
    np.savez_compressed(dest,**out)
    receipt = dict(frame=tag,checkpoint_sha256=P['checkpoint_sha256'],historical_replay_exact=True,
        k0_exact=True,conditions=receipts,model_and_buffers_unchanged=True,
        source_cache_sha256=sha(OLD/f'cache/{tag}.npz'),
        grid_evaluation_sha256=sha(PRIOR/f'evaluation/{tag}.npz'),
        output_sha256=sha(dest),new_analytic_calls=0,parameter_updates=0,
        controls={'net_is_training':net.is_training,'view_is_training':net.view.is_training,'torch_training':net.training},
        seconds=time.time()-start)
    dest.with_suffix('.json').write_text(json.dumps(receipt,indent=2))
    print('R2_DONE',tag,json.dumps(receipt),flush=True)
    del end,data,args,group0,capt,out,feat
    capt={}; torch.cuda.empty_cache()
(ROOT / (mode+'_complete.json')).write_text(json.dumps(dict(frames=len(pairs),model_unchanged=True,parameter_updates=0,new_analytic_calls=0)))
print('R2_COMPLETE',mode,flush=True)
