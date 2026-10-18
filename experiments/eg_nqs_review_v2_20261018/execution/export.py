"""Frozen inference feature export, official label matching on a separate endpoint dict."""
import os,sys,argparse,time,json,copy,resource
from pathlib import Path
import numpy as np
from common import *
p=argparse.ArgumentParser();p.add_argument('--scenes',required=True);p.add_argument('--role',default='subset');p.add_argument('--micro',action='store_true');a=p.parse_args()
scenes=[int(x) for x in a.scenes.split(',')]
sys.path.insert(0,str(CODE));sys.path.insert(0,str(CODE/'libs/pointnet2'))
sys.argv=[sys.argv[0],'--dataset_root',str(DATA),'--camera','kinect','--model','economicgrasp','--batch_size','1']
import torch
import scipy.io as sio
from dataset.graspnet_dataset import GraspNetDataset,collate_fn
from models.economicgrasp import economicgrasp,pred_decode
from utils.label_generation import process_grasp_labels,batch_viewpoint_params_to_matrix
os.nice(10);torch.set_num_threads(1);torch.manual_seed(20261018);torch.cuda.manual_seed_all(20261018)
checkpoint=OLD/'runs/EG_SUB25_V1/train/economicgrasp_epoch10.tar' if a.role=='subset' else G/'eg_full_repro_v1_20261015/downloads/economicgrasp_kinect.tar'
net=economicgrasp(seed_feat_dim=512,is_training=False).cuda().eval()
net.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model_state_dict'],strict=True)
for param in net.parameters():param.requires_grad_(False)
capture={}
def score_hook(module,inputs):capture['score']=inputs[0].detach().clone()
def group_hook(module,inputs,output):capture['group']=output.detach().clone()
handles=[net.grasp_head.conv_score.register_forward_pre_hook(score_hook),net.cy_group.register_forward_hook(group_hook)]
data=GraspNetDataset(str(DATA),split='train',camera='kinect',num_points=20000,voxel_size=.005,remove_outlier=True,augment=False,load_label=False)
start=time.time();rows=[];check=False
for scene in scenes:
    label=None
    if scene<100:
        with np.load(DATA/f'economic_grasp_label_300views/scene_{scene:04d}_labels.npz') as z:label={k:z[k] for k in z.files}
        assert label['topview'].shape[1]==300
    selected_frames=frames(scene)[:2] if a.micro else frames(scene)
    for frame in selected_frames:
        out=R/f'exports/{a.role}/scene_{scene:04d}/{frame:04d}.npz'
        if out.exists():continue
        inp=R/f'inputs/scene_{scene:04d}/{frame:04d}.npz'
        if inp.exists():
            with np.load(inp) as z:sample={k:z[k] for k in ['point_clouds','cloud_colors','coordinates_for_voxel','seg']}
        else:
            np.random.seed(seed(20261018,'input',scene,frame))
            # For official test scenes use the same official loader with only file indices changed.
            if scene<100:sample=data.get_data(scene*256+frame)
            else:
                for attr,sub,ext in [('colorpath','rgb','png'),('depthpath','depth','png'),('labelpath','label','png'),('metapath','meta','mat')]:
                    getattr(data,attr)[0]=str(DATA/f'scenes/scene_{scene:04d}/kinect/{sub}/{frame:04d}.{ext}')
                data.scenename[0]=f'scene_{scene:04d}';data.frameid[0]=frame;sample=data.get_data(0)
            save_npz(inp,**sample)
        batch={k:v.cuda() for k,v in collate_fn([sample]).items()}
        torch.cuda.synchronize();t=time.perf_counter()
        with torch.no_grad():ep=net(dict(batch));pred=pred_decode(ep)[0].cpu().numpy()
        torch.cuda.synchronize();elapsed=time.perf_counter()-t
        feature=torch.cat([capture['score'],capture['group']],1)[0].T.cpu().numpy()
        if not check:
            for h in handles:h.remove()
            with torch.no_grad():plain=pred_decode(net(dict(batch)))[0].cpu().numpy()
            assert np.array_equal(pred,plain),('HOOK_MISMATCH',float(np.max(np.abs(pred-plain))))
            handles=[net.grasp_head.conv_score.register_forward_pre_hook(score_hook),net.cy_group.register_forward_hook(group_hook)]
            check=True
        al=ep['grasp_angle_pred'][0,:12].cpu().numpy().T;dl=ep['grasp_depth_pred'][0,:4].cpu().numpy().T
        ai=np.argmax(al,axis=1);di=np.argmax(dl,axis=1);theta=ai*np.pi/12
        feature=np.column_stack([feature,np.sin(2*theta),np.cos(2*theta),pred[:,3]/.04,pred[:,1]/.1]).astype(np.float32)
        M=np.zeros(1024,bool);E=np.zeros(1024,np.float32);dist=np.full(1024,np.nan)
        if label is not None:
            meta=sio.loadmat(DATA/f'scenes/scene_{scene:04d}/kinect/meta/{frame:04d}.mat')
            q={'point_clouds':batch['point_clouds'],'xyz_graspable':ep['xyz_graspable'].clone(),'grasp_top_view_inds':ep['grasp_top_view_inds'].clone()}
            q['object_poses_list']=[[torch.from_numpy(meta['poses'][:,:,j].astype(np.float32)).cuda() for j in range(len(meta['cls_indexes']))]]
            fields={'grasp_points_list':('points',1),'grasp_rotations_list':('rotations',1),'grasp_depth_list':('depth',1),'grasp_scores_list':('scores',10),'grasp_widths_list':('widths',1000),'view_graspness_list':('vgraspness',1),'top_view_index_list':('topview',1)}
            for key,(field,scale) in fields.items():
                values=[]
                for j in range(len(meta['cls_indexes'])):
                    x=label[field][label['pointid']==j]
                    x=x.astype(np.int32) if field in ['rotations','depth','topview'] else x.astype(np.float32)/scale
                    values.append(torch.from_numpy(x).cuda())
                q[key]=[values]
            with torch.no_grad():_,q=process_grasp_labels(q)
            M=q['batch_valid_mask'][0].cpu().numpy();es=q['batch_grasp_score'][0].cpu().numpy()
            E=np.where(es>0,utility(1.1-es),0).astype(np.float32)
            dist=torch.linalg.norm(ep['xyz_graspable']-q['batch_grasp_point'],dim=-1)[0].cpu().numpy()
            assert np.all(dist[M]<.005)
        sample_ids=np.random.default_rng(seed(20261018,a.role,scene,frame)).choice(1024,256,replace=False)
        at=np.argsort(-al,axis=1,kind='stable')[:,:3];dt=np.argsort(-dl,axis=1,kind='stable')[:,:2]
        hy=np.repeat(pred[:,None,:],6,axis=1)
        approach=-ep['grasp_top_view_xyz'][0]
        for k,(x,y) in enumerate([(x,y) for x in range(3) for y in range(2)]):
            angles=torch.as_tensor(at[:,x],device='cuda')*np.pi/12
            rot=batch_viewpoint_params_to_matrix(approach,angles).reshape(1024,9).cpu().numpy()
            hy[:,k,4:13]=rot;hy[:,k,3]=(dt[:,y]+1)*np.float32(.01)
        assert np.array_equal(at[:,0],ai) and np.array_equal(dt[:,0],di)
        assert np.allclose(hy[:,0,1:],pred[:,1:],rtol=0,atol=1e-7)
        hy[:,0]=pred
        assert feature.shape==(1024,324) and np.isfinite(feature).all() and np.isfinite(pred).all()
        save_npz(out,pred=pred,feature=feature,angle_logits=al,depth_logits=dl,hypotheses=hy,M=M,E=E,distance=dist,sample_ids=sample_ids,input_sha256=np.array(sha(inp)))
        record=dict(role=a.role,scene=scene,frame=frame,M=int(M.sum()),sample_M=int(M[sample_ids].sum()),inference_seconds=elapsed,export_sha256=sha(out))
        rows.append(record);print(json.dumps(record),flush=True)
        del ep,batch
    del label
write_json(R/f'receipts/export_{a.role}_{scenes[0]}_{"micro" if a.micro else "full"}.json',dict(status='PASS',frames=len(rows),rows=rows,wall_seconds=time.time()-start,hook_exact=True,checkpoint_sha256=sha(checkpoint),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),cuda_peak=torch.cuda.max_memory_allocated(),rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
