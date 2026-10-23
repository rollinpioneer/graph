"""R1.1 frozen-query label provenance and storage-order tie intervention.
Read only: existing 16 training frames, raw object labels, scene collision grids.
Exact object-coordinate identity is required; no nearest-GT quality substitution.
Output gradients are analytic derivatives at the cached logits/raw width, not model updates.
"""
import argparse,csv,json,time,hashlib
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from width_diagnostics import SCENES,FRAMES,writecsv,sha

ORDERS={'original':np.arange(48),'cycle17':np.roll(np.arange(48),-17),'reverse':np.arange(47,-1,-1)}

def softmax(x):
    z=x.astype(np.float64)-np.max(x);e=np.exp(z);return e/e.sum()

def quant_width(w):
    # Same float32 millimetre quantization / uint8 storage as generate_economic.py.
    return float(np.asarray(np.float32(w)*np.float32(1000)).astype(np.uint8).astype(np.float32)/np.float32(1000))

def save_rows(path,rows):writecsv(path,rows)

def exact_keys(points):
    a=np.ascontiguousarray(points,dtype=np.float32)
    return [r.tobytes() for r in a]

def one_scene(s,frames,old,dataset,out):
    todo=[f for f in frames if not (out/f'frames/train_{s:04d}_{f:04d}_receipt.json').exists()]
    if not todo:return
    start=time.time();scene=f'scene_{s:04d}';ep=dataset/f'economic_grasp_label_300views/{scene}_labels.npz'
    cp=dataset/f'collision_label/{scene}/collision_labels.npz';mp=dataset/f'scenes/{scene}/kinect/meta/0000.mat'
    meta0=loadmat(mp);oids=meta0['cls_indexes'].reshape(-1).astype(int)-1
    with np.load(ep) as z:lab={k:z[k] for k in ['points','pointid','topview','rotations','depth','scores','widths']}
    eh=sha(ep);sources=[dict(path=str(ep),sha256=eh),dict(path=str(mp),sha256=sha(mp)),dict(path=str(cp),exists=cp.exists())]
    caches={f:dict(np.load(old/f'train_{s:04d}_{f:04d}.npz')) for f in todo}
    rows={f:[] for f in todo};ties={f:[] for f in todo};grids={f:[] for f in todo}
    cq=np.load(cp) if cp.exists() else None
    requests={}
    for f,a in caches.items():
        assert eh==str(a['label_sha256']),('frozen_label_hash_changed',scene)
        assert np.array_equal(a['object_ids_meta'],oids),('meta_object_order_changed',scene,f)
        for i in np.flatnonzero(a['batch_valid_mask']):
            oi=int(a['matched_object_id'][i]);requests.setdefault(oi,[]).append((f,int(i)))
    for oid,req in sorted(requests.items()):
        op=dataset/f'grasp_label/{oid:03d}_labels.npz';inst=np.flatnonzero(oids==oid)
        if len(inst)!=1 or not op.exists():
            for f,i in req:rows[f].append(dict(frame_id=f'train_{s:04d}_{f:04d}',scene=s,frame=f,candidate_id=i,object_id=oid,q_ref=float(caches[f]['batch_grasp_score'][i]),status='unknown',unknown_reason='ambiguous_object_instance' if len(inst)!=1 else 'raw_object_label_missing'))
            continue
        instance=int(inst[0]);sources.append(dict(path=str(op),exists=True))
        z=np.load(op);points=z['points'];index={}
        for ri,key in enumerate(exact_keys(points)):index.setdefault(key,[]).append(ri)
        selected=[]
        for f,i in req:
            a=caches[f];pi=int(a['label_point_index'][i]);slot=int(a['label_slot'][i]);view=int(a['label_source_view'][i])
            r=dict(frame_id=f'train_{s:04d}_{f:04d}',scene=s,frame=f,candidate_id=i,object_id=oid,object_instance=instance,
                label_point_index=pi,label_slot=slot,source_view=view,q_ref=float(a['batch_grasp_score'][i]),
                a_ref=int(a['batch_grasp_rotations'][i]),d_ref=int(a['batch_grasp_depth'][i]),w_ref_m=float(a['batch_grasp_width'][i]),
                match_distance_m=float(a['dist'][i]),status='unknown',unknown_reason='')
            found=index.get(np.ascontiguousarray(a['label_object_point'][i],dtype=np.float32).tobytes(),[])
            if len(found)!=1:r['unknown_reason']='raw_point_exact_missing' if not found else 'raw_point_exact_duplicate'
            elif not (0<=pi<len(lab['points']) and 0<=slot<lab['topview'].shape[1]):r['unknown_reason']='frozen_index_invalid'
            elif not (lab['pointid'][pi]==instance and np.array_equal(lab['points'][pi],a['label_object_point'][i]) and int(lab['topview'][pi,slot])==view):r['unknown_reason']='frozen_identity_not_reproduced'
            else:
                # Original GPU scatter can have duplicate-destination ambiguity; never silently rematch.
                stored=(int(lab['rotations'][pi,slot]),int(lab['depth'][pi,slot]),float(np.float32(lab['scores'][pi,slot])/10),float(np.float32(lab['widths'][pi,slot])/1000))
                frozen=(r['a_ref'],r['d_ref'],r['q_ref'],r['w_ref_m'])
                if not (stored[:2]==frozen[:2] and abs(stored[2]-frozen[2])<1e-7 and abs(stored[3]-frozen[3])<1e-7):r['unknown_reason']='frozen_slot_target_ambiguity'
            if r['unknown_reason']:rows[f].append(r)
            else:r['raw_point_index']=found[0];selected.append((f,i,r,found[0],view))
        if not selected:z.close();continue
        # Read each large array once per object/scene. Keep only requested 48-cell slices.
        raw=z['scores'];mus=[raw[ri,v].copy() for _,_,_,ri,v in selected];del raw
        offsets=z['offsets'];offs=[offsets[ri,v].copy() for _,_,_,ri,v in selected];del offsets;z.close()
        if cq is not None and f'arr_{instance}' in cq.files:
            coll=cq[f'arr_{instance}'];cols=[coll[ri,v].copy() for _,_,_,ri,v in selected];del coll
        else:cols=[None]*len(selected)
        for (f,i,r,ri,v),mu,off,col in zip(selected,mus,offs,cols):
            assert mu.shape==(12,4) and off.shape==(12,4,3)
            width=off[:,:,2];rawpos=mu>0
            r.update(raw_nonpositive_cells=int((mu<=0).sum()),raw_negative_cells=int((mu<0).sum()),raw_zero_cells=int((mu==0).sum()),
                raw_positive_cells=int(rawpos.sum()),raw_main_quality_cells=int(((mu>0)&(mu<=.4)).sum()),
                raw_width_zero_cells=int((width==0).sum()),raw_width_over100mm_cells=int((width>.1).sum()),
                raw_width_uint8_overflow_cells=int((width*1000>=256).sum()))
            if col is None:r['unknown_reason']='scene_collision_grid_missing';rows[f].append(r);continue
            col=col.astype(bool);mu_after=np.where(col,0,mu).astype(np.float32);mu_after[mu_after<0]=0
            eligible=(mu_after>0)&(width<=np.float32(.1));quality=mu_after.copy();quality[~eligible]=0
            pos=quality>0;quality[pos]=np.float32(1.1)-quality[pos]
            maximum=float(quality.max());maxmask=quality==maximum;orig=int(np.argmax(quality.reshape(-1)));oa,od=divmod(orig,4)
            expectedq=float(np.asarray(np.float32(maximum)*np.float32(10)).astype(np.uint8).astype(np.float32)/np.float32(10));expectedw=quant_width(width[oa,od])
            reason=[]
            if not rawpos.any():reason.append('no_positive_raw_quality')
            if (rawpos&col).any():reason.append('scene_collision_excludes_positive_cells')
            if (rawpos&(width>.1)).any():reason.append('width_exceeds_100mm_on_positive_cells')
            if not eligible.any():reason.append('no_surviving_positive_cell')
            if eligible.any() and not (quality>0).any():reason.append('positive_mu_nonpositive_transformed_quality')
            if maximum>0 and expectedq==0:reason.append('positive_quality_quantized_to_zero')
            r.update(scene_collision_cells=int(col.sum()),collision_positive_cells=int((rawpos&col).sum()),
                post_collision_positive_cells=int((mu_after>0).sum()),post_collision_main_quality_cells=int(((mu_after>0)&(mu_after<=.4)).sum()),
                width_excludes_raw_positive_cells=int((rawpos&(width>.1)).sum()),width_excludes_noncollision_positive_cells=int(((mu_after>0)&(width>.1)).sum()),
                surviving_positive_mu_cells=int(eligible.sum()),positive_transformed_quality_cells=int((quality>0).sum()),
                label_quality_max=maximum,max_tie_cells=int(maxmask.sum()),all48_quality_exact_zero=bool(np.all(quality==0)),
                overlapping_reason_tags=';'.join(reason),raw_reference_mu=float(mu[oa,od]),raw_reference_width_m=float(width[oa,od]),
                raw_reference_collision=bool(col[oa,od]),raw_reference_width_positive=bool(width[oa,od]>0),
                selected_raw_width_zero=bool(width[oa,od]==0),selected_raw_width_over100mm=bool(width[oa,od]>np.float32(.1)),
                reconstructed_q_ref=expectedq,reconstructed_w_ref_m=expectedw,reconstructed_a_ref=oa,reconstructed_d_ref=od)
            if not (oa==r['a_ref'] and od==r['d_ref'] and abs(expectedq-r['q_ref'])<1e-7 and abs(expectedw-r['w_ref_m'])<1e-7):
                r['unknown_reason']='raw_filter_compression_target_mismatch';rows[f].append(r);continue
            r['status']='exact';r['unknown_reason']='';rows[f].append(r)
            a=caches[f];n=int(a['batch_valid_mask'].sum());pa=softmax(a['angle_logits'][:,i]);pd=softmax(a['depth_logits'][:,i]);rw=float(a['raw_width'].reshape(-1)[i])
            kind='all_zero' if np.all(quality==0) else ('positive_unique_max' if maximum>0 and maxmask.sum()==1 else ('positive_tied_max' if maximum>0 else 'other'))
            gorig=None
            for name,order in ORDERS.items():
                cell=int(order[np.argmax(quality.reshape(-1)[order])]);aa,dd=divmod(cell,4);ww=quant_width(width[aa,dd]);ga=pa.copy();gd=pd.copy();ga[aa]-=1;gd[dd]-=1;gw=float(np.clip(rw-10*ww,-1,1));gv=np.r_[ga/n,gd/n,10*gw/n]
                if gorig is None:gorig=gv
                ties[f].append(dict(frame_id=r['frame_id'],scene=s,frame=f,candidate_id=i,object_id=oid,population=kind,order=name,original_quality_label=r['q_ref'],quality_label_unchanged=True,
                    point_view_geometry_unchanged=True,physical_cell=cell,angle_class=aa,depth_class=dd,raw_angle_rad=float(off[aa,dd,0]),raw_depth_m=float(off[aa,dd,1]),raw_width_m=float(width[aa,dd]),quantized_width_m=ww,
                    angle_target_changed=bool(aa!=oa),depth_target_changed=bool(dd!=od),width_target_changed=bool(abs(ww-expectedw)>1e-7),width_raw_gradient=gw,
                    angle_ce_gradient_l2=float(np.linalg.norm(ga)),depth_ce_gradient_l2=float(np.linalg.norm(gd)),weighted_normalized_output_gradient_l2=float(np.linalg.norm(gv)),
                    weighted_normalized_gradient_delta_l2=float(np.linalg.norm(gv-gorig)),gradient_cosine_to_original=float(np.dot(gv,gorig)/(np.linalg.norm(gv)*np.linalg.norm(gorig))) if np.linalg.norm(gv)*np.linalg.norm(gorig)>0 else None,
                    angle_ce_gradient_json=json.dumps(ga.tolist()),depth_ce_gradient_json=json.dumps(gd.tolist()),valid_denominator=n))
            grids[f].append(dict(candidate_id=i,mu=mu,width=width,collision=col,quality=quality,offsets=off))
        print('OBJECT_DONE',s,oid,len(selected),flush=True)
    if cq is not None:cq.close()
    for f in todo:
        tag=f'train_{s:04d}_{f:04d}';rr=sorted(rows[f],key=lambda x:x['candidate_id']);tt=sorted(ties[f],key=lambda x:(x['candidate_id'],list(ORDERS).index(x['order'])));gg=sorted(grids[f],key=lambda x:x['candidate_id'])
        assert len(rr)==int(caches[f]['batch_valid_mask'].sum())
        save_rows(out/f'frames/{tag}_trace.csv',rr);save_rows(out/f'frames/{tag}_ties.csv',tt)
        if gg:np.savez_compressed(out/f'frames/{tag}_raw_grids.npz',**{k:np.array([g[k] for g in gg]) for k in gg[0]})
        receipt=dict(frame=tag,n=len(rr),zero=sum(r['q_ref']==0 for r in rr),exact=sum(r['status']=='exact' for r in rr),
            unknown_reasons={k:sum(r['unknown_reason']==k for r in rr) for k in sorted(set(r['unknown_reason'] for r in rr)) if k},
            seconds_scene=time.time()-start,updates=0,inference=0,raw_label_rewritten=False,nearest_gt_used=False,tie_orders={k:v.tolist() for k,v in ORDERS.items()},source_files=sources,
            cache_source=dict(path=str(old/f'{tag}.npz'),sha256=sha(old/f'{tag}.npz')))
        (out/f'frames/{tag}_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print('FRAME_DONE',json.dumps({k:v for k,v in receipt.items() if k not in ['source_files','tie_orders','cache_source']}),flush=True)

def collect(out):
    rr=[];tt=[]
    for p in sorted((out/'frames').glob('*_trace.csv')):
        with p.open(encoding='utf-8') as f:rr.extend(csv.DictReader(f))
    for p in sorted((out/'frames').glob('*_ties.csv')):
        with p.open(encoding='utf-8') as f:tt.extend(csv.DictReader(f))
    save_rows(out/'label_semantics_trace.csv',rr);save_rows(out/'tie_rule_gradients.csv',tt)
    summary=dict(n=len(rr),zero_quality=sum(float(r['q_ref'])==0 for r in rr),exact=sum(r['status']=='exact' for r in rr),scenes=sorted(set(int(r['scene']) for r in rr)),
        unknown_reasons={k:sum(r.get('unknown_reason','')==k for r in rr) for k in sorted(set(r.get('unknown_reason','') for r in rr)) if k},
        scope='Frozen inference queries; tie derivative at cached outputs only. No harmful parameter update inference.',updates=0,inference=0,tie_groups=[])
    for pop in sorted(set(r['population'] for r in tt)):
        for order in ORDERS:
            xs=[r for r in tt if r['population']==pop and r['order']==order]
            if xs:summary['tie_groups'].append(dict(population=pop,order=order,n=len(xs),scenes=len(set(r['scene'] for r in xs)),angle_changed=sum(r['angle_target_changed']=='True' for r in xs),depth_changed=sum(r['depth_target_changed']=='True' for r in xs),width_changed=sum(r['width_target_changed']=='True' for r in xs),gradient_changed=sum(float(r['weighted_normalized_gradient_delta_l2'])>1e-12 for r in xs)))
    (out/'label_semantics_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print('SUMMARY',json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();b=Path.home()/'work/grasp';p.add_argument('--cache',type=Path,default=b/'eg_pose_quality_diagnosis_v1_20261022/cache');p.add_argument('--dataset',type=Path,default=b/'eg_sub25_v1_20261015/data/graspnet');p.add_argument('--out',type=Path,default=b/'eg_causal_mechanism_v2_20261023/r1');p.add_argument('--pilot',action='store_true');p.add_argument('--collect-only',action='store_true');args=p.parse_args();(args.out/'frames').mkdir(parents=True,exist_ok=True)
    if not args.collect_only:
        for s in (SCENES[:1] if args.pilot else SCENES):one_scene(s,FRAMES[:1] if args.pilot else FRAMES,args.cache,args.dataset,args.out)
    collect(args.out)
