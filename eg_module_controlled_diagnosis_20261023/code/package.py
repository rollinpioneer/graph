from pathlib import Path
import zipfile,io,numpy as np,json
R=Path.home()/'work/grasp/eg_module_controlled_diagnosis_20261023'
with zipfile.ZipFile(R/'light_results.zip','w',zipfile.ZIP_DEFLATED) as z:
 for folder in ['gradient','receipts','evaluation']:
  for p in sorted((R/folder).glob('*')):z.write(p,str(p.relative_to(R)))
 for p in sorted((R/'cache').glob('*.npz')):
  a=np.load(p);out={k:a[k] for k in a.files if k.endswith('_pred') or k.endswith('_logits') or k in ['mechanism_ids','grid_ids','E1_phi','E1_basis','candidate_id','head_attention']}
  if p.name.startswith('train'):out['quality_features']=a['tokens_post'][:,3,:]
  if 'E1_F' in a.files:out['E1_F_relative_L2']=np.linalg.norm(a['E1_F']-a['vp_features'])/np.linalg.norm(a['vp_features'])
  out['N_F_relative_L2']=np.linalg.norm(a['N_F']-a['vp_features'])/np.linalg.norm(a['vp_features'])
  buf=io.BytesIO();np.savez_compressed(buf,**out);z.writestr('cache/'+p.name,buf.getvalue())
 for p in R.glob('*.json'):z.write(p,p.name)
 for name in ['probe.py','evaluate_module.py']:z.write(R/name,'code/'+name)
print(json.dumps({'zip_bytes':(R/'light_results.zip').stat().st_size}))
