from common import *
cfg=json.load(open(R/'frozen_config.json'));checked=0
for scene in sum(cfg['C4'].values(),[]):
 p=R/f'evaluation/confirmation/{scene:04d}/0000.json';rows=json.load(open(p))
 with np.load(p.with_suffix('.npz')) as z:
  for v in rows:
   q=z['quality'][z['selected_'+v['configuration']]];n=min(50,len(q));success=np.zeros((50,6),bool);success[:n]=(q[:n,None]>0)&(q[:n,None]<=np.array([.2,.4,.6,.8,1.,1.2])[None,:]);matrix=success.cumsum(0)/np.arange(1,51)[:,None]
   actual=[matrix.mean()*100,matrix[:,1].mean()*100,matrix[:,3].mean()*100];expected=[v['ap'],v['ap04'],v['ap08']];assert actual==expected,(scene,v['configuration'],actual,expected);checked+=3
write(R/'results/independent_AP_check.json',dict(status='PASS',scenes=12,frames=12,configuration_records=168,AP_AP04_AP08_values_checked=checked,implementation='independent boolean padding and cumulative rank precision; no evaluation helper imported'))
print('INDEPENDENT_AP_PASS',checked,flush=True)
