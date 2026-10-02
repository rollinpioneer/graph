import json,glob,os,torch
OUT=open('/home/xushijie2/tmp/r2/R2_OUT.txt').read().strip(); ANA=OUT+'/zero_sample_analysis'
R='/home/xushijie2/graph_cp_disr_final_tb_launch/runs/final_master/2.1.1/T_B/'
def sd(p): 
    o=torch.load(p,map_location='cpu',weights_only=False); return o['model'],o
res={}
for plan,pat,last in [('R-TB-E-0','B1-K+E/seed_0/R-TB-E-0-*',14),('R-TB-DK-1','B2/seed_1/R-TB-DK-1-*',13),('R-TB-K-1','B1-K/seed_1/R-TB-K-1-*',14)]:
    d=glob.glob(R+pat)[0]; ck=d+'/checkpoints/'
    fin=[f for f in os.listdir(ck) if f.startswith('final_n_') and f.endswith('.pt')][0]
    F,fo=sd(ck+fin); G,go=sd(ck+'update_fragment_1.pt'); C,co=sd(ck+'update_complete_%d.pt'%last)
    def md(a,b): return max((a[k]-b[k]).abs().max().item() for k in a if a[k].is_floating_point()), sum(1 for k in a if a[k].is_floating_point() and (a[k]-b[k]).abs().max().item()>0)
    r={'final_vs_update_fragment_1':md(F,G),'final_vs_update_complete_%d'%last:md(F,C),'fragment_vs_complete':md(G,C)}
    # optimizer state equality final vs fragment
    def opt_eq(a,b):
        try:
            sa,sb=a['optimizer']['state'],b['optimizer']['state']
            if sa.keys()!=sb.keys(): return False
            for k in sa:
                for kk in sa[k]:
                    x,y=sa[k][kk],sb[k][kk]
                    if torch.is_tensor(x):
                        if not torch.equal(x,y): return False
                    elif x!=y: return False
            return True
        except Exception as e: return 'ERR %s'%e
    r['optimizer_state_final_equals_fragment']=opt_eq(fo,go)
    r['manifest_N_final_fragment_complete']=[fo['manifest'].get('N'),go['manifest'].get('N'),co['manifest'].get('N')]
    res[plan]=r
    print(plan,r)
json.dump(res,open(ANA+'/final_vs_fragment.json','w'),indent=1)
