"""Read-only asset/config audits and independently seeded initialization proof."""
from state_core import *
import subprocess
PREV=Path('/home/xushijie3/work/graph_saw_analysis03b_20261012/experiments/saw_analysis03b_20261012')

def equal(a,b):
    if isinstance(a,dict):return set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return np.array_equal(np.asarray(a),np.asarray(b))
def main():
    official=OLD/'vendor/saw';h=subprocess.check_output(['git','rev-parse','HEAD'],cwd=official,text=True).strip()
    assert h=='3535a16c901e5de1233ac5a586401b3ddfa5e30d'
    assert not subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=official,text=True).strip()
    inputs=[];cps=[]
    for source in (OLD,EV,PREV):
        manifest=json.loads((source/'LIGHT_MANIFEST.json').read_text())
        for f in manifest['files']:assert sha(source/f['relative'])==f['sha256'],str(source/f['relative'])
        inputs.append(dict(root=str(source),manifest_sha256=sha(source/'LIGHT_MANIFEST.json'),verified_files=len(manifest['files'])))
    for source in (OLD,PREV):
        for kind in ('navigate','stitch'):
            for cp in json.loads((source/'results'/f'{kind}_checkpoints.json').read_text()):
                assert sha(cp['path'])==cp['sha256'];cps.append(cp)
    pairs=[]
    for kind in ('navigate','stitch'):
        for seed in (1,2):
            env,data,val,a=load(f'pointmaze-large-{kind}-v0','A');b=dict(a,actor_p_trajgoal=.5,actor_p_randomgoal=.5)
            orig=json.loads((OLD/'runs'/f'formal_{kind}'/'config.json').read_text())['config']
            assert json.loads(json.dumps(a))==orig
            for v,cfg in (('A',a),('B',b)):
                agent=init_agent(data,cfg,seed);s=flax.serialization.to_state_dict(jax.device_get(agent));sync(agent)
                if v=='A':reference=s
                else:assert equal(reference,s),'CPU_UPDATE0_PAIR_IDENTITY'
            pairs.append(dict(kind=kind,seed=seed,agent_optimizer_rng_equal=True,network_step=int(reference['network']['step']),
                serialized_agent_sha256=hashlib.sha256(pickle.dumps(reference,4)).hexdigest()))
            env.close()
    write(ROOT/'results/PREFLIGHT.json',dict(status='PASS',official_commit=h,original_manifests=inputs,readonly_checkpoints=cps,CPU_update0=pairs,
        GPU_actual_update0_pair='verify actual formal checkpoint pairs before interpreting any training outcomes',additional_training_updates=0))
    print('PREFLIGHT_PASS',flush=True)
if __name__=='__main__':main()
