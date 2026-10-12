"""New deterministic scene streams, full state and physical duplicate audits."""
from preflight import *
def physical(s):
    return hashlib.sha256(b''.join(np.asarray(s[k],dtype='<f8').tobytes() for k in ('qpos','qvel','goal'))).hexdigest()
def old_scenes():
    hashes=set();full=set();files=[]
    def collect(x):
        if isinstance(x,dict):
            if all(k in x for k in ('qpos','qvel','goal')) and np.asarray(x['qpos']).shape==(2,):hashes.add(physical(x));full.add(digest(x))
            for k in ('initial','scene','trajectories'):
                if k in x:collect(x[k])
        elif isinstance(x,(list,tuple)):
            for item in x:collect(item)
    paths=list((PREV/'runs').glob('*_SCENES.pkl'))+list((EV/'runs').glob('A_*.pkl'))+list((EV/'runs').glob('B_*.pkl'))
    for p in paths:
        with p.open('rb') as f:collect(pickle.load(f))
        files.append(dict(path=str(p),sha256=sha(p)))
    return hashes,full,files
def main():
    previous,previous_full,files=old_scenes();rows=[];identities=[]
    for kind in ('navigate','stitch'):
        env=env_new(kind)
        plans=[('C',[1070131+(kind=='stitch'),1170131+(kind=='stitch')],50,range(1,6))]
        if kind=='navigate':plans.append(('Rconfirm',[1270131,1370131],5,[5]))
        for phase,seeds,n,tasks in plans:
            scenes=[]
            for group,base in enumerate(seeds):
                np.random.seed(base);random.seed(base);env.unwrapped.np_random=gym.utils.seeding.np_random(base)[0];env.action_space.seed(base+2000000)
                for task in tasks:
                    for ep in range(n):
                        s,info=env.reset(options={'task_id':task,'render_goal':False});x=snapshot(env,info,None)
                        x.update(scene_id=f'{phase}_{kind}_g{group}_t{task}_e{ep}',group=group,episode=ep,stream=base)
                        ph=physical(x);duplicate=ph in previous
                        rows.append(dict(kind=kind,phase=phase,scene_id=x['scene_id'],task=task,group=group,episode=ep,stream=base,initial_x=float(s[0]),initial_y=float(s[1]),goal_x=float(info['goal'][0]),goal_y=float(info['goal'][1]),
                            physical_sha256=ph,full_state_sha256=digest(x),old_physical_duplicate=duplicate,old_full_duplicate=digest(x) in previous_full))
                        scenes.append(x);previous.add(ph)
            p=ROOT/'runs'/f'{phase}_{kind}_SCENES.pkl';dump(p,scenes);identities.append(dict(path=str(p),sha256=sha(p),count=len(scenes),phase=phase,kind=kind))
        env.close()
    csv_rows(ROOT/'results/SCENES.csv',rows)
    write(ROOT/'results/SCENE_IDENTITIES.json',dict(scenes=identities,old_scene_files=files,old_unique_physical_count=len(previous),duplicates=sum(x['old_physical_duplicate'] for x in rows),
        status='CONFIRMATORY_INDEPENDENCE_LIMITED' if any(x['old_physical_duplicate'] for x in rows) else 'NOVEL_FULL_STATE_AND_PHYSICAL_SCENES',note='No duplicates replaced or seeds adjusted; scene-level stochastic independence does not create independent training models'))
    print('SCENES_FROZEN',len(rows),sum(x['old_physical_duplicate'] for x in rows),flush=True)
if __name__=='__main__':main()
