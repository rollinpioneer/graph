"""Freeze complete scenes, independent of any outcome."""
from state_core import *
for kind in ('navigate','stitch'):
    env=env_new(kind);rows=[]
    for phase,bases,n in [('E1',[73001+(kind=='stitch')],20),('final',[83001+(kind=='stitch'),93001+(kind=='stitch')],50)]:
        scenes=[]
        for group,base in enumerate(bases):
            np.random.seed(base);random.seed(base)
            env.unwrapped.np_random=gym.utils.seeding.np_random(base)[0];env.action_space.seed(base+2000000)
            for task in range(1,6):
                for ep in range(n):
                    s,info=env.reset(options={'task_id':task,'render_goal':False})
                    snap=snapshot(env,info,None)
                    snap.update(scene_id=f'{phase}_{kind}_g{group}_t{task}_e{ep}',group=group,episode=ep,stream=base)
                    scenes.append(snap)
                    rows.append(dict(phase=phase,kind=kind,scene_id=snap['scene_id'],task=task,group=group,episode=ep,
                        stream=base,initial_x=float(s[0]),initial_y=float(s[1]),goal_x=float(info['goal'][0]),goal_y=float(info['goal'][1]),
                        distance=float(np.linalg.norm(s-info['goal'])),full_state_sha256=digest(snap)))
        path=ROOT/'runs'/f'{phase}_{kind}_SCENES.pkl';dump(path,scenes)
        write(ROOT/'results'/f'{phase}_{kind}_SCENES_IDENTITY.json',dict(path=str(path),sha256=sha(path),count=len(scenes)))
    csv_rows(ROOT/'results'/f'{kind}_SCENES.csv',rows);env.close()
print('SCENES_FROZEN',flush=True)
