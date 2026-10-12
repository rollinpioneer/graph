"""New T evaluation scenes, sealed before T outcomes, distinct from R and C."""
from freeze_scenes import *
def main():
    seen,full,oldfiles=old_scenes()
    for x in csv.DictReader(open(ROOT/'results/SCENES.csv')):seen.add(x['physical_sha256'])
    with open(ROOT/'runs/R_SNAPSHOTS_stitch.pkl','rb') as f:
        for x in pickle.load(f):seen.add(physical(x['scene']))
    env=env_new('stitch');rows=[];scenes=[]
    for group,base in enumerate([2530132,2630132]):
        np.random.seed(base);random.seed(base);env.unwrapped.np_random=gym.utils.seeding.np_random(base)[0];env.action_space.seed(base+2000000)
        for task in range(1,6):
            for ep in range(50):
                s,info=env.reset(options={'task_id':task,'render_goal':False});x=snapshot(env,info,None);x.update(scene_id=f'T_stitch_g{group}_t{task}_e{ep}',group=group,episode=ep,stream=base)
                ph=physical(x);rows.append(dict(scene_id=x['scene_id'],task=task,group=group,episode=ep,stream=base,physical_sha256=ph,full_sha256=digest(x),previous_physical_duplicate=ph in seen));seen.add(ph);scenes.append(x)
    path=ROOT/'runs/T_SCENES.pkl';dump(path,scenes);csv_rows(ROOT/'results/T_SCENES.csv',rows);write(ROOT/'results/T_SCENE_IDENTITIES.json',dict(path=str(path),sha256=sha(path),count=500,duplicates=sum(x['previous_physical_duplicate'] for x in rows),streams=[2530132,2630132],no_replacement=True));env.close()
if __name__=='__main__':main()
