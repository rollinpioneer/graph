#!/usr/bin/env python3
"""E-C 专用 eval driver —— 放进 /home/xushijie/CUPID/repo/ 后运行。

为什么需要它 (探针+实测发现):
  1. 仓库自带 eval.py 是 click, **不接受 n_test 覆盖** -> 只能跑 checkpoint 配置里的 50
  2. 它同时跑 n_train rollouts, 对 E-C 是纯浪费
  3. 需要保证逐 episode outcome 落盘

本脚本镜像 eval.py 的加载流程, 只做三处最小改动:
  - 覆盖 cfg.task.env_runner.n_test = N,  n_train = 0
  - 可覆盖 n_envs (并行度)
  - 把 runner 返回的**全部** log 原样落盘, 并额外抽出逐 episode 数组

用法:
  python ec_eval.py -c <ckpt> -o <outdir> --n-test 400
  python ec_eval.py -c <ckpt> -o <outdir> --n-test 4 --timing   # 先测吞吐
"""
import os, sys, json, time, pathlib, collections, re
import click, dill, torch, numpy as np
import hydra
from omegaconf import OmegaConf, open_dict

def _set(cfg, path, val):
    """安全覆盖嵌套字段; 不存在则跳过并返回 False。"""
    node=cfg
    parts=path.split('.')
    for p in parts[:-1]:
        if not hasattr(node,p) and p not in node: return False
        node=node[p]
    if parts[-1] not in node: return False
    with open_dict(node): node[parts[-1]]=val
    return True

def per_episode(log, n):
    """从 runner log 抽逐 episode 数组。两条路: 值是 list[n]; 或键名带序号。"""
    for k,v in log.items():
        if isinstance(v,(list,tuple)) and len(v)==n:
            try: return k, [float(x) for x in v]
            except Exception: pass
    pat=collections.defaultdict(dict)
    for k,v in log.items():
        m=re.match(r'^(.*?)(\d+)$',str(k))
        if m and isinstance(v,(int,float,np.floating)):
            pat[m.group(1)][int(m.group(2))]=float(v)
    best=None
    for base,dd in pat.items():
        if best is None or len(dd)>len(best[1]): best=(base,dd)
    if best and len(best[1])>=1:
        base,dd=best
        return base+'*', [dd[i] for i in sorted(dd)]
    return None, None

@click.command()
@click.option('-c','--checkpoint', required=True)
@click.option('-o','--output_dir', required=True)
@click.option('-d','--device', default='cuda:0')
@click.option('--n-test', type=int, default=None, help='评估 episode 数 (E-C 用 400)')
@click.option('--n-envs', type=int, default=None, help='并行环境数; 不给则用配置默认')
@click.option('--keep-train/--no-keep-train', default=False, help='是否保留 n_train rollouts (E-C 默认关)')
@click.option('--timing', is_flag=True, help='只测吞吐并打印 s/episode')
def main(checkpoint, output_dir, device, n_test, n_envs, keep_train, timing):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    payload=torch.load(open(checkpoint,'rb'), pickle_module=dill)
    cfg=payload['cfg']
    cls=hydra.utils.get_class(cfg._target_)
    # Some legacy workspaces (notably transformer policies) only accept cfg.
    try:
        workspace=cls(cfg, output_dir=output_dir)
    except TypeError as exc:
        if 'output_dir' not in str(exc):
            raise
        workspace=cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy=workspace.model
    if getattr(cfg.training,'use_ema',False): policy=workspace.ema_model
    dev=torch.device(device); policy.to(dev); policy.eval()

    applied={}
    if n_test is not None:
        applied['n_test']=_set(cfg,'task.env_runner.n_test',int(n_test))
        applied['n_test_vis']=_set(cfg,'task.env_runner.n_test_vis',0)
    if not keep_train:
        applied['n_train']=_set(cfg,'task.env_runner.n_train',0)
        applied['n_train_vis']=_set(cfg,'task.env_runner.n_train_vis',0)
    if n_envs is not None:
        applied['n_envs']=_set(cfg,'task.env_runner.n_envs',int(n_envs))
    print("[ec_eval] 覆盖结果:", applied, flush=True)
    print("[ec_eval] env_runner:", OmegaConf.to_container(cfg.task.env_runner, resolve=False), flush=True)

    t0=time.time()
    env_runner=hydra.utils.instantiate(cfg.task.env_runner, output_dir=output_dir)
    runner_log=env_runner.run(policy)
    dt=time.time()-t0

    log={}
    for k,v in runner_log.items():
        if isinstance(v,(int,float,str,bool)) or v is None: log[k]=v
        elif isinstance(v,(list,tuple)): log[k]=list(v)
        elif isinstance(v,np.ndarray): log[k]=v.tolist()
        else: log[k]=getattr(v,'_path',str(type(v)))
    n=int(n_test) if n_test is not None else None
    key,vals=per_episode(log, n) if n else (None,None)
    if vals is None: key,vals=per_episode(log, len(log))
    out=dict(eval_wallclock_sec=round(dt,2), n_test_requested=n,
             n_episodes_found=len(vals) if vals else 0,
             per_episode_key=key, per_episode=vals, raw=log)
    p=os.path.join(output_dir,'ec_eval.json'); json.dump(out,open(p,'w'),indent=1)
    print(f"\n[ec_eval] 墙钟 {dt:.1f}s")
    if vals:
        u=sorted(set(vals))
        print(f"[ec_eval] 逐 episode: key={key}  n={len(vals)}  取值={u[:8]}{'...' if len(u)>8 else ''}")
        print(f"[ec_eval] 二值? {set(u)<={0.0,1.0}}   均值={np.mean(vals):.4f}")
        print(f"[ec_eval] s/episode = {dt/len(vals):.2f}")
        if n and len(vals)!=n:
            print(f"[ec_eval] !! episode 数 {len(vals)} != 请求的 {n} —— n_test 覆盖未生效, 检查上面的 env_runner")
    else:
        print(f"[ec_eval] !! 未找到逐 episode 结果。raw keys: {sorted(log)[:20]}")
        print("[ec_eval]    需最小修改 env_runner 的 log, 把每个 episode 的原始结果落盘")
    if timing and vals:
        print(f"\n[ec_eval] 吞吐外推: 400 ep/seed = {400*dt/len(vals)/60:.1f} min; "
              f"18 seed 串行 {18*400*dt/len(vals)/3600:.1f} h; 8 卡并行 {18*400*dt/len(vals)/3600/8:.1f} h")
    print(f"[ec_eval] -> {p}")

if __name__=='__main__': main()
