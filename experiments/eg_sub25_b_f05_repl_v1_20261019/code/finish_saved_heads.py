"""Validate existing step1000 heads and finish receipts; no optimizer or updates."""
from repl_common import *
import ast,resource
source=(R/'code/fit_heads.py').read_text().split('parser=argparse.ArgumentParser()')[0]
source=source.replace("device='cuda';torch.cuda.reset_peak_memory_stats()","device='cpu'")
assert "device='cuda'" not in source
exec(compile(source,'saved_head_validation_prefix','exec'),globals())
records=[];hashes={}
for seed in SEEDS:
 path=R/f'B/heads/{seed}/step1000.pt';before=sha256(path);ck=torch.load(path,map_location='cpu',weights_only=False)
 assert ck['seed']==seed and ck['step']==1000
 assert np.array_equal(ck['mean'],mean) and np.array_equal(ck['std'],std)
 assert all(torch.isfinite(v).all() for v in ck['model'].values())
 scorer=FrozenScorer(path,before);model=scorer.model
 q=scorer.quality(val['feature'][:1024],val['action'][:1024,0]);reference=predict(model,xv[:1024])[:,0]
 assert np.allclose(q,reference,atol=1e-6,rtol=1e-6)
 vm=metrics(predict(model,xv),val);assert abs(vm['scene_equal_mse']-ck['metrics']['scene_equal_mse'])<1e-6
 history=[]
 for line in (R/f'logs/fit_head_{seed}.log').read_text().splitlines():
  if line.startswith(f'HEAD {seed} '):history.append(ast.literal_eval(line.split(' ',2)[2]))
 assert [x['step'] for x in history]==list(range(100,1001,100))
 rng=np.random.default_rng(seed);order=rng.permutation(len(xf));cursor=0;stream=hashlib.sha256()
 for step in range(1000):
  if cursor+128>len(order):order=rng.permutation(len(xf));cursor=0
  ix=order[cursor:cursor+128];cursor+=128;stream.update(ix.tobytes())
 assert sha256(path)==before;hashes[str(seed)]=before
 records.append(dict(seed=seed,path=str(path),sha256=before,step=1000,fit_groups=len(xf),model_val_groups=len(xv),history=history,sample_stream_sha256=stream.hexdigest(),gate_diagnostics=metrics(predict(model,xg),gate),normalization_from='B fit samples only',reload_scoring_pass=True,verification_candidates=1024,validation_mse=vm['scene_equal_mse'],additional_training_updates=0,full_per_step_losses_unavailable=True,loss_record='Original log preserves every100-step diagnostics; failed writer did not persist its in-memory1000 losses'))
ledger=[json.loads(x) for x in (R/'results/resource_jobs.jsonl').read_text().splitlines()]
write(R/'results/B_heads.json',dict(status='PASS',heads=records,actions=196608,GPU_process_seconds=sum(x['seconds'] for x in ledger if x['name'].startswith('fit_head_')),model_selection='fixed step1000',post_save_validation_recovery=True,additional_training_updates=0))
write(R/'results/head_validation_recovery/recovery_validation.json',dict(status='PASS',seconds=time.time()-start,checkpoint_hashes_unchanged=hashes,head_count=3,additional_training_updates=0,CPU_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scorer_interface_candidates=1024,root_cause='Post-save validation caller used16 candidates; fixed scorer requires1024. Existing heads already completed1000 updates.'))
print('SAVED_HEADS_VALIDATED_NO_RETRAINING',flush=True)
