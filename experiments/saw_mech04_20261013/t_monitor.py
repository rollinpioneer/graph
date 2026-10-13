"""Read-only T progress and actual first-pair update0 identity."""
import pathlib,json,time,datetime,pickle,hashlib
P=pathlib.Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def main():
 resources=[read(p) for p in (P/"receipts").glob("*_resource.json")];active=[read(p) for p in (P/"receipts").glob("*_active.json")]
 seconds=sum(r["residency_seconds"] for r in resources if r["group"]=="T")+sum(max(0,time.time()-r["start"]) for r in active if r["group"]=="T")
 print(datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%SZ"),"T_GPUh",round(seconds/3600,4))
 for arm in range(4):
  status=P/"results"/f"T{arm}_training_status.json";progress=P/"receipts"/f"T{arm}_progress.json"
  if status.exists():s=read(status);print(f"T{arm}",s["student_updates"],s["status"])
  elif progress.exists():s=read(progress);print(f"T{arm}",s["step"],"RUNNING")
  elif any(x["label"]==f"T{arm}" for x in active):print(f"T{arm}",0,"INITIALIZING")
  else:print(f"T{arm}","NOT_STARTED")
 gate=P/"results/T_PRETRAIN_BEHAVIOR_GATE.json"
 if gate.exists():print("BASELINE_GATE",read(gate)["status"])
 episode=P/"receipts/T_EPISODE_BUDGET.json"
 if episode.exists():x=read(episode);print("T_EVAL_EPISODES",x["used"],"/",x["limit"])
 first=P/"receipts/T_ACTUAL_FIRST_PAIR_UPDATE0.json"
 pair=[P/"results"/f"T{a}_checkpoints.json" for a in (0,1)]
 if not first.exists() and all(p.exists() for p in pair):
  hashes=[]
  for p in pair:
   c=next(x for x in read(p) if x["step"]==0)
   with open(c["path"],"rb") as f:s=pickle.load(f)
   hashes.append(hashlib.sha256(pickle.dumps((s["agent"],s["actor_opt_state"],s["actor_rng"],s["numpy_rng"],s["python_rng"],s["stream_cursor"],s["stream_manifest_sha256"]),4)).hexdigest())
  first.write_text(json.dumps(dict(status="PASS" if len(set(hashes))==1 else "FAILED",hashes=hashes),indent=2)+"\n")
 if first.exists():print("ACTUAL_FIRST_PAIR_UPDATE0",read(first)["status"])
 identity=P/"receipts/T_ACTUAL_UPDATE0.json"
 if identity.exists():print("ACTUAL_FOUR_UPDATE0",read(identity)["status"])
 end=P/"receipts/T_EVALUATION_EXIT.json";train=P/"receipts/T_SCHEDULER_EXIT.json"
 if end.exists() and train.exists() and not active:print("T_ALL_EXITED")
 else:print("T_ACTIVE",len(active))
if __name__=="__main__":main()
