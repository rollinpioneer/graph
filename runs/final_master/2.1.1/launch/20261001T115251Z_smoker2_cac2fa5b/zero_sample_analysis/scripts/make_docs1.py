import json,os,glob,time,hashlib
OUT=open('/home/xushijie2/tmp/r2/R2_OUT.txt').read().strip(); ANA=OUT+'/zero_sample_analysis'
REPO='/home/xushijie2/graph_cp_disr_final_tb_launch'; R=REPO+'/runs/final_master/2.1.1/T_B/'
rel=lambda p: os.path.relpath(p,REPO)
man={}
for l in open(OUT+'/artifact_manifest.jsonl'):
    j=json.loads(l); man[j['path']]=j
cmpj={r['plan']:r for r in json.load(open(ANA+'/comparison.json'))}
wid=json.load(open(ANA+'/weights_identity.json')); cid=json.load(open(ANA+'/checkpoint_identity.json')); ffr=json.load(open(ANA+'/final_vs_fragment.json')); integ=json.load(open(ANA+'/integrity_lowdisk_check.json'))
dirs={'R-TB-E-0':glob.glob(R+'B1-K+E/seed_0/R-TB-E-0-*')[0],'R-TB-DK-1':glob.glob(R+'B2/seed_1/R-TB-DK-1-*')[0],'R-TB-K-1':glob.glob(R+'B1-K/seed_1/R-TB-K-1-*')[0]}
# --- code/split hashes across all 12 eval files
ident={}
for p,d in dirs.items():
    for f in sorted(glob.glob(d+'/eval_n_*.json'))+[d+'/eval_final.json']:
        e=json.load(open(f)); h=e['hashes']
        ident[(p,os.path.basename(f))]={k:h[k] for k in h if k not in('plan_id','attempt_id')}
vals={json.dumps(v,sort_keys=True) for v in ident.values()}
same_hashes=len(vals)==1
# --- DK-1 repeated eval
dk=cmpj['R-TB-DK-1']['eval_points']
pc=lambda x:[(c['case_id'],c['success'],c['G'],c['steps'],c['reason'],c['success_seconds']) for c in x['per_case']]
eq_4_8=pc(dk[1])==pc(dk[2]); eq_8_f=pc(dk[2])==pc(dk[3])
w=wid['R-TB-DK-1']['pairwise_model_weight_diff']
def wd(a,b):
    for x in w:
        if {x['a'],x['b']}=={a,b}: return x
fin='final_n_014170_u13.pt'
dkj={'run':'R-TB-DK-1','eval_points':[{'file':x['file'],'file_sha256':x['file_sha256'],'checkpoint':x['checkpoint'],'checkpoint_sha256':x['checkpoint_sha256'],'success_n':x['success_n'],'mean_discounted_return':x['mean_discounted_return'],'mean_success_seconds_field':x['mean_success_seconds']} for x in dk],
 'per_case_rows_identical_4096_vs_8192':eq_4_8,'per_case_rows_identical_8192_vs_final':eq_8_f,
 'checkpoint_sha256_all_distinct':len({x['checkpoint_sha256'] for x in dk})==len(dk),
 'model_weight_diffs':[{'pair':[x['a'],x['b']],'tensors_differing':x['tensors_differing'],'of':x['tensors_compared'],'max_abs_diff':x['max_abs_diff']} for x in w],
 'eval_code_and_split_hashes_identical_across_all_12_eval_files':same_hashes,
 'per_case_action_sequences_recorded':False,'identical_trajectories':'UNVERIFIED'}
json.dump(dkj,open(ANA+'/dk1_repeated_eval_check.json','w'),indent=1)
md=[]
md.append('# DK-1 repeated evaluation return — checkpoint identity and per-case records (no new episode)\n')
md.append('Observation (already in the registered evals): R-TB-DK-1 reports mean discounted return **0.4988490585995873**, success 10/10 and recorded `success_seconds` mean 3.54 at all three post-update evaluation points (N = 4096, 8192, final 14170). This file checks only what the existing records can establish.\n')
md.append('## Checked and found\n')
md.append('| item | result |\n|---|---|')
md.append('| evaluated checkpoints are different files | sha256 all distinct: `%s` |'%', '.join('%s %s'%(x['checkpoint'],x['checkpoint_sha256'][:12]) for x in dk))
md.append('| evaluated model weights differ between the points | n_004096 vs n_008192: %d of %d float tensors differ, max abs diff %.3e; n_008192 vs final: %d of %d, %.3e; n_004096 vs final: %d of %d, %.3e |'%(wd('n_004096.pt','n_008192.pt')['tensors_differing'],wd('n_004096.pt','n_008192.pt')['tensors_compared'],wd('n_004096.pt','n_008192.pt')['max_abs_diff'],wd('n_008192.pt',fin)['tensors_differing'],wd('n_008192.pt',fin)['tensors_compared'],wd('n_008192.pt',fin)['max_abs_diff'],wd('n_004096.pt',fin)['tensors_differing'],wd('n_004096.pt',fin)['tensors_compared'],wd('n_004096.pt',fin)['max_abs_diff']))
md.append('| each eval file names the checkpoint it evaluated | `eval_*.json` field `checkpoint` = the respective `.pt`; final eval checkpoint path == final_n_014170_u13.pt (realpath equal) |')
md.append('| code / split identity of the evals | the `hashes` block (final_tb, stage2a_v11, collector, torch_rl, neural, persistence, runtime_factory, manifests, dev10 split b16130d2…, git_commit cac2fa5b3) is %s across all 12 eval files of the three runs |'%('identical' if same_hashes else 'NOT identical'))
md.append('| per-case rows across the three points | success, G (full float precision), steps, reason, success_seconds are %s for 4096 vs 8192 and %s for 8192 vs final (10 cases each) |'%('identical' if eq_4_8 else 'NOT identical','identical' if eq_8_f else 'NOT identical'))
md.append('| final checkpoint = post-last-update model | final weights and Adam state equal `update_fragment_1.pt` (max diff 0.0, optimizer state equal) and differ from `update_complete_13.pt` (max abs diff %.3e, 66 tensors) |'%ffr['R-TB-DK-1']['final_vs_update_complete_13'][0])
md.append('\n## Not established (UNVERIFIED)\n')
md.append('- Whether the policy takes the same action sequence on each dev case at the three checkpoints: the eval rows hold only `case_id, success, G, steps, reason, success_seconds, source_n, prior_mode`; no selected-candidate sequence, logits or per-step trace is stored for evaluation episodes, and no eval decision log exists in the run directory. Identical G, step count (5) and final-skill duration are consistent with identical trajectories but do not prove it. -> **UNVERIFIED**.')
md.append('- Why the greedy (deterministic argmax) policy gives identical returns at differing weights: not analysed; no mechanism claim. Recomputing it would need new evaluation episodes, which this card does not run.')
md.append('- Total elapsed time to success per case: not recorded (see caveat C1 in `comparison_tables.md`).')
md.append('\nConclusion allowed from these records: the three DK-1 post-update evaluations are three distinct evaluations of three distinct checkpoints that returned identical per-case numbers; this is not a duplicated or mis-bound evaluation file. It is not shown that the trajectories are identical.\n')
open(ANA+'/dk1_repeated_eval_check.md','w').write('\n'.join(md))
print('\n'.join(md)[:3000])
