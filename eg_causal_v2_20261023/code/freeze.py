import json, hashlib, shutil
from pathlib import Path
W=Path(__file__).parent
O=W.parents[1]/'outputs/eg_causal_v2_20261023'
O.mkdir(parents=True,exist_ok=True)
p=json.loads((W/'protocol_design.json').read_text(encoding='utf-8-sig'))
p['status']='stage1_execution_frozen'
p['execution_authorization']={'source':'User requested execution of v2 plan in this turn','server_execution':'stage1_only_no_parameter_update','parameter_updates':False,'git_write':False,'C4_C5':False}
p['stage1_scope']={'train_scenes':[2,9,31,33,68,64,76,82],'train_frames':[0,128],'R1_trace':'all original 634 q_ref=0 valid fixed inference queries plus positive controls','R1_training_path':{'batches':16,'batch_size':1,'seed_rule':'20261023 + scene*1000 + frame','net_is_training':True,'view_is_training':True,'torch_train':True,'augmentation':'official dataset augment=True','repeat_control':'first batch RNG/state replay, duplicate view writes stratified','paired_eval_control':'same captured grouping inputs, directions and labels; cy/head eval only','optimizer':None,'restore_buffers_each_batch':True},'R2':{'k':[0,1,3,6],'frames':16,'queries':256,'width':'original frozen decoded width','new_analytic_evaluations':0,'identity_control':True,'pilot_frames':1},'R3':{'queries':256,'all_48_grid_cells':12288,'unit':'scene','random4':'exact hypergeometric expectation plus seeded reference if needed','new_analytic_evaluations':0},'new_C3_forward':False,'new_physical_evaluation':False}
p['next_stage_gate']='R1.4 temporary parameter updates need explicit additional authorization after stage1 results and concrete reviewable pulse design; R1.5 remains gated by pulse evidence and authorization.'
p['input_hashes']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [W/'protocol_design.json',W/'plan_v2.md']}
(O/'protocol_v2_frozen.json').write_text(json.dumps(p,ensure_ascii=False,indent=2),encoding='utf-8')
shutil.copy2(W/'plan_v2.md',O/'plan_v2.md')
print(O)
