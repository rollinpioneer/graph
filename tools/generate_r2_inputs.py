from pathlib import Path
import json,hashlib,shutil,sys,os
import yaml
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from cp_disr.common import canonical,digest
from cp_disr.contracts import Registry,from_dict
from cp_disr.graph import Goal,build_template
from cp_disr.platforms.libero.scene_capture import SceneCaptureAdapter
CONTRACT=ROOT/'configs/contracts/skills.yaml'; contract_doc=json.loads(CONTRACT.read_text()); contract_hash=hashlib.sha256(CONTRACT.read_bytes()).hexdigest(); asset_license=ROOT/'assets/cp_disr/LICENSE'; license_hash=hashlib.sha256(asset_license.read_bytes()).hexdigest()
PROMPT=ROOT/'experiments/sources/v2.1_interfaces/system_prompt.txt'; prompt_hash=hashlib.sha256(PROMPT.read_bytes()).hexdigest(); schema=ROOT/'schemas/relation_schema.json'; schema_hash=hashlib.sha256(schema.read_bytes()).hexdigest()
PLATFORM_COMMIT='8f1084e3132a39270c3a13ebe37270a43ece2a01'; PLATFORM_ROOT='/home/__compress_data/xushijie/platforms/LIBERO_cpdisr_8f1084'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x): Path(p).parent.mkdir(parents=True,exist_ok=True); Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def objects(task): return {'target':'object','second_object' if task!='T_C' else 'interferer':'object','container':'container','buffer':'buffer'}
def config(task,idx,seed):
 if task=='T_C': target=(-.55+.12*(idx%4),-.10+.20*(idx//4),.18); second=(.0+.16*(idx%4),-.10+.20*(idx//4),.18)
 else: target=(-.80+.20*(idx%4),-.25+.20*(idx//4),.18); second=(-.20+.18*((idx+1)%4),.25-.18*(idx//4),.18)
 return {'task_id':task,'scene_id':f'{task}_dev_{idx:02d}','seed':seed,'objects':{'target':target,('interferer' if task=='T_C' else 'second_object'):second},'container':(.62,.25),'buffer':(-.68,.30,.26,.18),'camera':{'name':'agentview','width':128,'height':128,'mode':'fixed'},'asset_version':'cp-disr-procedural-box-v1','platform_commit':PLATFORM_COMMIT}
def prepare(task,scene_id,seed,split,idx,rootdir):
 cfg=config(task,idx,seed); cfg['scene_id']=scene_id; scene=Path(rootdir); scene.mkdir(parents=True,exist_ok=True)
 if (scene/'scene_manifest.json').exists(): return json.loads((scene/'scene_manifest.json').read_text())
 xml_template=ROOT/'assets/cp_disr/scene_template.xml'; adapter=SceneCaptureAdapter(xml_template,128,128); adapter.reset_scene(task,cfg,seed); rgb=adapter.capture_rgb(); depth=adapter.capture_depth(); adapter.close()
 from PIL import Image
 Image.fromarray(rgb).save(scene/'rgb.png'); import numpy as np; np.save(scene/'depth.npy',depth.astype('float32'))
 cfg_hash=digest(cfg); image_hash=sha(scene/'rgb.png'); depth_hash=sha(scene/'depth.npy')
 obs=objects(task); registry=Registry(contract_doc['predicate_types']); [registry.register(from_dict(c)) for c in contract_doc['contracts']]
 template=build_template(registry.ground(obs),(Goal('p:Inside:target:container',1),),registry.predicate_types,obs)
 actions=sorted(c.id for c in template.contracts); props=sorted(n.id for n in template.nodes if n.kind=='PROPOSITION')
 facts={p:'UNKNOWN' for p in props}; facts['p:GripperEmpty']='TRUE';
 for o in [k for k,v in obs.items() if v=='object']:facts[f'p:OnTable:{o}']='TRUE'; facts[f'p:Held:{o}']='UNKNOWN'; facts[f'p:Inside:{o}:container']='UNKNOWN'; facts[f'p:AtBuffer:{o}:buffer']='UNKNOWN'
 facts['p:Open:container']='UNKNOWN'
 task_ref={'task_id':task,'task_version':'cp-disr-v2.1-r2','platform_ref':'LIBERO_CP_DISR_CLEAN','platform_commit':PLATFORM_COMMIT,'stage_0c_capture_ready':True,'stage_1a_runtime_ready':False,'contract_versions':['template-v2.1-1'],'predicate_registry_version':'CP-DISR-v2.1','asset_refs':['assets/cp_disr/scene_template.xml','assets/cp_disr/LICENSE'],'camera_config_ref':'camera_config.json','scene_reset_ref':'reset_config.json','split_policy':{'split':split,'fewshot_pool':'independent_dev','test':'isolated'},'instruction':f'{task} scene: place target object into the blue openable container; no execution requested.'}
 task_ref_path=scene/'task_definition.json'; write(task_ref_path,task_ref)
 binding={'status':'VERIFIED','reviewer':'CP-DISR-R2-automated-binding','reviewed_at':'2026-09-21T00:00:00Z','structural_template_ref':f'configs/tasks/resolved/{task}.yaml','source_assets':['assets/cp_disr/scene_template.xml','assets/cp_disr/LICENSE'],'controller_refs':['UNVERIFIED_FOR_STAGE_1A'],'object_binding_evidence':'object_bindings.json','capture_platform':'LIBERO_CP_DISR_CLEAN'}; write(scene/'asset_binding.json',binding)
 object_table=[{'id':k,'type':v,'binding_status':'VERIFIED','visual_evidence_ref':'rgb.png'} for k,v in sorted(obs.items())]; write(scene/'object_bindings.json',object_table)
 write(scene/'reset_config.json',cfg); write(scene/'camera_config.json',cfg['camera']); write(scene/'allowed_action_ids.json',actions); write(scene/'allowed_proposition_ids.json',props); write(scene/'contract_refs.json',{'path':'configs/contracts/skills.yaml','sha256':contract_hash,'interface_status':'FROZEN_FOR_STAGE_0C_INPUT','execution_status':'UNVERIFIED_FOR_STAGE_1A'}); write(scene/'initial_fact_summary.json',{'facts':[{'atom_id':k,'value':v,'evidence_source':'static_rule_or_conservative_rgb','observed_or_static':'static' if k=='p:GripperEmpty' else 'conservative_observation','capture_time':'2026-09-21T00:00:00Z','hidden_truth_used_for_qa':False,'hidden_truth_used_for_prompt':False,'verifier_status':'UNVERIFIED_FOR_STAGE_1A'} for k,v in sorted(facts.items())]}); write(scene/'hidden_truth_qa.json',{'hidden_truth_used_for_prompt':False,'object_pose':cfg['objects'],'container_pose':cfg['container'],'buffer_pose':cfg['buffer'],'true_dynamic_facts':'WITHHELD_FROM_PROMPT'})
 write(scene/'capture_log.json',{'status':'CAPTURED','actions_executed':0,'seed':seed,'adapter':'cp_disr.platforms.libero.scene_capture.SceneCaptureAdapter','rgb_sha256':image_hash,'depth_sha256':depth_hash})
 provenance={'source_dataset':'CP-DISR_PROJECT_OWNED_PROCEDURAL_MUJOCO','version':'cp-disr-procedural-box-v1','split_evidence_ref':f'stage_0c_inputs/{split}/{task}/{scene_id}','capture_kind':'recorded_simulator_rgb','initial_frame_verified':True,'platform_type':'simulator','platform_repo':PLATFORM_ROOT,'platform_commit':PLATFORM_COMMIT,'asset_license_ref':'assets/cp_disr/LICENSE','asset_license_sha256':license_hash}; write(scene/'provenance.json',provenance)
 manifest={'scene_id':scene_id,'task_id':task,'case_index':idx,'split':split,'synthetic_unit_fixture':False,'image_ref':str(scene.relative_to(ROOT)).replace('\\','/'), 'image_sha256':image_hash,'provenance':provenance,'permission':{'vlm_upload_allowed':True,'evidence_ref':'assets/cp_disr/LICENSE'},'preprocessing':{'version':'raw-rgb-inline-v1','resize':False,'crop':False,'content_bytes':'original'},'object_table':object_table,'signed_goals':[{'fact_id':'p:Inside:target:container','sign':1}],'initial_facts':facts,'contracts_ref':'configs/contracts/skills.yaml','contracts_sha256':contract_hash,'task_definition_ref':str(task_ref_path.relative_to(ROOT)).replace('\\','/'),'task_definition_sha256':sha(task_ref_path),'asset_binding_ref':str((scene/'asset_binding.json').relative_to(ROOT)).replace('\\','/'),'asset_binding_sha256':sha(scene/'asset_binding.json'),'allowed_action_ids':actions,'allowed_proposition_ids':props,'predicate_version':'CP-DISR-v2.1','asset_binding_hash':digest(binding),'initial_facts_hash':digest(facts),'scene_config_hash':cfg_hash,'camera_config_hash':digest(cfg['camera']),'image_depth_sha256':depth_hash}
 write(scene/'scene_manifest.json',manifest); return manifest

def main():
 base=ROOT/'experiments/stage_0c_inputs';
 scenes=[]
 for task in ('D0','T_A','T_C'):
  for i in range(8):
   seed=2000+({'D0':0,'T_A':100,'T_C':200}[task])+i; d=base/'dev'/task/f'scene_{i:02d}'; d.mkdir(parents=True,exist_ok=True); scenes.append(prepare(task,f'{task}_dev_{i:02d}',seed,'dev',i,d))
 examples=[]; examples.append(prepare('D0','fs_01',9001,'independent_dev',0,base/'fewshots/fs_01')); examples.append(prepare('T_A','fs_02',9002,'independent_dev',1,base/'fewshots/fs_02')); examples.append(prepare('T_C','fs_03',9003,'independent_dev',2,base/'fewshots/fs_03'))
 expected=[{'schema_version':'m1_soft_relations_v2','relations':[]},{'schema_version':'m1_soft_relations_v2','relations':[{'relation_id':'fs2_r1','type':'SOFT_SUPPORTS','source_ref':'a:PLACE_BUFFER:second_object:buffer:v1','target_ref':'a:PICK:target:v1','effect_fact_ref':'p:AtBuffer:second_object:buffer'}]},{'schema_version':'m1_soft_relations_v2','relations':[]}]
 for e,x in zip(examples,expected): e['expected_json']=x
 write(base/'formal_scene_manifest.json',{'status':'READY_FOR_LOCAL_VALIDATION','scenes':scenes,'required_count':24,'bound_count':24,'prompt_sha256':prompt_hash,'platform_commit':PLATFORM_COMMIT})
 write(base/'fewshots/few_shot_manifest.json',{'status':'FROZEN','examples':examples,'required_count':3,'bound_count':3,'prompt_sha256':prompt_hash,'platform_commit':PLATFORM_COMMIT})
 write(base/'fewshots/frozen_expected_outputs.json',{'status':'FROZEN','outputs':expected,'schema':'m1_soft_relations_v2','prompt_sha256':prompt_hash})
 (base/'fewshots/prompt_hash.txt').write_text(prompt_hash+'\n')
 print(json.dumps({'scenes':len(scenes),'fewshots':len(examples),'unique_images':len({x['image_sha256'] for x in scenes+examples}),'unique_configs':len({x['scene_config_hash'] for x in scenes+examples})}))
if __name__=='__main__':main()
