"""Finalize this BLOCKED Stage 0C preparation; never sends API requests."""
from pathlib import Path
import json,csv,hashlib,datetime,subprocess,sys,xml.etree.ElementTree as ET
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cp_disr.vlm_metrics import summarize
from cp_disr.vlm_provider import DashScopeProvider
F=ROOT/'experiments/part_0_validation/stage_0c'
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,obj):
 p=F/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def table(name,fields,rows):
 with (F/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def text(name,s):(F/name).write_text(s)
if (F/'request_execution.jsonl').exists():raise RuntimeError('Refuse to overwrite reports after requests')
assert not list((ROOT/'experiments/vlm_cache/dev').glob('*/COMPLETE')),'Real caches require reporting actual observations'
for stage in ['0a','0b']:
 assert (F/f'previous_stage_{stage}.json').read_bytes()==(ROOT/f'experiments/stage_status/stage_{stage}.json').read_bytes()
issues=['MUST_BIND: 24 real initial RGB dev scenes (D0/T_A/T_C x8), provenance, split and upload permission, typed object bindings, goals, grounded IDs, effects, initial facts, task-to-asset mapping','MUST_BIND: 3 authorized real independent dev fewshots with frozen expected JSON and provenance','MUST_BIND: protected server SDK environment credential (key_present=false); local file was not accessed','MUST_VERIFY: qwen3.8-max-0902 access, image/JSON support, endpoint/parameter acceptance and request IDs via first formal scene only after all gates pass']
required=['scene_id','task_id','case_index','split=dev','synthetic_unit_fixture=false','image_ref','image_sha256','provenance.source_dataset/version/capture_kind/initial_frame_verified/split_evidence_ref','permission.vlm_upload_allowed/evidence_ref','preprocessing','object_table[id,type,binding_status,visual_evidence_ref]','signed_goals','allowed_action_ids','allowed_proposition_ids','contracts_ref/sha256','predicate_version','initial_facts(TRUE/FALSE/UNKNOWN)','task_definition_ref/sha256','asset_binding_ref/sha256','asset binding reviewer/time/template/assets/controller_refs/object evidence']
prompt=ROOT/'experiments/sources/v2.1_interfaces/system_prompt.txt'
slots=[{'slot_id':f'{t}_dev_{i:02d}','task_id':t,'case_index':i,'status':'MUST_BIND','missing_fields':'; '.join(required)} for t in ['D0','T_A','T_C'] for i in range(8)]
dump('task_scene_manifest.json',{'status':'MUST_BIND','scenes':[],'required_count':24,'bound_count':0,'prompt_sha256':sha(prompt),'required_fields':required,'required_slots':slots,'note':'Slots are collection requirements, not scene observations. Discovery is bounded, not proof of global absence.'})
fs=[{'slot_id':f'fewshot_{i+1}','coverage':v,'status':'MUST_BIND','missing_fields':'; '.join(required+['split=independent_dev instead of dev','expected_json','selection_reason','selected_at','reviewer','prompt_version','independence_evidence_ref'])} for i,v in enumerate(['contract complete -> empty','scene-supported SOFT_SUPPORTS with legal effect','insufficient evidence / independent order -> empty'])]
dump('few_shot_manifest.json',{'status':'MUST_BIND','examples':[],'required_count':3,'bound_count':0,'prompt_sha256':sha(prompt),'required_slots':fs,'no_prompt_revision':True})
table('scene_collection_checklist.csv',list(slots[0]),slots);table('few_shot_missing_manifest.csv',list(fs[0]),fs)
dump('sanity_cases.json',{'status':'BLOCKED','cases':[],'required_slots':slots})
env=DashScopeProvider().environment();assert env['key_present'] is False
env.update({'checked_at':now,'credentials_read':False,'API_requests':0,'sdk_transport':'One audited send; SDK hidden ConnectionError retry disabled; redirects disabled; SDK logs suppressed inside call','note':'SDK signature/dispatch validated offline, actual remote service unverified'})
dump('provider_environment.json',env)
metrics=summarize([]);metrics.update({'status':'BLOCKED','bound_real_scenes':0,'bound_fewshots':0,'task_wise':{t:summarize([],planned=8) for t in ['D0','T_A','T_C']},'scene_wise':[],'plot_status':'NOT_GENERATED_NO_OBSERVATIONS'})
dump('stage_0c_metrics.json',metrics)
mr=[]
for k,v in metrics.items():
 if isinstance(v,dict) and 'denominator' in v:mr.append({'metric':k,'value':'N/A' if v['value'] is None else v['value'],'numerator':v['numerator'],'denominator':v['denominator']})
 elif not isinstance(v,(list,dict)):mr.append({'metric':k,'value':'N/A' if v is None else v,'numerator':'','denominator':''})
table('stage_0c_metrics.csv',['metric','value','numerator','denominator'],mr)
scene_rows=[{'slot_id':s['slot_id'],'task_id':s['task_id'],'scene_id':'','status':'NOT_RUN_MISSING_INPUT','requests':0,'retries':0,'final_relation_count':'N/A','empty_prior':'N/A'} for s in slots]
for name in ['scene_results.csv','cache_sanity.csv']:table(name,list(scene_rows[0]),scene_rows)
for name in ['relation_audit.csv','manual_semantic_review.csv','semantic_audit.csv']:table(name,['scene_id','relation_id','label','card_label','reviewer','reviewed_at','rule_version','notes'],[])
table('cache_index.csv',['scene_id','cache_key','cache_path','status','manifest_sha256'],[])
table('cache_summary.csv',['planned_inputs','bound_inputs','observed_inputs','successful_requests','failed_requests','first_json_valid_n','first_json_valid_rate','raw_relation_n','id_valid_n','id_valid_rate','effect_valid_n','effect_valid_rate','redundant_n','redundant_rate','empty_n','empty_rate','SOFT_SUPPORTS','SOFT_RELEVANT_TO_GOAL','semantic_issue_n'],[dict(zip(['planned_inputs','bound_inputs','observed_inputs','successful_requests','failed_requests','first_json_valid_n','first_json_valid_rate','raw_relation_n','id_valid_n','id_valid_rate','effect_valid_n','effect_valid_rate','redundant_n','redundant_rate','empty_n','empty_rate','SOFT_SUPPORTS','SOFT_RELEVANT_TO_GOAL','semantic_issue_n'],[24,0,0,0,0,0,'N/A',0,0,'N/A',0,'N/A',0,'N/A',0,'N/A',0,0,'N/A']))])
dump('cache_manifest.json',{'status':'BLOCKED','entries':[],'requests':0,'failure_is_empty_prior':False,'formal_cache_root':'experiments/vlm_cache/dev'})
text('manual_review_protocol.md','No observed relations; no human review performed. Reviewer/time are intentionally unset. Future review labels: clearly_scene_supported (CORRECT), plausible_but_uncertain (UNDECIDABLE), obvious_semantic_issue (OBVIOUS_ISSUE), contract_redundant (OBVIOUS_ISSUE), unverifiable_from_allowed_input (UNDECIDABLE). Rule: frozen Stage 0C card + current user request. Review all 24 scenes, including empty outputs; never edit cache or trigger semantic retries.\n')
text('provider_validation.md','Production adapter and frozen configuration implemented; offline SDK dispatch validated with BaseApi transport double and sockets disabled. Same parser/cache code is used in tests and production. Fixed qwen3.8-max-0902, cn-beijing, https://dashscope.aliyuncs.com/api/v1, dashscope 1.27.6, JSON object, temperature 0, max_tokens 2048, thinking/search disabled. SDK retry helper patched under lock for a single send and redirects disabled. No credential lookup outside SDK. Secret-shaped strings/fields redacted. Actual remote API capability, model permission, parameter acceptance, usage and request ID remain MUST_VERIFY. SDK package was not changed. Offline mock is not a formal image or result.\n')
text('model_access_report.md','MUST_VERIFY. No API request was permitted because real scenes/fewshots and protected SDK environment are unbound. No API error code exists to report. Local key file was not read/copied/hashed/transferred. No model/endpoint substitution. Future first formal scene counts within 24; permission/model/endpoint rejection stops immediately without fallback.\n')
text('request_audit.md','Real requests: 0. Retries: 0. Request ledger absent. Formal caches: 0. No API secret accessed. Offline tests use mocked responses and blocked sockets, not API observations. Missing image inputs do not count as empty priors. No RL/PPO, robot/simulator actions, evaluation, Stage 1A or push executed.\n')
text('errors/blockers.md','\n'.join(issues)+'\n')
dump('attempts/attempt_index.json',{'001_offline':'32 pass / 1 failure in SDK mock','002_sdk_mock_fix':'32 pass / 1 failure; classmethod binding change did not fix missing role','003_sdk_response_role':'33 pass; root cause: mocked assistant Message omitted required role','004_final_validation':'40 pass: 34 Stage 0C + 6 existing pure regression tests; no PPO tests','formal_attempts':0})
code={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'src/cp_disr').glob('*.py'))}
junit=F/'attempts/004_final_validation/junit.xml';suite=ET.parse(junit).getroot().find('testsuite');assert suite is not None and suite.attrib['failures']=='0' and suite.attrib['errors']=='0' and suite.attrib['tests']=='40'
dump('offline_validation.json',{'status':'PASS','redaction_passed':True,'tests':40,'formal_api_evidence':False,'code_hashes':code,'test_source_sha256':sha(ROOT/'tests/test_stage0c_pipeline.py'),'junit_ref':str(junit.relative_to(ROOT)),'junit_sha256':sha(junit),'validated_at':now})
source_paths=[p for base in ['experiments/sources','schemas','configs/tasks','configs/contracts','tests/fixtures'] for p in (ROOT/base).rglob('*') if p.is_file()]+[ROOT/'experiments/stage_status/stage_0a.json',ROOT/'experiments/stage_status/stage_0b.json',ROOT/'experiments/docs/CP_DISR_v2.1_Minimal_Experimental_Execution_Plan.md',ROOT/'experiments/stage_cards/stage_0c.md']
sources={str(p.relative_to(ROOT)):sha(p) for p in sorted(source_paths)};dump('source_hashes.json',sources)
m_path=ROOT/'experiments/manifests/vlm_manifest.yaml';m=yaml.safe_load(m_path.read_text());m.update({'provider_implementation':'src/cp_disr/vlm_provider.py','provider_source_sha256':code['src/cp_disr/vlm_provider.py'],'cache_pipeline':'src/cp_disr/vlm_cache_pipeline.py','stage_0c_runner':'src/cp_disr/stage0c.py','task_scene_manifest':'../part_0_validation/stage_0c/task_scene_manifest.json','fewshot_manifest':'../part_0_validation/stage_0c/few_shot_manifest.json','fewshot_binding_status':'MUST_BIND','requests_executed':0,'provider_validation_status':'OFFLINE_PASS_SERVICE_MUST_VERIFY'});m_path.write_text(yaml.safe_dump(m,sort_keys=False,allow_unicode=True))
e_path=ROOT/'experiments/manifests/entrypoints.yaml';e=yaml.safe_load(e_path.read_text())
for entry in e['commands'].values():entry['source_sha256']=code['src/cp_disr/cli.py']
e['commands']['generate-cache'].update({'status':'IMPLEMENTED_FORMAL_INPUTS_UNBOUND_GATE_BLOCKED','production_runner':'cp_disr.stage0c.run','offline_validation_ref':'../part_0_validation/stage_0c/offline_validation.json'})
e['note']='Stage 0B PASS (existing evidence unchanged). Stage 0C production preparation offline validated; formal inputs/credential unbound; service MUST_VERIFY. Runtime unbound.';e_path.write_text(yaml.safe_dump(e,sort_keys=False,allow_unicode=True))
runtime=yaml.safe_load((ROOT/'experiments/manifests/runtime_manifest.yaml').read_text());missing={k:v for k,v in runtime['runtime'].items() if isinstance(v,str) and v.startswith('MUST_')};dump('stage_1a_missing_runtime.json',{'readiness':False,'runtime':missing,'runtime_factory':runtime.get('runtime_factory'),'resource_settings':runtime.get('resource_settings'),'vlm_cache':'Stage 0C BLOCKED'})
summary='''# Stage 0C — BLOCKED

本轮完成生产代码与离线验证，未进入真实 VLM 调用。正式输入 0/24，独立真实 few-shot 0/3；SDK 环境 key_present=false。不得将本报告解释为已完成 24 场景 sanity 或模型可用性证明。

## 17 项答复

1. 最终状态：BLOCKED（资源/输入绑定不足）。
2. D0/T_A/T_C 各 8 个真实 dev 场景尚未绑定；24 行清单是待采集槽位，不是已观察场景。
3. 三个 few-shot 未冻结。保留 v2.1 的空 / SOFT_SUPPORTS / 空语义要求，缺真实图像、权限与独立性证据。
4. 模型权限未真实验证，仍为 MUST_VERIFY。
5. 配置为 qwen3.8-max-0902、cn-beijing、https://dashscope.aliyuncs.com/api/v1、dashscope 1.27.6、JSON object、temperature=0、2048 tokens、非思考。SDK 离线分发通过；无实际服务响应模式可报告。
6. 未替换模型、地域、endpoint、schema 或冻结 prompt。
7. 真实请求 0，重试 0；离线 transport double 不计入请求。
8. JSON/schema/ID/effect 合法率均 N/A，观察分母均为 0。
9. 空 prior 率 N/A；缺失输入/API 失败不会算 EMPTY_PRIOR。
10. 观察到的合法关系数为 0（未执行）；平均数 N/A。
11. 两类关系观察数均为 0（未执行），不能据此断言模型输出为空。
12. 合同冗余观察数 0；明显语义错误数 N/A，未进行人工语义审阅。
13. 无实际输出可判断是否产生越权内容；离线测试验证拒绝禁用关系、新 ID、直接动作答案等。
14. 尚无正式 raw/parsed/rejected/cache。生产审计链通过离线写入、校验、不可覆盖及篡改拒绝测试；正式服务可追溯性尚待验证。
15. 新增冻结 provider、请求预检/输入门禁、生产解析与 schema/ID/type/effect/冗余/去重/冲突隔离、不可覆盖缓存、统计；扩展 key 覆盖初始事实/资产绑定/实际 payload；接通 generate-cache；限制 SDK 隐藏重试及跳转；保留所有失败测试日志。
16. 不具备 Stage 1A 的 VLM/cache 条件；未启动 Stage 1A。
17. Stage 1A 仍缺平台/版本、真实任务资产和 split、控制器、相机/标定、感知 checkpoint、verifier 阈值、超时/deadline/实际计时及参考技能秒数、评价器、安全授权、可信 runtime factory、资源设置；完整字段见 stage_1a_missing_runtime.json。

## 验证与限制

最终 40 项通过：34 项 Stage 0C 离线测试 + 6 项既有纯逻辑回归；没有运行 PPO 测试或任何训练。早期两次离线测试各有一项失败，根因为 mock SDK response 缺少 role；全部日志保留。第三次 33 项通过，最终加固后 40 项通过。SDK deprecated Assistants import warning 不表示本轮使用 Assistants API。

只读搜索覆盖 13 个受限根目录，见 input_discovery.json；深度/时限/文件数限制已记录，不能推断服务器全局没有图像。找到的 logo、纹理与历史任务 review 图像未被认证为本次 D0/T_A/T_C 初始 dev RGB；历史 split 属于其他任务，代码 MIT 许可证不足以证明图片可上传。没有使用 synthetic 图像补齐。

图表未生成：NOT_GENERATED_NO_OBSERVATIONS。没有把 24 个未运行槽位画成零关系柱。语义审阅者与时间保持空，规则见 manual_review_protocol.md。

冻结 Method/schema/prompt/task templates/contracts 不变；Stage 0A BLOCKED 与 Stage 0B PASS 既有证据不变。仅更新 Stage 0C 状态与生产入口相关 manifest；git_hash 字段记录起始 commit，最终提交通过外部 delivery receipt 记录，避免自引用 hash。

## 未绑定项与停止点

'''+ '\n'.join('- '+i for i in issues)+ '\n\n只完成本阶段可执行的准备和验证，正式运行等待真实资源。未推送 GitHub、未改旧仓库、未执行动作/训练/性能评价；交付后关闭 SSH 并停止。\n'
text('stage_0c_summary.md',summary);text('cache_sanity_report.md',summary)
status=json.loads((F/'previous_stage_0c.json').read_text());status.update({'status':'BLOCKED','git_hash':'502d33cbca9a83eccd1c32840a0d0b50189c3ad1','git_hash_semantics':'starting commit; final delivery commit in external receipt','started_at':datetime.datetime.fromtimestamp((F/'previous_stage_0c.json').stat().st_mtime,datetime.timezone.utc).isoformat(),'completed_at':now,'runs_completed':['offline validation: 40 passed'],'runs_failed':['offline attempt 001: one SDK mock failure','offline attempt 002: one SDK mock failure'],'runs_reused':[],'selected_tasks':['D0','T_A','T_C'],'selected_config':'experiments/manifests/vlm_manifest.yaml','issues':issues,'tuning_changes':[],'execution_scope':'Stage 0C production preparation and offline validation only; real requests blocked','readiness':{'Stage_1A':False,'formal_scenes':0,'fewshots':0,'API_requests':0,'key_present':False,'model_access':'MUST_VERIFY'},'source_hashes':sources,'report':'experiments/part_0_validation/stage_0c/stage_0c_summary.md'})
(ROOT/'experiments/stage_status/stage_0c.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
print('Stage 0C BLOCKED report generated; no API calls.')
