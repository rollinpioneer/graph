"""Publish evidence-based Stage 0A BLOCKED report; never execute another stage."""
import csv,hashlib,json,shutil,subprocess,tomllib
from datetime import datetime,timezone
from pathlib import Path
import yaml
from jsonschema import Draft202012Validator

R=Path('/home/__compress_data/xushijie/graph_cp_disr_v2_1')
E=R/'experiments'; O=E/'part_0_validation/stage_0a'; M=E/'manifests'
def jload(p): return json.loads(p.read_text())
def yload(p): return yaml.safe_load(p.read_text())
def jout(p,obj): p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def yout(p,obj): p.write_text(yaml.safe_dump(obj,allow_unicode=True,sort_keys=False))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*args): return subprocess.check_output(['git','-C',str(R),*args],text=True).strip()
now=datetime.now(timezone.utc).isoformat()
host=jload(O/'environment_probe.json'); sw=jload(O/'software_checks.json')
source=jload(O/'source_inventory.json'); review=jload(O/'legacy_resource_review.json')
old=jload(E/'stage_status/stage_0a.json')
base_hash=git('rev-parse','HEAD')
software_ok=sw['profile_validated'] and (R/'uv.lock').is_file() and (O/'host_logs/pip_check.exit').is_file() and (O/'host_logs/pip_check.exit').read_text().strip()=='0'
if not software_ok:
    # Preserve a separate exact blocker if the target environment did not finish installing.
    sw['installation_blocked']=True
runtime=yload(M/'runtime_manifest.yaml')
bound={
 'experiment_root':str(E),'repository_path':str(R),'repository_url':'git@github.com:rollinpioneer/graph.git',
 'git_hash':base_hash,'hardware':{'host':'gpu03','cpu_count':host['cpu_count'],'gpu_inventory':host['nvidia_smi']['stdout'].strip().splitlines(), 'reservation':'NOT_RESERVED_FOR_TRAINING'},
 'gpu_driver':'610.57.04','os':{'distribution':'Rocky Linux 10.2','kernel':host['kernel'],'architecture':host['architecture'],'container':'NOT_APPLICABLE_HOST_EXECUTION'},
}
runtime['runtime'].update(bound)
runtime['manifest_status']='PARTIALLY_BOUND_BLOCKED'
runtime['execution_host_identity']='xushijie@gpu03 (172.19.164.160)'
runtime['execution_scope']='Stage 0A infrastructure only; no robot, simulator rollout, training, VLM call or Stage 0B'
runtime['entrypoints_manifest_ref']='manifests/entrypoints.yaml'
runtime['resource_settings']={'num_envs':'MUST_BIND_AFTER_PLATFORM_BINDING','candidate_view_chunk':'MUST_BIND_AFTER_MODEL_IMPLEMENTATION','clock_unit_required':'seconds','actual_clock_source':'MUST_BIND'}
yout(M/'runtime_manifest.yaml',runtime)
software=yload(M/'software_manifest.yaml')
lock=tomllib.loads((R/'uv.lock').read_text()) if (R/'uv.lock').exists() else {}
actual_indexes=sorted({p.get('source',{}).get('registry') for p in lock.get('package',[]) if p.get('source',{}).get('registry')})
software.update(profile_status='VALIDATED' if software_ok else 'PROPOSED_NOT_VALIDATED',
 python_executable=str(R/'.venv/bin/python'),bootstrap_uv_executable=str(R/'.bootstrap/bin/uv'),
 lockfile='../../uv.lock' if (R/'uv.lock').exists() else 'MUST_GENERATE_ON_TARGET_HOST',
 lockfile_path_from_repository='uv.lock',lockfile_sha256=sha(R/'uv.lock') if (R/'uv.lock').exists() else None,
 wheel_indexes=actual_indexes,
 evidence='part_0_validation/stage_0a/software_checks.json',
 validation_scope='Library imports, minimal RGCN forward/backward and untrained state_dict round-trip; NOT CP-DISR method validation',
 torchvision_status='NOT_INSTALLED_OPTIONAL_NO_BOUND_VISUAL_FRONTEND',
 system_environment_modified=False)
yout(M/'software_manifest.yaml',software)
for path in [M/'experiment_manifest_v0.yaml',E/'configs/resolved_defaults.yaml']:
 obj=yload(path); obj['runtime'].update(bound); obj['state']='STAGE_0A_PARTIALLY_BOUND_BLOCKED'
 obj['software'].update(profile_status=software['profile_status'],lockfile='../../uv.lock' if (R/'uv.lock').exists() else 'MUST_GENERATE_ON_TARGET_HOST')
 yout(path,obj)
entry=yload(M/'entrypoints.yaml')
entry.update(status='PARTIALLY_BOUND_BLOCKED',repository=str(R),
 infrastructure_only={
  'host_probe':{'path':'tools/stage0a_host_probe.py','cli':'python3 tools/stage0a_host_probe.py','sha256':sha(R/'tools/stage0a_host_probe.py')},
  'software_probe':{'path':'tools/stage0a_software_probe.py','cli':'.venv/bin/python tools/stage0a_software_probe.py','sha256':sha(R/'tools/stage0a_software_probe.py')},
  'kit_plan':{'path':'experiments/tools/plan_stage.py','cli':'.venv/bin/python experiments/tools/plan_stage.py --stage 0A','sha256':sha(E/'tools/plan_stage.py')},
 },note='Only infrastructure paths are implemented. train/eval/tests/cache/report production CLIs remain MUST_BIND; no invented executable paths.')
yout(M/'entrypoints.yaml',entry)
tasks=yload(M/'task_manifest.yaml'); tasks['status']='UNBOUND_REAL_ASSETS_NOT_SUBSTITUTED_WITH_TEMPLATES'
tasks['required_for_stage_0a']=['D0','T_A','T_C']; tasks['optional_for_stage_0a']=['T_B','T_D','T_E']
tasks['legacy_discovery_ref']='part_0_validation/stage_0a/legacy_resource_review.json'
yout(M/'task_manifest.yaml',tasks)
splits=jload(M/'splits.json'); splits['status']='BLOCKED_NO_BOUND_INITIAL_STATE_GENERATOR'; splits['case_definitions']=[]
splits['note']='64/50/50 are planned per-task counts only. No real task case, image, object binding or graph signature has been created or claimed.'
jout(M/'splits.json',splits)
vlm=yload(M/'vlm_manifest.yaml'); vlm['credential_presence_evidence']=host['credential_presence_only']; vlm['credential_check_scope']=host['credentials_scope']; vlm['requests_executed']=0
yout(M/'vlm_manifest.yaml',vlm)
checks=[
 {'check':'source_hashes','status':'PASS' if host['source_hashes_all_match'] else 'BLOCKED','evidence':'source_inventory.json'},
 {'check':'target_host_and_new_repository','status':'PASS','evidence':'environment_probe.json'},
 {'check':'dependency_resolution','status':'PASS' if (R/'uv.lock').is_file() else 'BLOCKED','evidence':'host_logs/resolver.log; ../../../../uv.lock'},
 {'check':'dependency_install_and_pip_check','status':'PASS' if software_ok else 'BLOCKED','evidence':'host_logs/install_attempt_02.log; host_logs/install_attempt_02.exit; pip_check if installation completed'},
 {'check':'imports','status':'PASS' if all(v['import_status']=='PASS' for v in sw['packages'].values()) else 'BLOCKED','evidence':'software_checks.json'},
 {'check':'RGCN_forward_backward','status':sw['checks'].get('rgcn_forward_backward',{}).get('status','BLOCKED'),'evidence':'software_checks.json'},
 {'check':'checkpoint_roundtrip','status':sw['checks'].get('checkpoint_roundtrip',{}).get('status','BLOCKED'),'evidence':'software_checks.json'},
 {'check':'CP_DISR_production_entrypoints','status':'BLOCKED','evidence':'implementation_inventory.md'},
 {'check':'D0_T_A_T_C_real_asset_bindings','status':'BLOCKED','evidence':'task_manifest.yaml; legacy_resource_review.json'},
 {'check':'controller_camera_verifier_evaluator_time_safety','status':'BLOCKED','evidence':'runtime_manifest.yaml; binding_table.csv'},
 {'check':'VLM_account_region_endpoint_fewshot','status':'BLOCKED','evidence':'manifests/vlm_manifest.yaml; environment_probe.json'},
]
jout(O/'check_results.json',checks)
missing=[]
def walk(obj,prefix,file):
 if isinstance(obj,dict):
  for k,v in obj.items(): walk(v,prefix+'.'+str(k) if prefix else str(k),file)
 elif isinstance(obj,list):
  for i,v in enumerate(obj): walk(v,f'{prefix}[{i}]',file)
 elif isinstance(obj,str) and obj.startswith('MUST_'):
  missing.append({'file':file,'field':prefix,'value':obj})
for p in sorted(M.glob('*.yaml')):
 walk(yload(p),'',str(p.relative_to(E)))
walk(yload(E/'configs/resolved_defaults.yaml'),'','configs/resolved_defaults.yaml')
jout(O/'unbound_must_bind.json',missing)
with (O/'unbound_must_bind.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['file','field','value']); w.writeheader(); w.writerows(missing)
with (O/'binding_table.csv').open('w',newline='') as f:
 w=csv.writer(f); w.writerow(['field','proposed','observed','status','evidence'])
 for k,v in runtime['runtime'].items():
  unbound=isinstance(v,str) and v.startswith('MUST_')
  w.writerow(['runtime.'+k,'MUST_BIND',json.dumps(v,ensure_ascii=False) if isinstance(v,dict) else v,'MUST_BIND' if unbound else 'BOUND', 'legacy_resource_review.json; manifests/runtime_manifest.yaml' if unbound else 'environment_probe.json'])
 for item in missing:
  if item['file']=='manifests/runtime_manifest.yaml' and item['field'].startswith('runtime.'): continue
  w.writerow([item['file']+':'+item['field'],item['value'],'NOT_BOUND','MUST_BIND' if 'MUST_BIND' in item['value'] else 'MUST_VERIFY', 'Active manifest placeholder; no approved runtime evidence'])
modules={
 'contracts/registry.py':'技能、谓词、类型 grounding 与真实合同来源未绑定',
 'graph/snapshots.py, intervention.py':'不可变快照、三态 overlay 与不变量生产实现未建立',
 'graph/rgcn.py, readout.py':'仅执行第三方 RGCN 库探针；尚无生产四路编码和 goal readout',
 'vlm/client.py, cache.py, validate.py':'只有冻结 schema/prompt；无已绑定 API adapter、few-shot 和缓存入口',
 'policy/model.py':'冻结 v2.1 七配置、GRU、V/Q 与单一参数所有者未实现',
 'rl/collector.py, rollout.py':'无真实平台 on-policy skill transition 接口',
 'rl/targets.py, losses.py, recurrent.py':'旧 PPO/reward 实现未认证为本版 SMDP/当前参数 prefix 重算',
 'env/adapter.py, skills/executor.py':'无 D0/T_A/T_C 的正式平台和固定技能接口绑定',
 'evaluation/run.py, metrics.py':'无本版独立 task_success evaluator 和配对 case 入口',
 'reporting/build.py':'本次仅有 Stage 0A 审计报告生成器；无科研结果报告入口',
}
inventory='# Stage 0A 实现清单\n\n状态：BLOCKED。生产入口不以建议路径冒充已存在实现。\n\n| 计划 §3.3 模块 | 盘点结论 |\n|---|---|\n'
inventory+='\n'.join('| `'+k+'` | '+v+' |' for k,v in modules.items())
inventory+='\n\n## 已有历史代码的实际核查\n\n'
for repo in review['repositories']:
 inventory+=f"- `{repo['repository']}`，commit `{repo['git_hash']}`；路径检索 CP-DISR/M1 候选 {len(repo['cp_disr_path_candidates'])} 项，定点源码审查 {len(repo['inspected_sources'])} 个文件。\n"
inventory+='\n完整范围、真实路径、SHA-256 和行号证据见 `legacy_resource_review.json`。该检索不宣称穷尽服务器所有文件。历史 reward shaping、旧离散任务或控制器可供后续适配审查，但未直接绑定为 v2.1 训练器或 D0/T_A/T_C 资产。\n\n因外部任务与控制器合同仍未绑定，本轮在 Stage 0A 核心资源停止条件处结束；没有用自造平台、toy nominal transition 或空壳 CLI 填补缺项。后续补做 0A 时需实现/绑定上述生产模块。\n'
(O/'implementation_inventory.md').write_text(inventory)
groups=[
 ('运行平台','simulator_or_robot、environment_version、task_assets'),
 ('技能与事实','controller_manifest、skill_contracts、verifier_thresholds；真实前置/ADD/DEL、三态证据与标定'),
 ('视觉','camera、calibration_manifest、perception_checkpoint'),
 ('时间与安全','skill_timeouts、task_deadlines、actual_interaction_time_unit、reference_skill_seconds_by_task、safety_authorization'),
 ('独立评价','task_evaluator_version；真实终端成功和结束规则'),
 ('实现入口','runtime.entrypoints；train_cli/eval_cli/tests_cli/cache_cli/report_cli；模型源码 hash 与参数量'),
 ('任务与 split','D0/T_A/T_C 的 asset_ref、controller_refs、goal_atoms、skill_contract_refs、camera_verifier_refs、initial_state_generator、deadline_seconds、reference_skill_seconds、split_ref；64/50/50 实例池'),
 ('VLM','region、base_http_api_url、api_account_authorized、fewshot_manifest；seed 参数实际支持能力另待核实'),
 ('资源配置','实际 num_envs、candidate view 分块与运行时钟来源；GPU 清单不是专属训练资源预约'),
]
if not software_ok:
 groups.insert(0,('软件安装/验证','推荐 profile 虽已解析，但安装/导入/数值检查未全部通过；详见 host_logs 与 software_checks.json'))
summary=f'''# CP-DISR v2.1 Stage 0A 完整报告

**最终状态：BLOCKED。Stage 0B 条件尚未具备；Stage 0B 未执行，保持 NOT_STARTED。**

完成时间（UTC）：{now}。Method：CP-DISR v2.1；计划：EEP-v1.0。

## 1. 授权范围与目录

本轮仅执行 Stage 0A Environment & Manifest Freeze。文档中的后续阶段、旧 prompt 和示例命令只用于理解约束，不扩展本轮授权。

- 服务器项目：`{R}`
- EXPERIMENT_ROOT：`{E}`
- 分支：`{git('branch','--show-current')}`
- 取证基准 commit：`{base_hash}`；最终交付 commit 在 delivery receipt 中单独记录。
- origin：`git@github.com:rollinpioneer/graph.git`
- 新仓库采用独立根提交历史，未把过时分支或主仓库文件复制为当前方法实现。旧仓库仅被读取。
- 本轮未推送 GitHub；未切换或修改旧仓库分支。

## 2. 完成了什么

保存全部用户附件与冻结规范；核对源 hash；建立独立仓库、分支和虚拟环境；记录真实 host/GPU/driver/OS；解析推荐依赖；执行以下基础设施检查；定点审查旧项目资源；生成 manifests、绑定表、缺项清单、实现清单、完整报告和状态历史。

Kit 自带的聊天沙箱预检已归档为 supplied_kit 历史，不能当作本服务器执行证据。`inputs/` 和 `experiments/sources/` 原件保持不变。

## 3. 检查结果

| 检查项 | 结果 | 证据（相对阶段目录） |
|---|---|---|
'''
summary+='\n'.join(f"| {c['check']} | {c['status']} | `{c['evidence']}` |" for c in checks)
summary+=f'''

实际主机为 gpu03 / Rocky Linux 10.2 / {host['architecture']}；8 张 NVIDIA A100-SXM4-40GB，driver 610.57.04。完整 GPU 清单及探测时显存占用见 environment_probe.json；不承诺资源独占或性能。实际系统 Python 未被覆盖。

软件 profile：**{software['profile_status']}**。目标 Python `{sw['python']}`；完整已解析版本、导入结果和错误见 dependency_table.csv / software_checks.json。独立环境位于 `.venv`，uv 0.8.22 位于 `.bootstrap`。真实 resolver 和安装日志保存在 host_logs；pip check / freeze 仅在安装完成后执行，实际有无日志以 host_logs 为准，wheel 来源和 uv.lock hash 写入 software_manifest.yaml。

安装 attempt 01 遇到 torch wheel 网络超时，原始失败日志和首次缺包检查保存在 host_logs/attempt_01。attempt 02 保持版本与来源不变，只提高 UV_HTTP_TIMEOUT 至 300 秒、下载并发降为 2；实际结果见 install_attempt_02.log / .exit。安装失败不记作训练失败。软件检查在安装恢复后才可升级 VALIDATED。

RGCN 探针使用合成张量，仅检查库前后向；若 round-trip 成功，文件是未训练库模型的 state_dict，不是 CP-DISR policy checkpoint。两者不证明方法语义通过，不替代 T01–T25。

## 4. 实际运行与结果边界

- 科研训练完成 / 失败 / 复用：**0 / 0 / 0**；无 training run_id。
- 真实 skill transitions：**0**；PPO updates / optimizer steps：**0 / 0**。
- 性能评价 episodes：**0**；控制器/机器人动作：**0**；VLM 请求：**0**。
- 执行的是一次 host 盘点和一次软件基础设施检查批次，详细检查计数见 check_results.json。
- 没有成功率、学习曲线、机制收益或泛化结果；没有用历史训练结果充当本版结果。

## 5. 仍未绑定的 MUST_BIND 项

'''
summary+='\n'.join(f'- **{title}**：{fields}。' for title,fields in groups)
summary+=f'''

可选任务 T_B/T_D/T_E 同样未绑定，但它们不是当前 0A 的独立阻塞原因。主任务 D0/T_A/T_C 缺项本身已足以阻塞。完整逐文件逐字段占位项共 **{len(missing)}** 行（包含镜像字段和 MUST_VERIFY 类项，不是 {len(missing)} 个独立问题），见 `unbound_must_bind.csv` / `.json`；`binding_table.csv` 同时列出已绑定字段。

API 检查只记录当前非 login SSH 进程中常用环境变量是否设置，未读取或打印密钥值；未发现不等于账户在其他凭据存储中绝对不存在。真实 region、endpoint、账户权限和 3 个开发 few-shot 均未确认。

## 6. 任务与实现判定

D0/T_A/T_C 尚未形成可验证的资产→技能合同→相机/Verifier→独立评价器映射，真实 case_definitions 为空。64/50/50 只是计划数，不伪造初始图像、对象绑定或合法动作序列。CUPID 和 LIBERO 路径真实存在，LIBERO commit 为 8f1084e3132a39270c3a13ebe37270a43ece2a01，发现 BDDL 任务文件；尚无本次模板与固定技能、视觉权限、Verifier、安全合同的一致映射。详见 implementation_inventory.md、legacy_resource_review.json 和 platform_discovery.json。

## 7. 相对冻结配置的变更

冻结 Method、关系 schema、prompt 原件及七方法定义未修改；无性能调参。目录、Git 分支和软件环境由真实 host 绑定。OS 从推荐的 Ubuntu 兼容候选记录为实际 Rocky Linux 10.2；这属于允许的环境适配，软件通过与否以探针为准。torchvision 为可选依赖，尚无绑定视觉前端需要它，故未安装。软件推荐版本未暗中升级或替换。

## 8. 状态与下一步

PASS 要求核心代码和真实运行资产完成绑定；软件检查通过不能抵消这些核心缺项，因此选择 **BLOCKED**，不能标 PASS 或 PASS_WITH_NOTES。

**目前不具备完整执行 Stage 0B 的条件。** 后续应先补做 Stage 0A：明确真实平台及 D0/T_A/T_C 资产，冻结控制器/视觉/Verifier/评价器/时间/安全合同，完成生产模块和 CLI，生成真实 split，并绑定 API 账户地区/端点/few-shot，再重新判定 0A。Stage 0B 即使可用纯程序 fixture，当前生产实现缺失也不能视为可验收。

本轮报告写入后停止。0B 及其他未授权阶段保持 NOT_STARTED；不会自动进入 0B，不调用模型或启动训练。
'''
(O/'stage_0a_summary.md').write_text(summary)
(O/'binding_report.md').write_text('# Stage 0A 绑定报告\n\n**BLOCKED**。已绑定 host、repo、分支、真实硬件/OS、软件环境；核心运行资源仍未绑定。\n\n'+ '\n'.join(f'- {a}：{b}。' for a,b in groups)+'\n\n完整逐字段证据见 binding_table.csv 与 unbound_must_bind.csv；完整阶段报告见 stage_0a_summary.md。\n')
for name in ['entrypoints.yaml','software_manifest.yaml','runtime_manifest.yaml','task_manifest.yaml','splits.json']:
 shutil.copy2(M/name,O/name)
shutil.copy2(E/'configs/resolved_defaults.yaml',O/'resolved_defaults.yaml')
issues=[{'field':k,'issue':v,'state':'MUST_BIND'} for k,v in groups]
status={'stage':'0A','status':'BLOCKED','method_version':'CP-DISR-v2.1','plan_version':'EEP-v1.0',
 'git_hash':base_hash,'git_hash_semantics':'Checked repository base plus git_snapshot.json dirty diff and untracked hashes; final delivery commit is recorded externally',
 'source_hashes':{s['relative_path']:s['observed_sha256'] for s in source},
 'started_at':old['started_at'],'completed_at':now,'runs_completed':[],'runs_failed':[],'runs_reused':[],
 'selected_tasks':[],'requested_task_templates':['D0','T_A','T_C'],'selected_config':'configs/resolved_defaults.yaml',
 'issues':issues,'tuning_changes':[],'runtime_adjustments':[{'field':'OS','observed':'Rocky Linux 10.2'},{'field':'dependency_indexes','observed':actual_indexes},{'field':'UV_HTTP_TIMEOUT','old':30,'new':300,'reason':'torch wheel download timeout; no package version change'}],'next_stage':'0B','skipped':False,'execution_scope':'Stage 0A only on target gpu03; no auto advance',
 'readiness':{'target_host':True,'software':software_ok,'core_code':False,'runtime':False,'account':False,'stage_0b':False},
 'infrastructure_checks':checks,'rl_jobs_executed':0,'real_skill_transitions':0,'ppo_updates':0,'performance_episodes':0,
 'vlm_requests_executed':0,'robot_actions_executed':0,'stage_0b_executed':False,'unbound_inventory':'part_0_validation/stage_0a/unbound_must_bind.json'}
Draft202012Validator(jload(E/'schemas/stage_status.schema.json')).validate(status)
jout(E/'stage_status/stage_0a.json',status)
delivery_path=E/'DELIVERY_STATUS.json'
shutil.copy2(delivery_path,O/'supplied_kit_delivery_status.json')
jout(delivery_path,{'artifact':'CP-DISR v2.1 Stage 0A target-host execution','plan_version':'EEP-v1.0','method_version':'CP-DISR-v2.1','stage_0a_status':'BLOCKED','other_stages':'NOT_STARTED','actual_rl_jobs_this_response':0,'actual_vlm_calls_this_response':0,'actual_scope':'Stage 0A on gpu03 only','production_tests_run':False,'trainer_included':False,'report':'part_0_validation/stage_0a/stage_0a_summary.md'})
(R/'README.md').write_text('# CP-DISR v2.1 — Stage 0A\n\nStatus: **BLOCKED**. Only Stage 0A is authorized and executed.\n\n- Full report: `experiments/part_0_validation/stage_0a/stage_0a_summary.md`\n- Current status: `experiments/stage_status/stage_0a.json`\n- Exact unresolved fields: `experiments/part_0_validation/stage_0a/unbound_must_bind.csv`\n- Frozen originals: `inputs/` and `experiments/sources/`\n- Independent project environment: `.venv`; resolver: `.bootstrap/bin/uv`\n\nThis repository begins a new independent history; the previous graph repositories are discovery sources only. Origin is configured, but this branch has not been pushed. Do not start Stage 0B automatically. The bundled Kit README and original preflight history describe the supplied package, not target-host completion.\n')
history=E/'stage_status/history'; history.mkdir(exist_ok=True)
jout(history/('stage_0a_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),status)
for p in (E/'stage_status').glob('stage_*.json'):
 if p.name!='stage_0a.json': assert jload(p)['status']=='NOT_STARTED',(p,jload(p)['status'])
for item in source:
 if item['relative_path'].startswith('sources/'):
  assert sha(E/item['relative_path'])==item['observed_sha256']
required=yload(E/'configs/stages/stage_0a.yaml')['required_files']
assert all((O/name).is_file() for name in required)
diff=subprocess.check_output(['git','-C',str(R),'diff','--binary','HEAD'])
(O/'git_dirty.diff').write_bytes(diff)
untracked=git('ls-files','--others','--exclude-standard').splitlines()
jout(O/'git_snapshot.json',{'git_hash':base_hash,'branch':git('branch','--show-current'),'dirty_diff_sha256':hashlib.sha256(diff).hexdigest(),
 'untracked_files':{f:sha(R/f) for f in untracked if (R/f).is_file() and f not in [str((O/'git_snapshot.json').relative_to(R)),str((O/'checksums.sha256').relative_to(R))]},
 'scope':'Snapshot after report/status generation, before final validation receipt and delivery commit; generated audit files excluded from self-hashing'})
validation={'status':'PASS','stage_0a_status':'BLOCKED','required_files_present':required,'status_schema_valid':True,
 'frozen_sources_unchanged':True,'other_stages_not_started':True,'must_placeholder_rows':len(missing),'software_profile':software['profile_status']}
jout(O/'delivery_validation.json',validation)
(O/'checksums.sha256').write_text(''.join(sha(p)+'  '+str(p.relative_to(R))+'\n' for p in sorted(O.rglob('*')) if p.is_file() and p.name!='checksums.sha256' and 'preflight_history' not in p.parts))
print(json.dumps(validation,indent=2))
