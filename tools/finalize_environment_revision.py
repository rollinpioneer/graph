"""Finalize Stage 0A evidence only; never execute Stage 0B or read credentials."""
import csv, hashlib, json, shutil, subprocess
from datetime import datetime, timezone
from pathlib import Path
import yaml

R=Path('/home/__compress_data/xushijie/graph_cp_disr_v2_1')
E=R/'experiments'; M=E/'manifests'
O=R/(R/'.stage0a_environment_revision').read_text().strip()
PREV=R/(R/'.remediation_revision').read_text().strip()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def ysave(p,v):p.write_text(yaml.safe_dump(v,allow_unicode=True,sort_keys=False))
def git(*args):return subprocess.check_output(['git','-C',str(R),*args],text=True).strip()
probe=json.loads((O/'software_checks.json').read_text())
checks=json.loads((O/'infrastructure_command_checks.json').read_text())
validated=probe['profile_validated'] and all(c['exit_code']==0 for c in checks)
software=yaml.safe_load((M/'software_manifest.yaml').read_text())
software.update(profile_status='VALIDATED_COMPATIBILITY_PROFILE' if validated else 'BLOCKED',
    python='3.10.19',python_executable=str(R/'.venv-stage0a/bin/python'),
    preferred_python='3.11.13',compatibility_adjustment_reason='Reuse existing PyTorch 2.7.1+cu126/CUDA binaries without modifying the source environment',
    evidence=str((O/'software_checks.json').relative_to(E)),lockfile_sha256=sha(R/'uv.lock'),
    reuse_provenance=str((O/'reused_package_links.json').relative_to(E)),
    source_environment_modified=False,reuse_mode='Isolated venv with explicit read-only reuse by symlink; source packages must remain available and unchanged',
    method_validation='NOT_RUN_STAGE_0B_NOT_AUTHORIZED')
software['wheel_indexes']=['https://mirrors.nju.edu.cn/pypi/web/simple']
software['torch_distribution_metadata']='2.7.1'
software['torch_runtime_build']='2.7.1+cu126'
software['frozen_sync_validation']='UV_PROJECT_ENVIRONMENT=.venv-stage0a .bootstrap/bin/uv sync --frozen --dry-run: Would make no changes'
ysave(M/'software_manifest.yaml',software)
ep=yaml.safe_load((M/'entrypoints.yaml').read_text())
exe='.venv-stage0a/bin/python'
for k in ['train_cli','eval_cli','tests_cli','cache_cli']:
    ep[k]=ep[k].replace('PYTHONPATH=src python ',f'PYTHONPATH=src {exe} ')
for v in ep['commands'].values():v['cli']=v['cli'].replace('PYTHONPATH=src python ',f'PYTHONPATH=src {exe} ')
ep['infrastructure_only']['software_probe']={'path':'tools/stage0a_reuse_software_probe.py','cli':exe+' tools/stage0a_reuse_software_probe.py','sha256':sha(R/'tools/stage0a_reuse_software_probe.py')}
ep['infrastructure_only']['kit_plan']['cli']=exe+' experiments/tools/plan_stage.py --stage 0A'
ep['report_cli']=exe+' tools/finalize_environment_revision.py'
ep['note']='Core code present. Stage 0A library validation is separate from Stage 0B method tests, which remain NOT_RUN. Real runtime unbound. API region/endpoint user-bound; model access MUST_VERIFY in Stage 0C.'
ysave(M/'entrypoints.yaml',ep)
credential=json.loads((O/'credential_availability.json').read_text())
vlm=yaml.safe_load((M/'vlm_manifest.yaml').read_text())
vlm['credential_availability']=credential
vlm['credential_status']='LOCAL_FILE_PRESENT_NOT_LOADED_OR_SERVER_BOUND'
vlm.update(region='cn-beijing',base_http_api_url='https://dashscope.aliyuncs.com/api/v1',api_account_authorized='MUST_VERIFY_MODEL_ACCESS',model_access_status='MUST_VERIFY',model_access_verification_stage='0C',allow_automatic_model_or_endpoint_fallback=False,configuration_provenance='User explicitly supplied Beijing region and endpoint; instructed no API request in Stage 0A')
experiment=yaml.safe_load((M/'experiment_manifest_v0.yaml').read_text())
experiment['vlm'].update(region='cn-beijing',base_http_api_url=vlm['base_http_api_url'],api_account_authorized='MUST_VERIFY_MODEL_ACCESS')
experiment['software']=software
ysave(M/'experiment_manifest_v0.yaml',experiment)
resolved=yaml.safe_load((E/'configs/resolved_defaults.yaml').read_text())
resolved['software']=software
resolved['vlm'].update(region='cn-beijing',base_http_api_url=vlm['base_http_api_url'],api_account_authorized='MUST_VERIFY_MODEL_ACCESS')
ysave(E/'configs/resolved_defaults.yaml',resolved)
ysave(M/'vlm_manifest.yaml',vlm)
runtime=yaml.safe_load((M/'runtime_manifest.yaml').read_text())
runtime['vlm_runtime'].update(region='cn-beijing',base_http_api_url=vlm['base_http_api_url'],api_account_authorized='MUST_VERIFY_MODEL_ACCESS',model_access_verification_stage='0C',automatic_fallback_allowed=False)
runtime['execution_scope']='Stage 0A environment reuse, library probe, manifest audit; no API/robot/training/evaluation/Stage 0B'
runtime['runtime']['git_hash']=git('rev-parse','HEAD')
runtime['runtime']['git_hash_semantics']='Pre-revision base; delivery receipt records resulting commit'
ysave(M/'runtime_manifest.yaml',runtime)
status=json.loads((O/'previous_stage_0a.json').read_text())
status.update(status='BLOCKED',git_hash=git('rev-parse','HEAD'),
    git_hash_semantics='Pre-revision base; exact delivery commit recorded in receipt',
    completed_at=datetime.now(timezone.utc).isoformat(),current_revision=str(O.relative_to(E)),
    execution_scope=runtime['execution_scope'],stage_0b_executed=False)
status['readiness'].update(software=validated,account=False,runtime=False,stage_0b=validated,
    torch_runtime_validation_pending=not validated,external_runtime_binding_pending=True,
    api_binding_pending=True,credential_file_present=credential['exists'],
    credential_server_bound=False,stage_0b_authorization_required=True,
    api_configuration_bound=True,model_access_verification_pending=True,method_unit_tests_validated=False)
status['issues']=[
    {'category':'external_runtime_missing','issue':'D0/T_A/T_C concrete assets, controllers, perception/verifier, evaluator, safety, timing and initialization splits are not certified; see unbound_must_bind.csv'},
    {'category':'api_verification_deferred','issue':'Local credential file exists; no key read or API call. Beijing region and endpoint are user-bound. Model access MUST_VERIFY in Stage 0C; no automatic model or endpoint fallback.'},
    {'category':'method_validation','issue':'Stage 0B T01-T25 remain NOT_RUN; library smoke tests are not Method validation.'}]
if not validated:status['issues'].append({'category':'software','issue':'See software_checks.json and infrastructure_command_checks.json for remaining failures'})
status['runtime_adjustments'].append({'field':'software_profile.python','old':'3.11.13','new':'3.10.19','reason':software['compatibility_adjustment_reason']})
status['infrastructure_checks'].append({'check':'environment_reuse_revision','status':'PASS_WITH_NOTES' if validated else 'BLOCKED','evidence':str((O/'software_checks.json').relative_to(E))})
status['runs_completed'].append({'kind':'Stage 0A library infrastructure probe','evidence':str((O/'software_checks.json').relative_to(E)),'trained_policy':False})
save(E/'stage_status/stage_0a.json',status)
save(O/'stage_0a.json',status)
save(E/'stage_status/history'/('stage_0a_'+O.name+'.json'),status)
rows=list(csv.DictReader((PREV/'binding_table.csv').open()))
for row in rows:
    if row['field'].endswith('.region'):
        row.update(value='cn-beijing',category='discovered_and_bound',status='BOUND_USER_DECLARATION',evidence='Explicit user reply in this revision')
    elif row['field'].endswith('.base_http_api_url'):
        row.update(value='https://dashscope.aliyuncs.com/api/v1',category='discovered_and_bound',status='BOUND_USER_DECLARATION',evidence='Explicit user reply in this revision')
    elif row['field'].endswith('.api_account_authorized'):
        row.update(value='MUST_VERIFY_MODEL_ACCESS',category='deferred_stage_0c_verification',status='MUST_VERIFY',evidence='User defers verification to Stage 0C and prohibits automatic model/endpoint fallback')
    if row['field']=='software.torch_cuda_pyg_runtime':
        row.update(value='VALIDATED_COMPATIBILITY_PROFILE' if validated else 'BLOCKED',category='discovered_and_bound' if validated else 'environment_missing',status='PASS_WITH_NOTES' if validated else 'BLOCKED',evidence=str(O/'software_checks.json'))
rows.append({'field':'vlm.credential_file','value':'PRESENT_NONEMPTY','category':'discovered_and_bound','status':'LOCAL_AVAILABILITY_ONLY','evidence':str(O/'credential_availability.json')})
rows.append({'field':'vlm.credential_server_binding','value':'MUST_BIND','category':'external_runtime_missing','status':'MUST_BIND','evidence':'Credential intentionally not loaded/transferred during Stage 0A'})
for name,subset in [('binding_table.csv',rows),('unbound_must_bind.csv',[r for r in rows if 'MUST_' in r['value'] or r['status'] in ['MUST_BIND','BLOCKED','NOT_RUN_ENVIRONMENT_BLOCKED']])]:
    with (O/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['field','value','category','status','evidence'],lineterminator='\n');w.writeheader();w.writerows(subset)
source_check=[]
for name,wanted in status['source_hashes'].items():
    p=(R/name) if (R/name).is_file() else E/name;source_check.append({'path':name,'sha256':sha(p),'unchanged':sha(p)==wanted})
assert all(x['unchanged'] for x in source_check)
save(O/'source_inventory.json',source_check)
for name in ['software_manifest.yaml','entrypoints.yaml','runtime_manifest.yaml','task_manifest.yaml','splits.json','vlm_manifest.yaml','experiment_manifest_v0.yaml']:
    shutil.copy2(M/name,O/name)
shutil.copy2(E/'configs/resolved_defaults.yaml',O/'resolved_defaults.yaml')
shutil.copy2(PREV/'implementation_inventory.md',O/'implementation_inventory.md')
if not (O/'environment_probe.json').exists():shutil.copy2(E/'part_0_validation/stage_0a/environment_probe.json',O/'environment_probe.json')
summary=f'''# Stage 0A 环境复用与前置复核报告

**完整 Stage 0A：BLOCKED。软件基础设施：{'PASS_WITH_NOTES' if validated else 'BLOCKED'}。正式 Stage 0B：NOT_STARTED。**

项目：`{R}`；分支：`{git('branch','--show-current')}`；基准提交：`{git('rev-parse','HEAD')}`。最终提交见交付 receipt。

## 本轮完成

- 检查已有 conda/venv 环境，使用已有 Python 3.10.19 和 PyTorch 2.7.1+cu126/CUDA 12.6 文件，在 `.venv-stage0a` 建立隔离环境；仅补齐缺失依赖，未修改旧项目或源环境。
- 首选 Python 3.11.13 改为兼容配置 3.10.19，属于手册允许的软件兼容调整；冻结 Method、Torch/PyG 核心版本和方法参数未改。
- 依赖解析、pip check、import、生产核心模块 import、最小 RGCN 前/反向与未训练 state_dict round-trip 结果见 `software_checks.json` / `infrastructure_command_checks.json`。
- API 文件仅 stat 检查存在与非空，未读取内容、未上传、未写入 Git、未发送 API 请求。文件存在不能证明账户授权、region、endpoint 或模型权限。
- API 地域按用户声明绑定为华北2（北京，中国大陆）`cn-beijing`，endpoint 为 `https://dashscope.aliyuncs.com/api/v1`。`qwen3.8-max-0902` 权限记为 `MUST_VERIFY`，按用户要求在后续获准的 Stage 0C 验证；不可访问即停止，不自动更换模型或 endpoint。
- 全部旧报告、状态及 manifest 保存；本轮独立 revision，已更新 active stage_status 和软件/入口 manifest。

## 真实结果与数量

运行 1 次 Stage 0A 软件 probe；已记录 2 次安装准备中止（遗留队列下载取消、初次复用安装超时）、1 次离线解析失败及后续成功修复，详见 bootstrap_attempts.json。RGCN 前/反向与 round-trip 的实际结果详见 JSON。0 个训练 transition，0 次 PPO update，0 个性能 episode，0 次机器人动作，0 次 VLM 请求。未运行正式 T01–T25。以前的 13 项纯逻辑通过记录仅作为历史证据，不冒充本轮新运行。

## 剩余 MUST_BIND

逐字段清单：`unbound_must_bind.csv`。主要为 D0/T_A/T_C 真实对象与场景、控制器、相机及标定、冻结感知 checkpoint、Verifier 阈值、独立 evaluator、安全授权、skill timeout/deadline/clock/d_ref、64/50/50 初始化池，以及模型访问权限和 few-shot。地域/endpoint 已按用户声明绑定，模型权限明确延后到 Stage 0C 实测。

之前发现的 LIBERO/历史 graph 资源只是可适配来源。本轮未取得能把结构模板认定为真实可执行绑定的新证据，不虚构 resolved task、时长或实验结果。

## 是否具备 Stage 0B 条件

{'已满足手册所列 core_code + 可运行软件 + schema/fixtures 的独立单元验收前置。' if validated else '软件前置尚未全部通过，暂不具备。'}仍须用户明确授权才执行正式 Stage 0B；数值方法测试尚未通过验收。完整 Stage 0A 仍因真实 runtime 绑定缺失为 BLOCKED，不能写 PASS。模型权限按用户指令延后至 Stage 0C 验证，不声称已通过。

## 复用限制与后续

复用的 Torch/CUDA 文件仍依赖原环境路径；源环境被删除或更新后必须重新验收。具体链接及来源见 `reused_package_links.json`，实际版本快照见 `installed_requirements.txt`。Stage 0C 还需 API 配置/provider、真实 dev 输入及缓存审核；Stage 1A 还需真实 runtime 全部绑定和前置阶段验收。

本轮停止于 Stage 0A，不推送 GitHub，不自动进入 Stage 0B。
'''
(O/'stage_0a_summary.md').write_text(summary)
(O/'binding_report.md').write_text(summary)
save(E/'DELIVERY_STATUS.json',{'stage_0a_status':'BLOCKED','other_stages':'NOT_STARTED','revision':str(O.relative_to(E)),'software_validated':validated,'formal_stage_0b_executed':False})
(R/'README.md').write_text('# CP-DISR v2.1\n\nStage 0A: **BLOCKED** on real runtime. API model access verification deferred to Stage 0C; region and endpoint user-bound.\n\nLatest report: `'+str((O/'stage_0a_summary.md').relative_to(R))+'`.\n\nSoftware: '+('validated compatibility profile' if validated else 'blocked')+'; use `PYTHONPATH=src .venv-stage0a/bin/python -m cp_disr --help`.\n\nStage 0B remains NOT_STARTED and requires explicit authorization. Frozen sources and earlier evidence are preserved.\n')
print(json.dumps({'revision':str(O),'software_validated':validated,'stage_0a':'BLOCKED','stage_0b_executed':False},indent=2))
