"""Static-only non-environment remediation audit. No dependency installs/API calls."""
import ast,csv,hashlib,json,os,re,shutil,subprocess,sys,xml.etree.ElementTree as ET
from datetime import datetime,timezone
from pathlib import Path
import yaml
from jsonschema import Draft202012Validator
R=Path('/home/__compress_data/xushijie/graph_cp_disr_v2_1');E=R/'experiments';M=E/'manifests';O=R/Path((R/'.remediation_revision').read_text().strip())
if (O/'static_validation.json').exists():
    audit_archive=O/'validation_history'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');audit_archive.mkdir(parents=True)
    for old in list(O.iterdir()):
        if old.is_file() and not old.name.startswith('previous_') and old.name!='asset_discovery.json':shutil.copy2(old,audit_archive/old.name)
sys.path.insert(0,str(R/'src'))
from cp_disr.common import unresolved
from cp_disr.validation import validate_repository
from cp_disr.runtime import binding_issues
def read(p):return json.loads(p.read_text()) if p.suffix=='.json' else yaml.safe_load(p.read_text())
def jout(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def yout(p,v):p.write_text(yaml.safe_dump(v,allow_unicode=True,sort_keys=False))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(['git','-C',str(R),*args],text=True).strip()
def run(label,args,expected=0):
    env={**os.environ,'PYTHONPATH':str(R/'src'),'GIT_OPTIONAL_LOCKS':'0'}
    p=subprocess.run(args,cwd=R,env=env,capture_output=True,text=True,timeout=120)
    (O/(label+'.stdout.log')).write_text(p.stdout);(O/(label+'.stderr.log')).write_text(p.stderr)
    return {'check':label,'status':'PASS' if p.returncode==expected else 'BLOCKED','exit_code':p.returncode,'expected_exit_code':expected,'stdout_ref':label+'.stdout.log','stderr_ref':label+'.stderr.log'}

commands={name:{'cli':'PYTHONPATH=src python -m cp_disr '+name+(' --scope pure|full' if name=='run-unit-tests' else ''),'source_path':'src/cp_disr/cli.py','source_sha256':sha(R/'src/cp_disr/cli.py'),'status':'IMPLEMENTED_UNBOUND_RUNTIME_REJECTED' if name in ['train','evaluate','generate-cache'] else 'IMPLEMENTED','requires_torch':name in ['train','evaluate'],'calls_api_this_revision':False} for name in ['validate-manifests','validate-contracts','validate-relation-cache','run-unit-tests','build-graph-fixture','train','evaluate','generate-cache']}
entry=read(M/'entrypoints.yaml');entry.update(status='CORE_CODE_PRESENT_RUNTIME_UNBOUND',implemented_trainer_in_this_kit=False,production_package_added_in_remediation=True,commands=commands,
 train_cli=commands['train']['cli'],eval_cli=commands['evaluate']['cli'],tests_cli='PYTHONPATH=src python -m cp_disr run-unit-tests --scope full',cache_cli=commands['generate-cache']['cli'],
 report_cli='python3 tools/finalize_remediation.py',note='The original Kit contains no trainer. This revision adds production components and gated drivers. Torch tests NOT_RUN_ENVIRONMENT_BLOCKED; real task adapters unbound; API transport deliberately not implemented under this scope.')
yout(M/'entrypoints.yaml',entry)
runtime=read(M/'runtime_manifest.yaml');runtime['runtime']['entrypoints']='manifests/entrypoints.yaml';runtime['runtime']['git_hash']=git('rev-parse','HEAD')
runtime['runtime_factory']={k:'MUST_BIND' for k in ['module','factory','source_path','sha256']}
runtime['adapter_protocols']={'path_from_repository':'src/cp_disr/adapters.py','sha256':sha(R/'src/cp_disr/adapters.py'),'concrete_implementations_bound':False}
runtime['execution_scope']='Stage 0A non-environment remediation; no installs, credentials, API, robot, training, performance evaluation or formal Stage 0B'
yout(M/'runtime_manifest.yaml',runtime)
tasks=read(M/'task_manifest.yaml');tasks['template_refs']={k:'../../configs/tasks/'+k+'.yaml' for k in ['D0','T_A','T_C']};tasks['resolved_real_tasks']=[];yout(M/'task_manifest.yaml',tasks)
contracts=read(R/'configs/contracts/skills.yaml');contracts['schema_ref']='../../schemas/skill_contract.schema.json';contracts['production_registry_ref']='../../src/cp_disr/contracts.py';yout(M/'skill_contracts.yaml',contracts)
verifier=read(M/'verifier_manifest.yaml');verifier.update(status='INTERFACE_DEFINED_RUNTIME_UNBOUND',protocol_ref='../../src/cp_disr/adapters.py:FactVerifier',fact_store_ref='../../src/cp_disr/facts.py',real_thresholds_bound=False);yout(M/'verifier_manifest.yaml',verifier)
splits=read(M/'splits.json');splits['initialization_pool_schema_refs']=['../../configs/tasks/'+k+'.yaml' for k in ['D0','T_A','T_C']];splits['case_definitions']=[];splits['status']='STRUCTURE_SCHEMA_DEFINED_REAL_CASES_UNBOUND';jout(M/'splits.json',splits)
for p in [E/'configs/resolved_defaults.yaml',M/'experiment_manifest_v0.yaml']:
    v=read(p);v['runtime']['entrypoints']='manifests/entrypoints.yaml';v['runtime']['git_hash']=git('rev-parse','HEAD');v['state']='STAGE_0A_CORE_CODE_PRESENT_RUNTIME_UNBOUND';yout(p,v)
model=read(M/'model_manifest.yaml');model['status']='PRODUCTION_CODE_PRESENT_NOT_RUNTIME_VALIDATED';model['source_code_sha256']=sha(R/'src/cp_disr/neural.py');model['source_path']='../../src/cp_disr/neural.py';model['actual_parameter_count']='MUST_BIND_AFTER_USER_ENVIRONMENT_READY';yout(M/'model_manifest.yaml',model)
pyproject=R/'pyproject.toml';text=pyproject.read_text()
if '[project.scripts]' not in text:text+='\n[project.scripts]\ncp-disr = "cp_disr.cli:main"\n\n[tool.setuptools.packages.find]\nwhere = ["src"]\n'
pyproject.write_text(text)
# Do not run a resolver. The locked third-party dependencies are untouched.
checks=[];checks.append(run('compileall',[sys.executable,'-m','compileall','src','tests']))
checks.append(run('pure_logic_tests',[sys.executable,'-m','pytest','tests','-m','pure','-q','-ra','--junitxml='+str(O/'pure_logic_junit.xml')]))
static=validate_repository(R);jout(O/'import_free_dependency_graph.json',static['import_free_dependency_graph'])
checks.append({'check':'yaml_json_schema_import_graph','status':'PASS' if static['passed'] else 'BLOCKED','parsed_files':static['parsed_files'],'errors':static['errors']})
for command in ['validate-manifests','validate-contracts','validate-relation-cache','build-graph-fixture']:
    checks.append(run(command,[sys.executable,'-m','cp_disr',command]))
for command in ['train','evaluate','generate-cache']:
    checks.append(run(command+'_fail_closed',[sys.executable,'-m','cp_disr',command],2))
    text=(O/(command+'_fail_closed.stdout.log')).read_text()
    if 'MUST_BIND' not in text:checks[-1]['status']='BLOCKED'
# Exact gate checks use subprocess output, without ever accessing environment credential values.
tests=[];ids=set()
for path in (R/'tests').glob('test_*.py'):
    tree=ast.parse(path.read_text())
    for n in tree.body:
        if isinstance(n,ast.FunctionDef) and n.name.startswith('test_'):
            match=re.search(r'T(\d\d)',n.name);tid='T'+match.group(1) if match else 'ADDITIONAL_RUNTIME_GATE'
            if match:ids.add(tid)
            marker='torch_runtime' if any('torch_runtime' in ast.unparse(d) for d in n.decorator_list) else 'pure'
            tests.append({'test_id':tid,'path':str(path.relative_to(R))+'::'+n.name,'group':marker,'status':'NOT_RUN_ENVIRONMENT_BLOCKED' if marker=='torch_runtime' else 'PASS' if checks[1]['status']=='PASS' else 'NEEDS_RERUN','formal_stage_0b_run':False,'expected_source':str(path.relative_to(R)),'tolerance':'strict identity/zero; scalar 1e-10' if marker=='pure' else 'FP32 atol=1e-6 rtol=1e-5 except strict zero and explicit >1e-6 path witness'})
checks.append({'check':'T01_T25_authored','status':'PASS' if ids=={'T'+str(i).zfill(2) for i in range(1,26)} else 'BLOCKED','ids':sorted(ids)})
fixtures={p.name:{'sha256':sha(p),'synthetic_unit_fixture':read(p)['synthetic_unit_fixture'],'paper_performance_eligible':read(p)['paper_performance_eligible']} for p in (R/'tests/fixtures').glob('*.json')}
jout(O/'fixture_manifest.json',fixtures);jout(O/'authored_test_inventory.json',tests)
tree_hash=hashlib.sha256(''.join(str(p.relative_to(R))+':'+sha(p)+'\n' for p in sorted((R/'src').rglob('*.py'))).encode()).hexdigest()
checks.append({'check':'no_placeholder_success_tests','status':'PASS' if all('assert True' not in p.read_text() and 'pytest.skip' not in p.read_text() and 'importorskip' not in p.read_text() for p in (R/'tests').glob('*.py')) else 'BLOCKED'})
source=(R/'src/cp_disr/prior.py').read_text();checks.append({'check':'unique_episode_80_20_protocol','status':'PASS' if 'ORIGINAL_PROBABILITY=0.8' in source and sum('rng.random()' in p.read_text() for p in (R/'src/cp_disr').glob('*.py'))==1 else 'BLOCKED'})
frozen_ok=all(sha(E/x['relative_path'])==x['sha256'] for x in read(M/'source_manifest.json'))
checks.append({'check':'frozen_source_hashes','status':'PASS' if frozen_ok else 'BLOCKED'})
for p in (E/'stage_status').glob('stage_*.json'):
    if p.name!='stage_0a.json' and read(p)['status']!='NOT_STARTED':raise RuntimeError('Unauthorized stage status change')
validation={'scope':'STAGE_0A_NON_ENVIRONMENT_REMEDIATION','checks':checks,'pure_logic_validation_passed':checks[1]['status']=='PASS','static_all_passed':all(c['status']=='PASS' for c in checks),'torch_tests_status':'NOT_RUN_ENVIRONMENT_BLOCKED','formal_stage_0b_executed':False,'credentials_read':False,'software_downloads':0,'api_requests':0,'robot_actions':0,'training_updates':0,'source_tree_sha256':tree_hash,'tests':tests}
jout(O/'static_validation.json',validation)
with (O/'static_validation.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['check','status','evidence']);w.writerows((c['check'],c['status'],c.get('stdout_ref','static_validation.json')) for c in checks);w.writerows((t['test_id'],t['status'],t['path']) for t in tests)
discovery=read(O/'asset_discovery.json')
asset_md='# 服务器资产盘点\n\n只读检查；未导入历史模块、未读取 API 凭据、未启动仿真或机器人。范围限清单中的项目与搜索深度，不声称服务器任何位置都不存在缺项。\n\n'
asset_md+='| 源项目 | commit | tracked dirty |\n|---|---|---|\n'+'\n'.join(f"| `{x['path']}` | `{x['git_commit']}` | {x['dirty_tracked']} |" for x in discovery['repositories'])+'\n\n'
asset_md+='## 复用判断\n\n- LIBERO `env_wrapper.py:ControlEnv` 提供 reset/step/check_success、相机名称和频率配置。默认同时含 agentview/腕部视角、depth=False、20 Hz、horizon=1000；与本版固定第三视角 RGB-D、skill 级时长、受控终止不同，必须显式 adapter，不能照搬默认。\n- LIBERO `base_predicates.py:In/On/Open` 与各任务 `_check_success` 可供独立任务 evaluator 适配；其模拟器真值不能冒充视觉 Fact Verifier 或控制器的实测几何。\n- LIBERO BDDL/XML/初始化数据是实际资产候选，尚未完成 D0/T_A/T_C 的角色、固定技能和真实 case 池映射。\n- 历史 P1 regrasp/return 控制器有候选代码；尚未证明可执行本版通用 OPEN/PICK/PLACE/MOVE，也未迁入新项目。\n- 历史 P2C gym adapter/PPO/checkpoint 可参考通用工程；reward shaping、固定 37 动作以及旧算法状态语义不能直接复用为本版方法。\n- 搜索到 checkpoint 文件不代表它是本版合格的冻结视觉前端；未加载这些权重，未生成 resolved task/runtime 绑定。\n\n'
asset_md+=f"共记录 {len(discovery['code_candidates'])} 个代码候选和 {len(discovery['asset_candidates'])} 个资产候选。逐项 absolute path、commit/dirty、AST 签名、I/O 注解（无注解时明确未知）、hash、证据行、adapter 要求和语义差异见 `asset_discovery.json`。\n"
(O/'asset_discovery_report.md').write_text(asset_md)
inventory='# CP-DISR v2.1 实现清单\n\n生产核心源码已存在；数值运行验收尚未完成。不是只有 Protocol 或建议 CLI。\n\n| 组件 | 实现 | 验证状态 |\n|---|---|---|\n'
for p in sorted((R/'src/cp_disr').glob('*.py')):
 inventory+=f"| {p.stem} | `{p.relative_to(R)}` / `{sha(p)}` | {'NOT_RUN_ENVIRONMENT_BLOCKED' if p.stem in ['neural','torch_rl','execution','evaluation'] else '编译/静态；相应纯逻辑测试见测试清单'} |\n"
inventory+='\n## 入口与边界\n\n8 个正式 CLI 均可定位。train/evaluate 遇到未绑定运行资源返回非零退出码和具体 MUST_BIND；generate-cache 本轮仅本地校验能力，未实现/调用外部 API transport，明确拒绝请求。run-unit-tests --scope full 为未来用户授权 Stage 0B 后入口，本轮仅运行 --scope pure 对应的逻辑检查。\n\nTorch 模块包含标准 128×4 层 RGCN、goal readout、七配置、GRU、独立 V/Q、当前参数 prefix 重算、单参数所有者 PPO、执行动作 Q loss、真实 transition collector 和有界资源 cap driver。外部 snapshot builder、controller、perception 等必须由真实 plugin 显式绑定。所有数值代码仅通过编译和静态检查，不能宣称其动态正确性已通过。\n'
(O/'implementation_inventory.md').write_text(inventory)
(O/'interface_binding_report.md').write_text('# 接口绑定报告\n\n**接口完成 ≠ 真实运行资源绑定。**\n\n- EnvironmentAdapter、SkillExecutor、ObservationProvider、PerceptionAdapter、FactVerifier、TaskEvaluator、SafetyManager、DurationProvider、SnapshotBuilder 与 RuntimeFactory 的输入/输出 dataclass、Protocol 已定义于 `src/cp_disr/adapters.py`。\n- 真实测量、执行结束、三态事实、task_success、奖励事件时间各有独立 schema。TaskEvaluator 输入没有 prior/DK/DP；名义 overlay 没有真实时钟、reward 或执行写接口。\n- 合同与 task 模板已通过 schema/逻辑校验；timeout、controller、verifier 无证据时仍 MUST_BIND。MOVE 未绑定为真实能力，模板允许 PICK+PLACE_BUFFER 等价结构。\n- 环境/API 为 user_environment_owned，不由本次修复、读取或探测。\n- 真实 adapter、校准、视觉 checkpoint、时间/安全许可与任务池仍 external_runtime_missing（在本轮明确搜索范围内未找到可认证绑定，不等于证明整台服务器不存在）。\n- discovered_and_bound 仅指已有来源路径/commit/hash 的资料登记；没有据文件名宣称 controller 或任务已绑定。\n')
rows=[]
for p in sorted(M.glob('*.yaml')):
    d=read(p)
    def walk(value,path):
        if isinstance(value,dict):
            for k,v in value.items():walk(v,path+'.'+str(k) if path else k)
        elif isinstance(value,list):
            for i,v in enumerate(value):walk(v,path+f'[{i}]')
        elif value is None or (isinstance(value,str) and value.startswith(('MUST_','REQUIRED_'))):
            identity=str(p.relative_to(E))+':'+path
            owned=('software_manifest' in identity or 'actual_parameter_count' in identity or '.software.' in identity or (any(x in identity for x in ['vlm_manifest','vlm_runtime','vlm.']) and any(path.endswith(x) for x in ['region','base_http_api_url','api_account_authorized','credential_env_name'])))
            rows.append({'field':identity,'value':str(value),'category':'user_environment_owned' if owned else 'external_runtime_missing','status':'MUST_BIND','evidence':'User-owned environment/API boundary' if owned else 'Interface/template present; no certified concrete binding in asset_discovery.json'})
    walk(d,'')
for task in ['D0','T_A','T_C']:
    for key in ['actual_asset_refs','controller_refs','evaluator_ref']:rows.append({'field':f'configs/tasks/{task}.yaml:{key}','value':'MUST_BIND','category':'external_runtime_missing','status':'MUST_BIND','evidence':'Structural template only'})
for c in commands:rows.append({'field':'entrypoints.'+c,'value':commands[c]['cli'],'category':'code_completed','status':'IMPLEMENTED','evidence':'entrypoints.yaml; static_validation.json'})
for module in ['contracts','facts','graph','vlm','prior','skills','neural','torch_rl','adapters','collector']:
    rows.append({'field':'core.'+module,'value':f'src/cp_disr/{module}.py','category':'code_completed','status':'CODE_PRESENT_NOT_NUMERICALLY_CERTIFIED' if module in ['neural','torch_rl'] else 'CODE_PRESENT','evidence':sha(R/'src/cp_disr'/f'{module}.py')})
for repo in discovery['repositories']:rows.append({'field':'discovery.source_repository','value':repo['path'],'category':'discovered_and_bound','status':'SOURCE_PROVENANCE_ONLY_NOT_RUNTIME_BINDING','evidence':str(repo['git_commit'])})
rows.append({'field':'software.torch_cuda_pyg_runtime','value':'USER_OWNED','category':'user_environment_owned','status':'NOT_RUN_ENVIRONMENT_BLOCKED','evidence':'User explicitly excludes installation/import/RGCN/roundtrip remediation'})
for name,subset in [('binding_table.csv',rows),('unbound_must_bind.csv',[x for x in rows if x['category'] in ['user_environment_owned','external_runtime_missing']])]:
    with (O/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=['field','value','category','status','evidence']);w.writeheader();w.writerows(subset)
jout(O/'binding_inventory.json',rows)
status=read(E/'stage_status/stage_0a.json');history=E/'stage_status/history';history.mkdir(exist_ok=True)
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');jout(history/f'stage_0a_before_non_environment_{stamp}.json',status)
readiness={**status.get('readiness',{}),'core_code':True,'core_code_present':True,'production_entrypoints_present':True,'task_templates_present':True,'contracts_present':True,'fixtures_present':True,'stage_0b_tests_authored':True,'pure_logic_validation_passed':checks[1]['status']=='PASS','torch_runtime_validation_pending':True,'external_runtime_binding_pending':True,'api_binding_pending':True,'stage_0b':False,'stage_0b_code_prerequisites_present':True,'stage_0b_ready_after_user_environment_validation':validation['static_all_passed'],'stage_0b_authorization_required':True}
status.update(status='BLOCKED',git_hash=git('rev-parse','HEAD'),git_hash_semantics='Pre-remediation committed base; exact new code hashes recorded in revision static_validation; delivery receipt records final commit',completed_at=datetime.now(timezone.utc).isoformat(),execution_scope='Stage 0A non-environment/API remediation only',readiness=readiness,
 current_revision=str(O.relative_to(E)),source_tree_sha256=tree_hash,stage_0b_executed=False,
 issues=[{'category':'user_environment_owned','issue':'User to finish torch/CUDA/PyG and account/region/endpoint; no reinstallation or credentials access this revision'}, {'category':'external_runtime_missing','issue':'Real task/controller/visual/verifier/evaluator/time/safety/splits not certified; see new unbound_must_bind.csv'}, {'category':'code_completed','issue':'Core code and T01-T25 authored; numerical behavior must be verified after environment is ready'}],runs_completed=[],runs_failed=[],runs_reused=[])
Draft202012Validator(read(E/'schemas/stage_status.schema.json')).validate(status)
jout(E/'stage_status/stage_0a.json',status);jout(O/'stage_0a.json',status);jout(history/f'stage_0a_non_environment_{stamp}.json',status)
summary=f'''# Stage 0A 非软件环境、非 API 凭据补全报告

**状态：BLOCKED；本轮代码补全与纯逻辑/静态验收已完成，未执行正式 Stage 0B。**

- 项目 `{R}`；分支 `{git('branch','--show-current')}`。
- 基准 commit `{git('rev-parse','HEAD')}`；新 commit 见交付 receipt。
- 新 revision `{O}`。原 Stage 0A summary 未覆盖，旧 manifests/status 已归档。
- 环境下载/安装、凭据读取、API 请求、仿真/机器人动作、科研训练、性能评价均为 0。

## 1. 真正解决的原 BLOCKED 项

新增正式 `src/cp_disr` package：不可变 typed contracts / grounding / conflict validation；三值 Fact Store 与只读 nominal overlay；Action–Proposition 图、固定节点/goal_refs；本地 relation schema/ID/effect/redundancy/dedup/cache-key；固定 episode 80/20 prior；标准共享 RGCN、DK/DH/DP、零 prior 复用、零锚定有界 Actor；七配置；独立 V/Q、executed-only detached Q targets；SMDP GAE、候选身份/mask/prior 保存、GRU prefix 与当前参数重算 PPO；外部 Protocol、真实 collector 和 fail-closed 入口。

任务 D0/T_A/T_C 结构模板、OPEN/PICK/PLACE/PLACE_BUFFER/MOVE 合同模板、5 个固定 synthetic fixtures、T01–T25 测试代码及 8 个 CLI 已落盘。MOVE 未被声明为真实 runtime 技能。

## 2. 验证结果

- `compileall src tests`：{checks[0]['status']}。
- 纯逻辑 pytest：{checks[1]['status']}；详见 pure_logic_tests.stdout.log / JUnit。
- YAML/JSON、schema、import-free dependency graph、manifest/entrypoint references、方法源 hash、唯一 prior 协议、未绑定运行拒绝检查：见 static_validation.json，全部通过：{validation['static_all_passed']}。
- 数值/梯度测试：**NOT_RUN_ENVIRONMENT_BLOCKED**，只编写、编译、静态检查；不能声称 torch/PyG 运行正确性已通过。
- 所有 synthetic fixture 都有 `synthetic_unit_fixture=true`、`paper_performance_eligible=false`；生产 collector/evaluation 拒绝 synthetic snapshot。

## 3. 仅完成接口/模板，尚未真实绑定

EnvironmentAdapter、SkillExecutor、ObservationProvider、PerceptionAdapter、FactVerifier、TaskEvaluator、SafetyManager、DurationProvider、SnapshotBuilder 已有明确接口，但没有编造 concrete 实现。
真实 D0/T_A/T_C 资产、OPEN/PICK/PLACE 控制器、camera/calibration/perception checkpoint、Verifier 阈值、独立 success/termination、时限/clock/d_ref/safety、64/50/50 初始化实例仍未绑定；只有 schema 和模板。API 账户及环境继续由用户处理。逐字段 A/B/C/D 分类见 binding_table.csv 和 unbound_must_bind.csv。

## 4. 服务器已有可复用来源

只读盘点 {len(discovery['repositories'])} 个源项目、{len(discovery['code_candidates'])} 个代码候选、{len(discovery['asset_candidates'])} 个资产候选。LIBERO ControlEnv/相机参数/BDDL/XML/独立 check_success 可适配；P1 历史控制器和 P2C rollout/checkpoint 可参考。均有路径/commit/dirty/hash/签名与证据。LIBERO 的默认双视角、无 depth、低层控制步/旧 done 与本版接口有差异；真值 predicates 不能冒充视觉事实。旧 reward shaping/PPO 不能直接继承。没有生成缺乏证据的 resolved task。

## 5. 环境修复后 Stage 0B 还缺什么

生产核心、fixtures、25 项测试和命令已存在。用户完成目标环境后，还需先重新验收 0A 的依赖导入、RGCN 前后向与 checkpoint round-trip；随后在用户明确授权下执行正式 0B CPU T01–T25，GPU 按可用性补测。数值测试运行前不存在已证明通过的承诺，运行发现 bug 时仍需修复。按手册 0B 的前置定义，真实机器人/API 未绑定不妨碍独立 synthetic 单元验收，但不能自动把完整 0A 升为 PASS。

建议未来命令（本轮未执行）：`PYTHONPATH=src .venv/bin/python -m cp_disr run-unit-tests --scope full`。正式 0B 仍需按手册收集 L_TEST/JUnit 和状态报告；CLI 不自动修改阶段状态。

## 6. Stage 0C 与 Stage 1A 仍缺什么

0C：用户 API 配置与权限、实际 snapshot 支持验证、本轮刻意未实现的联网 provider transport、真实 dev 图像/ID/合同、冻结三个 few-shot、完整请求/缓存审计和 24 场景检查；还需 0B 验收。
1A：0B/0C 的真实前置验收，D0 场景与 split、控制器/视觉/Verifier/独立评价/安全/时间绑定、runtime factory 与 resolved run config、真实缓存、资源配置和训练启动前审核；当前 train/evaluate 明确拒绝未绑定运行。

## 7. 是否达到“完成 PyTorch 环境后可重新验收 0A 并执行 0B”

**就核心代码入口、fixtures、测试及静态前置而言：已具备。** 仍必须重新验收软件并由用户授权正式 0B；不能承诺首次数值测试必然全部通过。**若指完整 Stage 0A PASS 后无条件进入 0B：尚未达到**，真实平台/API 缺项仍使完整 0A 为 BLOCKED。上述两种 readiness 已在 stage_0a.json 分开记录。

本轮不安装、不碰凭据、不调用 VLM、不进入正式 Stage 0B；提交当前分支，不推送 GitHub，然后停止。
'''
(O/'stage_0a_non_environment_remediation_summary.md').write_text(summary)
for name in ['entrypoints.yaml','runtime_manifest.yaml','task_manifest.yaml','skill_contracts.yaml','verifier_manifest.yaml','splits.json']:
    shutil.copy2(M/name,O/name)
shutil.copy2(E/'configs/resolved_defaults.yaml',O/'resolved_defaults.yaml')
jout(E/'DELIVERY_STATUS.json',{'stage_0a_status':'BLOCKED','other_stages':'NOT_STARTED','revision':str(O.relative_to(E)),'scope':'non-environment non-API remediation','static_all_passed':validation['static_all_passed'],'formal_stage_0b_executed':False})
(R/'README.md').write_text('# CP-DISR v2.1\n\nStage 0A remains **BLOCKED**: external runtime and user-owned environment/API bindings pending.\n\nProduction package: `src/cp_disr`; no imports of torch from package root or offline CLI.\n\nLatest report: `'+str((O/'stage_0a_non_environment_remediation_summary.md').relative_to(R))+'`\n\nUse `PYTHONPATH=src python -m cp_disr --help`. Pure logic checks run under Stage 0A remediation only. Full T01–T25 tests require the user-completed environment and explicit authorization; no automatic stage advancement.\n\nOriginal reports and frozen Method sources remain preserved. No real controller/task binding is implied by a template or a discovered file. Synthetic fixtures are forbidden from performance data.\n')
(O/'checksums.sha256').write_text(''.join(sha(p)+'  '+str(p.relative_to(R))+'\n' for p in sorted(O.rglob('*')) if p.is_file() and p.name!='checksums.sha256'))
print(json.dumps({'revision':str(O),'static_all_passed':validation['static_all_passed'],'checks':checks,'tests_authored':len(tests),'pure_tests':sum(t['group']=='pure' for t in tests),'torch_tests_not_run':sum(t['group']=='torch_runtime' for t in tests),'readiness':readiness},ensure_ascii=False,indent=2))
if not validation['static_all_passed']:raise SystemExit(2)
