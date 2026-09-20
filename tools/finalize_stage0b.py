"""Build Stage 0B reports from preserved pytest evidence; no tests or API calls."""
import ast,csv,hashlib,json,os,platform,re,subprocess,xml.etree.ElementTree as ET
from datetime import datetime,timezone
from pathlib import Path
import importlib.metadata as metadata
import yaml

R=Path('/home/__compress_data/xushijie/graph_cp_disr_v2_1');O=R/'experiments/part_0_validation/stage_0b';A=O/'attempts'
BASE='3669e5a11e59b4b327db714f092b2d9ca14d78d3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def git(*args):return subprocess.check_output(['git','-C',str(R),*args],text=True).strip()

expected={
'T01':'Empty prior explicitly reuses K encodings; every DP element is exactly zero',
'T02':'DP=0 gives exactly zero uP and residual; same-parameter contract distribution agrees',
'T03':'Nominal overlays reject writes; real Fact Store/snapshot hashes and metadata unchanged',
'T04':'Four graph views retain identical node IDs/order/types/bindings and goal references',
'T05':'Goal signs, completed goals, reordered rows and padding readout remain aligned',
'T06':'UNKNOWN satisfies neither signed requirement; conditional UNKNOWN keeps only common effects',
'T07':'ADD/DEL/UNKNOWN and exclusive Held conflicts raise instead of overwriting',
'T08':'Only registered ADD/DEL effect anchors and valid IDs/types survive; redundancy and duplicates removed',
'T09':'Schema rejects ordering relation/missing anchor; production registry has no ordering relation',
'T10':'Old/new canonical candidate IDs and masks agree; changed masks and length mismatches rejected',
'T11':'Only executed Q output slots receive direct gradient; all other output gradients exactly zero',
'T12':'Q/V targets have no grad_fn or requires_grad; no old-value target gradient; FP64 manual values match',
'T13':'One shared encoder retains all four autograd paths; actor/V/Q gradients finite and nonzero',
'T14':'Termination removes bootstrap; truncation/buffer boundaries use final valid state; no reset leakage',
'T15':'Candidate/object/goal permutations preserve identity-mapped outputs; exact ties use canonical ID',
'T16':'Semantic cache-key changes isolate inputs/splits; unordered JSON key order does not change key',
'T17':'One independent RNG draw per episode; prior identity/hash fixed across rollout boundaries',
'T18':'Production PPO recomputes encoder under current parameters across four epochs; zero API/prior sampling',
'T19':'Duration Gamma, discounted event reward and episode-start weights match FP64 calculations',
'T20':'Prior cannot alter facts/candidates/masks or enter evaluator/base GRU/contract context',
'T21':'Bounded residual <=0.5 and probability ratio within exp(+/-1); exact zero anchor',
'T22':'Read-only value probes; prefix recompute/current params; environment/episode/prior identity isolation',
'T23':'Four diagnostic linear witnesses plus nondegenerate production soft-path DP response/gradients',
'T24':'No candidates yields independent safe ending without Categorical; registered WAIT has empty patch',
'T25':'A_DD uses DH (zero if no prior); A_Q changes only Q coefficient; A_B removes only tanh retaining B'}
components={
'T01':'neural.differences;graph.four_views','T02':'neural.AnchoredPrior;Policy',
'T03':'contracts.nominal_overlay;facts.FactStore','T04':'graph.four_views',
'T05':'neural.GoalReadout;CandidateReadout','T06':'facts;contracts.precondition_value;nominal_overlay',
'T07':'contracts.Effects;nominal_overlay','T08':'vlm.validate_relations','T09':'graph.RELATIONS;vlm.validate_relations',
'T10':'rl.remap_stored;neural.Policy','T11':'torch_rl.q_loss','T12':'torch_rl.q_targets;value_targets',
'T13':'neural.GraphEncoder;differences;Policy','T14':'rl.Transition;scalar_targets',
'T15':'neural.Policy;PolicyOutput.select','T16':'vlm.cache_key','T17':'prior.PriorSampler',
'T18':'torch_rl.PPO.update;recompute_transition','T19':'rl.gamma;interval_reward;cumulative_weights',
'T20':'adapters.EvaluationInput;skills.candidate_mask;neural.Policy','T21':'neural.AnchoredPrior',
'T22':'torch_rl.RecurrentState;prefix_hidden','T23':'neural.differences;GraphEncoder',
'T24':'contracts;neural.Policy','T25':'neural.Policy;torch_rl.ppo_losses'}

sources=[p for p in (R/'src').rglob('*.py')]
tests=[p for p in (R/'tests').glob('*.py')]
codehash={str(p.relative_to(R)):sha(p) for p in sorted(sources+tests+[R/'tools/stage0b_plugin.py'])}
treehash=hashlib.sha256(json.dumps(codehash,sort_keys=True).encode()).hexdigest()
save(O/'code_hashes.json',{'base_commit':BASE,'code_tree_sha256':treehash,'files':codehash,'commit_semantics':'The worktree source digest identifies exact tested content; final Git commit is recorded in delivery_receipt.json'})
schemas=[p for p in (R/'schemas').rglob('*.json')]
save(O/'schema_hashes.json',{str(p.relative_to(R)):sha(p) for p in schemas})
fixtures=[]
for p in sorted((R/'tests/fixtures').glob('*.json')):
    d=json.loads(p.read_text());assert d['synthetic_unit_fixture'] is True and d['paper_performance_eligible'] is False
    fixtures.append({'path':str(p.relative_to(R)),'sha256':sha(p),'synthetic_unit_fixture':True,'paper_performance_eligible':False})
save(O/'fixture_manifest.json',fixtures)
initial=json.loads((O/'initial_hashes.json').read_text())
frozen={k:sha(R/k)==v for k,v in initial.items() if k.startswith(('tests/fixtures/','schemas/','configs/','experiments/sources/','experiments/configs/','experiments/stage_cards/'))}
assert all(frozen.values())
assert (R/'experiments/stage_status/stage_0a.json').read_bytes()==(O/'stage_0a_before.json').read_bytes()
assert json.loads((R/'experiments/stage_status/stage_0a.json').read_text())['status']=='BLOCKED'
production_audit=[]
for p in tests:
    tree=ast.parse(p.read_text());assert not any(isinstance(n,ast.Assert) and isinstance(n.test,ast.Constant) and n.test.value is True for n in ast.walk(tree))
    production_audit.append({'test_file':str(p.relative_to(R)),'production_imports':sorted({n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom) and n.module and n.module.startswith('cp_disr')}),'shared_fixtures':'tests/conftest.py uses production build_fixture and Policy','sha256':sha(p)})
conf=yaml.safe_load((R/'experiments/configs/resolved_defaults.yaml').read_text())
assert conf['prior_training']['original']==.8 and conf['prior_training']['absent']==.2 and conf['prior_training']['resample_during_ppo'] is False
assert not any('SOFT_ORDER' in p.read_text() for p in sources)
assert 'relation_rewrite' not in json.dumps(conf['prior_training'])
save(O/'production_component_audit.json',{'tests':production_audit,'fixture_hashes_unchanged':True,'frozen_inputs_unchanged':frozen,'no_assert_true':True,'gpu_only_skip_guard':'Only standalone device parity may skip when CUDA is absent; current device available and no final test skipped','prior_protocol':conf['prior_training'],'stage_0a_unchanged':True})

final_runs=[('009_final_cpu','cpu','FP32/logic'),('010_final_fp64','cpu','FP64'),('006_gpu_fp32','cuda:0','FP32'),('008_parity_artifact_fallback','cpu+cuda:0','FP32')]
all_rows=[];final_rows=[];counts={};combined=ET.Element('testsuites')
test_source={}
for p in tests:
    for n in ast.parse(p.read_text()).body:
        if isinstance(n,ast.FunctionDef) and n.name.startswith('test_'):test_source[n.name]=(p,n.lineno)
def read_run(path,device,dtype,final):
    root=ET.parse(path).getroot();rows=[]
    for case in root.iter('testcase'):
        name=case.attrib['name'];ids=sorted(set(re.findall(r'T\d{2}',name))) or ['GATE']
        status='FAIL' if case.find('failure') is not None or case.find('error') is not None else 'SKIP' if case.find('skipped') is not None else 'PASS'
        failure=case.find('failure')
        if failure is None:failure=case.find('error')
        actual_dtype='FP64' if 'fp64' in name else dtype
        source,line=test_source.get(name,(Path('unknown'),0))
        try:source_ref=str(source.relative_to(R))
        except ValueError:source_ref=str(source)
        row={'test_id':'+'.join(ids),'test_name':name,'production_component':'; '.join(components.get(i,'runtime.require_runtime;require_cache_configuration') for i in ids),
          'fixture':'tests/fixtures/*.json (hashes in fixture_manifest.json); synthetic inline inputs in test source',
          'device':device,'dtype':actual_dtype,'expected':'; '.join(expected.get(i,'Unbound runtime/API rejected without credentials') for i in ids),
          'actual':'All production assertions passed; see test source and JUnit evidence' if status=='PASS' else (failure.text if failure is not None else 'Skipped'),
          'tolerance':('atol=1e-10 rtol=1e-9' if actual_dtype=='FP64' else 'atol=1e-6 rtol=1e-5; logic/zero/hash/detach exact'),
          'status':status,'failure_location':str(failure.attrib.get('message','')) if failure is not None else '',
          'attempt':path.parent.name,'code_commit':BASE,'notes':'Base commit plus recorded worktree code hash/diff; final commit in delivery receipt. '+('Final acceptance run.' if final else 'Historical attempt; superseded.'),
          'evidence':str(path.relative_to(R)),'test_source':source_ref+':'+str(line),'duration_seconds':case.attrib.get('time'),'final_acceptance':final}
        rows.append(row)
    return root,rows
for name,dev,dtype in final_runs:
    xml=A/name/'junit.xml';root,rows=read_run(xml,dev,dtype,True);final_rows+=rows
    counts[name]={k:sum(r['status']==k for r in rows) for k in ['PASS','FAIL','SKIP']}
    for suite in root.iter('testsuite'):
        suite.set('name',name+'/'+suite.attrib.get('name','pytest'));combined.append(suite)
for path in sorted(A.glob('*/*.xml')):
    name=path.parent.name
    if name in {x[0] for x in final_runs}:continue
    dev='cpu+cuda:0' if 'parity' in name else 'cuda:0' if 'gpu' in name else 'cpu';dtype='FP64' if 'fp64' in name else 'FP32/logic'
    _,rows=read_run(path,dev,dtype,False);all_rows+=rows
all_rows+=final_rows
assert all(r['status']=='PASS' for r in final_rows)
id_status={tid:'PASS' if any(tid in r['test_id'].split('+') and r['device']=='cpu' for r in final_rows) else 'BLOCKED' for tid in expected}
assert set(id_status.values())=={'PASS'}
ET.ElementTree(combined).write(O/'junit.xml',encoding='utf-8',xml_declaration=True)
fields=['test_id','test_name','production_component','fixture','device','dtype','expected','actual','tolerance','status','failure_location','attempt','code_commit','notes','evidence','test_source','duration_seconds','final_acceptance']
with (O/'test_results.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(all_rows)
parity=json.loads((A/'008_parity_artifact_fallback/numeric_comparison.json').read_text())
parity_summary={'tensor_comparisons':len(parity),'max_absolute_error':max(x['max_absolute_error'] for x in parity),'max_relative_error':max(x['max_relative_error'] for x in parity),'failed_tensor_comparisons':sum(x['status']!='PASS' for x in parity),'failure_locations':[x for x in parity if x['failure_locations']],'relative_denominator':'max(abs(cpu reference), 1e-12)','comparison_rule':'abs(gpu-cpu) <= 1e-6 + 1e-5*abs(cpu), applied elementwise; near-zero relative errors may exceed rtol while still satisfying atol'}
save(O/'test_results.json',{'stage':'0B','status':'PASS','code_tree_sha256':treehash,'per_test_id':id_status,'final_counts':counts,'history_included':True,'results':all_rows,'device_comparison':parity_summary})
save(O/'device_comparison_summary.json',parity_summary)
import torch
env={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'host':platform.node(),'python':platform.python_version(),'executable':str(R/'.venv-stage0a/bin/python'),'packages':{p:metadata.version(p) for p in ['torch','torch-geometric','numpy','pytest','jsonschema','PyYAML']},'torch_runtime':torch.__version__,'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'driver_inventory':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,driver_version','--format=csv,noheader'],text=True),'tf32_in_test_plugin':False,'seed_protocol':'seed32(unit_fixture,test_name,0,0); policy fixture seed32(unit_fixture,contracts_v1,0,0)','legacy_initial_policy_seed':7,'scientific_training_updates':0,'synthetic_dummy_optimizer_updates':'T18 only: permitted test assertions, not RL results','api_requests':0,'robot_actions':0,'performance_episodes':0,'stage_0c_executed':False}
save(O/'environment_snapshot.json',env)
fixes=json.loads((A/'003_identity_fix/root_cause.json').read_text())
save(O/'attempt_manifest.json',{'fixes':fixes,'attempts':[{'directory':str(p.relative_to(R)),'files':[x.name for x in p.iterdir() if x.is_file()]} for p in sorted(A.iterdir()) if p.is_dir()],'initial_test_failures':2,'all_final_required_checks_passed':True})
gate=json.loads((O/'logs/runtime_gate_checks.json').read_text());assert all(x['passed'] for x in gate)
summary=f'''# Stage 0B — Unit & Invariant Tests

**最终状态：PASS。Stage 0A 保持 BLOCKED；Stage 0C 未启动。**

本报告只验收生产组件公式、权限与数据链，不含任务性能结果。项目 `{R}`，分支 `{git('branch','--show-current')}`，基准提交 `{BASE}`；精确测试代码 SHA-256 见 code_hashes.json，最终提交见交付 receipt。

## 最终运行结果

- T01–T25：25/25 PASS。完整 CPU 回归：37 passed、0 failed、0 skipped（其中 T24 与若干增强回归有多个测试函数，另含 runtime gate）。
- CPU FP64 手算/数值补充：7 passed、0 failed、0 skipped。
- GPU FP32 数值/梯度子集：22 passed、0 failed、0 skipped；15 个纯逻辑/CPU 项按选择器未在 GPU 重复执行，属于不适用而非缺失 GPU 验证。
- CPU/GPU 输出及梯度对照：1 passed；比较 {len(parity)} 个张量，最大绝对误差 {parity_summary['max_absolute_error']:.12g}，最大相对误差 {parity_summary['max_relative_error']:.12g}；超容差位置 0。采用逐元素 `abs(error) <= 1e-6 + 1e-5*abs(CPU)`，接近零的参考值会放大相对误差；完整逐张量记录保存在 attempts/008_parity_artifact_fallback/numeric_comparison.json。
- 原始套件首次三组为 9/9/8 passed；增强回归曾发现 2 个错误，原日志与 traceback 保留。修复后受影响 4 项通过，再完整 CPU 回归及适用 GPU 补测全部通过。历史失败没有删除或伪装为首次通过。

## 不变量结论

没有发现真实 Fact Store 污染；nominal overlay 写入拒绝、真实快照 hash 不变。空 prior 的 DP、uP、Delta 严格为零，使用同参数合同分布对照；未以容差掩盖零性质。candidate ID/mask 重算一致，并主动拒绝长度、集合、值变化。Structural Q 只 gather executed action，未选输出直接梯度严格为零；Q/V target 完全 detach。

prior 只在 episode 开始抽样一次，固定 80/20，episode 内 hash 不变；生产 PPO 在四个 epoch 重算 encoder/差分，参数版本变化，禁止网络及 prior 重抽。terminated 不 bootstrap，truncated/buffer 边界使用最后有效状态，不跨 reset。GRU probe 不提交状态，prefix 使用当前参数，环境与 episode 隔离。Full/A_DD/A_Q/A_B 分别验证规定输入、Q 系数和 tanh 开关，未改变其他参数。

## 修复的生产 bug

1. `rl.remap_stored` 在 zip 前未校验 mask 长度，多余 mask 元素被截断后可能被接受。已添加长度一致性拒绝。
2. `torch_rl.RecurrentState` 未检查 snapshot 的 env/episode/prior 身份。已在 commit/probe 前检查环境、决策连续性、episode 与 prior；错误输入在状态写入前拒绝。

这两项仅修复实现权限/数据完整性，不改变冻结 Method、公式、schema、fixture 语义或超参数。全部修复前日志、根因与 diff 位于 attempts/。

## Stage 0C / 1A readiness

Stage 0B 的 schema、ID/effect/redundancy/cache-key 单元前置已通过。但完整 Stage 0C 执行代码与真实输入还不齐：联网 provider transport 仍未绑定；D0/T_A/T_C 各 8 张真实 dev 图像及对象/合同绑定、冻结 3 个 few-shot、请求/原始输出/拒绝/缓存审计尚需完成。API 已由用户指定北京 endpoint `https://dashscope.aliyuncs.com/api/v1`；`qwen3.8-max-0902` 权限及图像输入、非思考、JSON object、temperature=0 等真实行为为 MUST_VERIFY，仅可在后续授权 Stage 0C 验证；不可访问即停止，不自动换模型或 endpoint。

Stage 1A 仍缺真实任务和 split、控制器、相机/标定、冻结感知、Verifier 阈值、独立 evaluator、安全许可、clock/d_ref/timeout/deadline、runtime factory 与 resolved run config、有效缓存和 Stage 0C 验收。train/evaluate 在未绑定时仍 fail closed；本轮只测试拒绝入口，未启动任务动作。

已更新 stage_0b.json；Stage 0A 文件逐字节未变。0 科研训练、0 API 请求、0 机器人/仿真任务动作、0 性能 episode。未推送 GitHub；停止于 Stage 0B。
'''
(O/'stage_0b_summary.md').write_text(summary)
table='\n'.join('| '+tid+' | PASS | '+components[tid]+' |' for tid in expected)
(O/'invariant_report.md').write_text(summary+'\n## T01–T25 状态\n\n| ID | 状态 | 生产组件 |\n|---|---|---|\n'+table+'\n')
status=json.loads((O/'stage_0b_before.json').read_text());status.update(status='PASS',git_hash=BASE,git_hash_semantics='Base plus code_hashes.json; final commit in delivery receipt',started_at=datetime.fromtimestamp((A/'001_initial_cpu').stat().st_mtime,timezone.utc).isoformat(),completed_at=datetime.now(timezone.utc).isoformat(),execution_scope='Synthetic production unit/invariant tests only; no scientific training/API/task actions',selected_tasks=['T%02d'%i for i in range(1,26)],selected_config='configs/resolved_defaults.yaml',runs_completed=[{'attempt':n,**counts[n]} for n,_,_ in final_runs],runs_failed=[{'attempt':'002_extended_cpu','failed':2,'fixed_in':'003_identity_fix','preserved':True}],runs_reused=[],issues=[],tuning_changes=[],code_tree_sha256=treehash,readiness={'stage_0b_passed':True,'stage_0a_status':'BLOCKED','stage_0c_schema_unit_prerequisites':True,'stage_0c_runtime_code_and_inputs_ready':False,'stage_0c_authorization_required':True,'stage_1a_runtime_ready':False},next_stage='0C',stage_0c_executed=False)
save(R/'experiments/stage_status/stage_0b.json',status)
save(O/'stage_0b.json',status)
save(R/'experiments/DELIVERY_STATUS.json',{'stage_0a_status':'BLOCKED','stage_0b_status':'PASS','later_stages':'NOT_STARTED','latest_report':'part_0_validation/stage_0b/stage_0b_summary.md','stage_0c_executed':False})
(R/'README.md').write_text('# CP-DISR v2.1\n\nStage 0A: **BLOCKED** on real runtime bindings.\nStage 0B: **PASS** for production synthetic unit/invariant tests.\nStage 0C: **NOT_STARTED**; provider and real scene/few-shot inputs pending.\n\nLatest report: `experiments/part_0_validation/stage_0b/stage_0b_summary.md`.\n\nRuntime: `PYTHONPATH=src .venv-stage0a/bin/python -m cp_disr --help`.\nNo training, VLM request or robot/task action is authorized by these test results.\n')
print(json.dumps({'status':'PASS','counts':counts,'device_comparison':parity_summary,'stage_0a_unchanged':True},indent=2))
