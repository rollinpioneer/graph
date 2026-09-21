from pathlib import Path
import json,csv,hashlib,datetime,subprocess,yaml
ROOT=Path(__file__).resolve().parents[1]; F=ROOT/'experiments/part_0_validation/stage_0c'; now=datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,s):
 p=F/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s,encoding='utf-8')
def dump(name,x):write(name,json.dumps(x,ensure_ascii=False,indent=2)+'\n')
# preserve previous states byte-for-byte
assert (F/'previous_stage_0a.json').read_bytes()==(ROOT/'experiments/stage_status/stage_0a.json').read_bytes()
assert (F/'previous_stage_0b.json').read_bytes()==(ROOT/'experiments/stage_status/stage_0b.json').read_bytes()
status=json.loads((ROOT/'experiments/stage_status/stage_0c.json').read_text())
platform='''# Stage 0C-R1 平台选择审计

## 结论

本轮唯一推荐平台为 **LIBERO + robosuite + MuJoCo**，但当前只能记为 `CANDIDATE_BLOCKED`，没有任何平台达到 CP-DISR 正式绑定的最低条件。Stage 0C 继续 `BLOCKED`，不生成正式 24 场景或 few-shot 图像。

## 审计对象

| 项目 | 绝对路径 | git commit | dirty | 结果 |
|---|---|---|---|---|
| LIBERO | `/home/__compress_data/xushijie/LIBERO` | `8f1084e3132a39270c3a13ebe37270a43ece2a01` | dirty（多处源码/配置修改） | 唯一推荐候选 |
| Metaworld | `/home/__compress_data/xushijie/Metaworld` | `a98086ababc81560772e27e7f63fe5d120c4cc50` | 未在本轮作为目标平台绑定 | 不匹配容器/合同结构 |
| CP-DISR仓库 | `/home/__compress_data/xushijie/graph_cp_disr_v2_1` | `7d8ef02d984a8266b39bf7c2facd138761127834` | clean at audit start | 冻结协议和待绑定模板 |

## 已确认能力

LIBERO 仓库包含本地 BDDL、对象/场景 MuJoCo XML 和 MIT LICENSE；固定 `agentview` 相机接口可产生 RGB 和 depth。使用 `/home/__compress_data/xushijie/.conda/envs/lerobotpi0/bin/python`、本地 LIBERO 资产和固定 BDDL 做了只读 `reset()` 探针：KITCHEN_SCENE3 产生 `96x96x3` RGB 与 `96x96x1` depth。该探针不是正式场景采集，没有写入正式输入目录，也没有 VLM/API 请求。LIBERO 资产缓存路径 `/home/__compress_data/xushijie/.cache/libero/assets` 已存在；未下载网络图片或生成图片。

## 未满足的最低条件

- LIBERO/robosuite 现有接口是低层 `OSC_POSE` 7-DoF action，不是 CP-DISR `OPEN`, `PICK`, `PLACE`, `PLACE_BUFFER` 或 `MOVE` 高层技能。
- 没有已绑定的高层 controller、技能后置条件 verifier、独立 success verifier、TaskEvaluator、安全许可、skill timeout/deadline 或参考技能秒数。
- 现有 D0/T_A/T_C 仍是 `STRUCTURE_TEMPLATE_NOT_RUNTIME_ASSET`，其 `controller_ref`, `verifier_ref`, `evaluator_ref`, asset 和 split 均为 MUST_BIND。
- LIBERO 工作树 dirty；不能把未审计修改后的平台状态作为冻结实验平台版本。
- LIBERO LICENSE 证明代码许可，不自动证明所有 bundled asset 的 VLM 上传权限；每个正式 asset 仍需逐项权限证据。

## 精确实施清单

1. 在最终选定且干净的 LIBERO commit 上冻结版本和资产清单，或提供另一个已具备高层技能的目标平台。
2. 实现并审计 CP-DISR `OPEN/PICK/PLACE/PLACE_BUFFER`（或真实 `MOVE`）controller adapter；每个 Action ID 写入 controller manifest。
3. 实现 postcondition verifier、independent success verifier、TaskEvaluator、安全授权、clock unit、timeout/deadline 和 reference skill seconds。
4. 为 D0/T_A/T_C 写真实 task BDDL/scene reset 配置，记录 asset hashes、对象位姿、buffer 区域和 camera config。
5. 逐项确认 bundled asset 的使用与上传许可，建立独立 dev_fewshot pool 和 test split 隔离。
6. 只在上述绑定完成后运行无 API 的 27 输入预检；随后用户单独处理凭证和模型权限。
'''
write('platform_selection_report.md',platform)
write('stage_0c_resource_binding_summary.md','''# Stage 0C-R1 资源绑定摘要\n\n状态：**BLOCKED**。本轮完成了平台候选审计和一次只读 reset/render 能力探针，确认 LIBERO 的本地资产、MuJoCo、robosuite 与固定相机接口存在。由于缺少 CP-DISR 高层技能 controller、Verifier、TaskEvaluator、安全和时限绑定，不能声明目标平台已选定并完成正式绑定。没有创建正式 scene 或 few-shot 图像，没有 API 请求。\n\nStage 0A=BLOCKED、Stage 0B=PASS、Stage 0C=BLOCKED 均保留。\n''')
write('input_readiness_report.md','''# Stage 0C-R1 输入就绪性\n\n正式输入：0/24；few-shot：0/3；本地 27 输入预检：未执行（输入尚未合法绑定）。现有模板和历史图像不能替代正式输入。`key_present`、模型权限和 API 行为仍不在本轮范围内。\n\n阻塞：目标平台高层技能接口、controller/verifier/evaluator、安全与时限、任务资产和逐项上传许可、24 个 dev scene、3 个独立 few-shot。\n''')
write('scene_diversity_report.md','''# Scene diversity\n\n未生成正式 scene，因此没有将随机 reset、历史截图或图像增强计为场景。D0/T_A/T_C 各 8 个 scene 的配置、image hash 和差异矩阵均为待绑定。\n''')
write('asset_license_report.md','''# Asset/license audit\n\nLIBERO 仓库包含 MIT LICENSE，代码仓库许可已记录；这不足以单独证明每个 bundled object/texture 可以上传到 VLM 服务。现阶段 0/24 正式 asset 获得 CP-DISR 所需逐项 `vlm_upload_allowed` 与证据引用。没有使用网络图片、生成图片、logo、纹理或历史 review 截图。\n''')
write('split_leakage_report.md','''# Split leakage\n\n未创建正式 dev 或 few-shot pool，故没有可宣称的零重叠结果。待绑定规则：24 个 scene 使用 `dev`，3 个 few-shot 使用独立 `independent_dev` pool；不得与 test split、scene config 或 image hash 重叠。\n''')
write('stage_1a_runtime_progress.md','''# Stage 1A runtime progress\n\n本轮仅确认 MuJoCo/robosuite/LIBERO reset/render 组件存在。仍未绑定：CP-DISR 高层 controller、postcondition/independent verifier、TaskEvaluator、安全授权、skill timeout/deadline、clock/reference seconds、perception/calibration、runtime factory 和可复现实验资产版本。Stage 1A 未启动。\n''')
rows=[]
for t in ['D0','T_A','T_C']:
 for i in range(8):rows.append({'slot_id':f'{t}_dev_{i:02d}','task_id':t,'case_index':i,'status':'BLOCKED_PLATFORM_SKILL_BINDING','scene_id':'','image_sha256':'','scene_config_sha256':'','missing':'high_level_controller; verifier; evaluator; permission; task_asset_binding'})
with (F/'formal_scene_manifest.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
rows2=[{'slot_id':f'fewshot_{i}','status':'BLOCKED_PLATFORM_SKILL_BINDING','image_sha256':'','scene_config_sha256':'','missing':'independent_dev_pool; expected_json; provenance; upload_permission'} for i in range(1,4)]
with (F/'unresolved_inputs.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows2[0]));w.writeheader();w.writerows(rows2)
dump('input_validation.json',{'status':'BLOCKED','formal_scene_count':0,'required_scene_count':24,'fewshot_count':0,'required_fewshot_count':3,'local_input_validation_passed':False,'reason':'No target platform satisfies CP-DISR high-level skill binding prerequisites; no formal images fabricated'})
write('input_validation.csv','input_type,observed,required,status\nformal_dev_scenes,0,24,BLOCKED\nfewshots,0,3,BLOCKED\nplatform_high_level_skill_controller,0,1,BLOCKED\nverifier_evaluator_safety_timing,0,1,BLOCKED\n')
dump('formal_scene_manifest.json',{'status':'BLOCKED','scenes':[],'required_count':24,'platform':'LIBERO_CANDIDATE_BLOCKED'})
dump('few_shot_validation.json',{'status':'BLOCKED','examples':[],'required_count':3,'platform':'LIBERO_CANDIDATE_BLOCKED'})
# retain Stage 0C blocked but add R1 evidence/readiness
status.update({'status':'BLOCKED','issues':sorted(set(status.get('issues',[])+['Stage 0C-R1: no platform currently satisfies CP-DISR high-level skill/controller/verifier/evaluator binding; 24+3 inputs not created'])),'readiness':{**status.get('readiness',{}),'provider_code_ready':True,'formal_scene_count':0,'required_scene_count':24,'frozen_fewshot_count':0,'required_fewshot_count':3,'local_input_validation_passed':False,'api_credential_present':False,'model_access_verified':False,'ready_for_formal_requests_after_credential':False,'r1_platform':'LIBERO_CANDIDATE_BLOCKED'}})
(ROOT/'experiments/stage_status/stage_0c.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
# Add only audit references to task manifest, leave template placeholders unchanged
tp=ROOT/'experiments/manifests/task_manifest.yaml';td=yaml.safe_load(tp.read_text());td['stage_0c_r1_platform_audit']='experiments/part_0_validation/stage_0c/platform_selection_report.md';td['stage_0c_r1_status']='BLOCKED_PLATFORM_SKILL_BINDING';tp.write_text(yaml.safe_dump(td,sort_keys=False,allow_unicode=True))
# new VLM input manifest records no bound input
dump('input_manifest_temp.json',{'status':'BLOCKED','formal_scene_manifest':'experiments/part_0_validation/stage_0c/formal_scene_manifest.json','few_shot_manifest':'experiments/part_0_validation/stage_0c/few_shot_validation.json'})
vp=ROOT/'experiments/manifests/vlm_input_manifest.yaml';vp.write_text(yaml.safe_dump({'status':'BLOCKED','platform':'LIBERO_CANDIDATE_BLOCKED','formal_scene_manifest':'experiments/part_0_validation/stage_0c/formal_scene_manifest.json','few_shot_manifest':'experiments/part_0_validation/stage_0c/few_shot_validation.json','formal_scene_count':0,'fewshot_count':0,'api_requests':0},sort_keys=False,allow_unicode=True))
# summarize all R1 outputs
write('stage_0c_resource_binding_summary.md',Path(F/'stage_0c_resource_binding_summary.md').read_text()+'\n报告文件已写入：platform_selection_report.md、input_readiness_report.md、scene_diversity_report.md、asset_license_report.md、split_leakage_report.md、formal_scene_manifest.csv、unresolved_inputs.csv。\n')
print('R1 BLOCKED reports written')
