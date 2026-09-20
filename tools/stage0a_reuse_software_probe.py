"""Stage 0A dependency/serialization probe only; no task or policy rollout."""
import csv, hashlib, importlib, importlib.metadata, json, platform, subprocess, sys, traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/home/__compress_data/xushijie/graph_cp_disr_v2_1')
OUT = ROOT / (ROOT / '.stage0a_environment_revision').read_text().strip()
PACKAGES = {
    'torch': ('2.7.1', 'torch'), 'torch-geometric': ('2.6.1', 'torch_geometric'),
    'numpy': ('1.26.4', 'numpy'), 'pandas': ('2.2.3', 'pandas'),
    'matplotlib': ('3.9.4', 'matplotlib'), 'jsonschema': ('4.23.0', 'jsonschema'),
    'PyYAML': ('6.0.2', 'yaml'), 'tensorboard': ('2.19.0', 'tensorboard'),
    'pytest': ('8.3.5', 'pytest'), 'gymnasium': ('1.1.1', 'gymnasium'),
    'dashscope': ('1.27.6', 'dashscope'),
}
report = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'python': platform.python_version(),
          'python_executable': sys.executable, 'packages': {}, 'checks': {},
          'scope': 'STAGE_0A_INFRASTRUCTURE_ONLY', 'rl_jobs': 0, 'ppo_updates': 0,
          'real_skill_transitions': 0, 'performance_episodes': 0, 'vlm_requests': 0,
          'robot_actions': 0, 'stage_0b_executed': False}
for package, (version, module) in PACKAGES.items():
    item = {'requested': version, 'resolved': None, 'import_status': 'NOT_TESTED'}
    try:
        item['resolved'] = importlib.metadata.version(package)
        importlib.import_module(module)
        item['import_status'] = 'PASS'
    except Exception:
        item['import_status'] = 'BLOCKED'
        item['error'] = traceback.format_exc()
    report['packages'][package] = item
try:
    import torch
    from torch_geometric.nn import RGCNConv
    torch.set_num_threads(2)
    torch.manual_seed(0)
    # A synthetic library smoke probe, never a task fixture or a CP-DISR result.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    report['cuda'] = {'available': torch.cuda.is_available(), 'runtime': torch.version.cuda,
                      'device': str(device), 'torch_build':torch.__version__, 'gpu_arch_list': torch.cuda.get_arch_list()}
    layers = torch.nn.ModuleList([RGCNConv(128, 128, 12, num_bases=4, aggr='mean', root_weight=True) for _ in range(4)]).to(device)
    norms = torch.nn.ModuleList([torch.nn.LayerNorm(128) for _ in range(4)]).to(device)
    x = torch.randn(5, 128, device=device, requires_grad=True)
    edges = torch.tensor([[0,1,1,2,2,3,3,4], [1,0,2,1,3,2,4,3]],device=device)
    rel = torch.tensor([0,1,2,3,4,5,10,11],device=device)
    y = x
    for layer, norm in zip(layers,norms):
        y = norm(torch.relu(layer(y,edges,rel)))
    y[:,0].sum().backward()
    params = list(layers.parameters()) + list(norms.parameters())
    assert torch.isfinite(y).all() and x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    report['checks']['rgcn_forward_backward'] = {'status':'PASS', 'device':str(device), 'dtype':'float32', 'layers':4, 'hidden_dim':128, 'num_bases':4, 'optimizer_steps':0}
    path = OUT / 'infrastructure_roundtrip.pt'
    state = {'layers':layers.state_dict(),'norms':norms.state_dict(),'label':'STAGE_0A_UNTRAINED_LIBRARY_PROBE_NOT_POLICY'}
    torch.save(state,path)
    loaded = torch.load(path, map_location=device, weights_only=True)
    assert all(torch.equal(value,loaded[group][name]) for group in ('layers','norms') for name,value in state[group].items())
    report['checks']['checkpoint_roundtrip'] = {'status':'PASS', 'path':str(path.relative_to(ROOT)), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'trained_policy':False}
except Exception:
    report['checks']['library_probe_error'] = {'status':'BLOCKED','error':traceback.format_exc()}
sys.path.insert(0, str(ROOT / 'src'))
report['core_imports'] = {}
for name in ['cp_disr.neural','cp_disr.torch_rl','cp_disr.execution','cp_disr.collector','cp_disr.evaluation']:
    try:
        importlib.import_module(name)
        report['core_imports'][name] = 'PASS'
    except Exception:
        report['core_imports'][name] = traceback.format_exc()
report['compatibility_adjustment'] = {'python': {'preferred': '3.11.13', 'selected': '3.10.19'}, 'reason': 'Reuse existing PyTorch 2.7.1+cu126 binaries in an isolated venv; no frozen Method change'}
report['profile_validated'] = (platform.python_version() == '3.10.19'
    and all(v['import_status']=='PASS' and v['resolved']==v['requested'] for v in report['packages'].values())
    and report.get('cuda',{}).get('torch_build')=='2.7.1+cu126' and report['checks'].get('checkpoint_roundtrip',{}).get('status')=='PASS' and all(v=='PASS' for v in report['core_imports'].values()))
(OUT/'software_checks.json').write_text(json.dumps(report,indent=2)+'\n')
with (OUT/'dependency_table.csv').open('w',newline='') as f:
    writer=csv.writer(f); writer.writerow(['package','requested','resolved','import_status'])
    writer.writerow(['Python','3.10.19',platform.python_version(),'CURRENT_PROCESS'])
    for name,item in report['packages'].items(): writer.writerow([name,item['requested'],item['resolved'],item['import_status']])
    writer.writerow(['uv','0.8.22',subprocess.check_output([str(ROOT/'.bootstrap/bin/uv'),'--version'],text=True).strip(),'CLI_PASS'])
print(json.dumps(report,indent=2))
