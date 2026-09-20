"""Explicit binding gate and plugin loader. Never infer controllers from filenames."""
import importlib,hashlib
from pathlib import Path
from .common import BindingError,unresolved

REQUIRED=('simulator_or_robot','environment_version','task_assets','controller_manifest','camera','calibration_manifest','perception_checkpoint','verifier_thresholds','skill_timeouts','task_deadlines','actual_interaction_time_unit','reference_skill_seconds_by_task','task_evaluator_version','safety_authorization','task_splits')
COMPONENTS=('environment','executor','observations','perception','verifier','evaluator','safety','clock','snapshot_builder')

def binding_issues(manifest):
    runtime=manifest.get('runtime',{});issues=[]
    for key in REQUIRED:
        if key not in runtime or runtime[key] in ('',{},[]) or unresolved(runtime[key],key):issues.append('runtime.'+key)
    implementation=manifest.get('runtime_factory',{})
    for key in ('module','factory','source_path','sha256'):
        if not implementation.get(key) or unresolved(implementation[key],key):issues.append('runtime_factory.'+key)
    if runtime.get('actual_interaction_time_unit') not in ('seconds',):issues.append('runtime.actual_interaction_time_unit=seconds')
    return sorted(set(issues))

def require_runtime(manifest):
    issues=binding_issues(manifest)
    if issues:raise BindingError('MUST_BIND: '+', '.join(issues))
    spec=manifest['runtime_factory'];path=Path(spec['source_path'])
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=spec['sha256']:raise BindingError('MUST_BIND: runtime_factory verified source hash')
    return spec

def load_runtime(manifest):
    spec=require_runtime(manifest)
    module=importlib.import_module(spec['module'])
    if Path(module.__file__).resolve()!=Path(spec['source_path']).resolve():raise BindingError('Runtime module/source mismatch')
    bundle=getattr(module,spec['factory'])(manifest)
    for name in COMPONENTS:
        if not hasattr(bundle,name):raise BindingError('MUST_BIND: runtime component '+name)
    return bundle

def require_cache_configuration(manifest):
    issues=[]
    for k in ('region','base_http_api_url','api_account_authorized','fewshot_manifest','credential_env_name','provider_plugin'):
        if k not in manifest or unresolved(manifest[k],k):issues.append('vlm.'+k)
    if manifest.get('api_account_authorized') is not True:issues.append('vlm.api_account_authorized=true')
    # Never inspect os.environ here. Credential values belong exclusively to a bound provider.
    if issues:raise BindingError('MUST_BIND: '+', '.join(sorted(set(issues))))
    raise BindingError('MUST_BIND: approved provider implementation; this revision implements local parsing only')
