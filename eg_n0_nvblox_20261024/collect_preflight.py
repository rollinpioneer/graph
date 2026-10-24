"""Record only this round's assets; export small evidence, never credentials or weights."""
import json,hashlib,subprocess,tarfile,base64,importlib.metadata
from pathlib import Path
R=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(['git','-C',str(R/'official'),*args],text=True).strip()
tag=git('rev-parse','HEAD');wheel='e4a2b00f45ecf989682d72a33a2c29defe39fd2d'
critical=['nvblox/include/nvblox/integrators/internal/cuda/impl/projective_tsdf_integrator_impl.cuh',
'nvblox/include/nvblox/sensors/internal/impl/camera_impl.h',
'nvblox/include/nvblox/interpolation/internal/impl/interpolation_2d_impl.h',
'nvblox_torch/cpp/src/sdf_query.cu','nvblox_torch/nvblox_torch/mapper.py']
diff=git('diff','--name-only',wheel,tag,'--','nvblox/include','nvblox/src','nvblox_torch')
manifest=dict(tag='v0.0.10',tag_commit=tag,wheel_commit=wheel,
 wheel_url='https://github.com/nvidia-isaac/nvblox/releases/download/v0.0.10/nvblox_torch-0.0.10+cu12ubuntu22-py3-none-linux_x86_64.whl',
 core_difference_files=diff.splitlines(),all_difference_files=git('diff','--name-only',wheel,tag).splitlines(),
 critical_sources={p:sha(R/'official'/p) for p in critical},
 installed_versions={n:importlib.metadata.version(n) for n in ['torch','nvblox-torch','open3d','numpy','scipy','pandas','pillow']},
 C3_formal_runs=0,C4_C5_accessed=False,training=False,parameter_updates=0,
 excluded_assets=['official checkout and LFS examples','environment and downloaded binary packages','full frozen EG input/prediction caches'])
(R/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
files=[]
for pattern in ['preflight/*.json','cache/*.json','*summary.json','runtime.json','plane_sanity.json','source_manifest.json']:
 files.extend(R.glob(pattern))
manifest2={p.relative_to(R).as_posix():sha(p) for p in files}
(R/'export_manifest.json').write_text(json.dumps(manifest2,indent=2));files.append(R/'export_manifest.json')
out=R/'preflight_light.tar.gz'
with tarfile.open(out,'w:gz') as t:
 for p in sorted(set(files)):t.add(p,arcname=p.relative_to(R).as_posix())
print(base64.b64encode(out.read_bytes()).decode())
