"""Read-only original assets and from-scratch initialization identity checks."""
from common import *
EV=Path('/home/xushijie3/work/graph_saw_eval02_20261012/experiments/saw_eval02_20261012')
def compare(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b);return max((compare(a[k],b[k]) for k in a),default=0.)
    if isinstance(a,(tuple,list)):
        assert len(a)==len(b);return max((compare(x,y) for x,y in zip(a,b)),default=0.)
    aa,bb=np.asarray(a),np.asarray(b);assert aa.shape==bb.shape
    if np.issubdtype(aa.dtype,np.number):return float(np.max(np.abs(aa.astype(np.float64)-bb.astype(np.float64)))) if aa.size else 0.
    assert np.array_equal(aa,bb);return 0.
def main():
    checks=[];initial=[];weights=[]
    for root in (OLD,EV):
        manifest=json.loads((root/'LIGHT_MANIFEST.json').read_text())
        for item in manifest['files']:
            path=root/item['relative'];assert sha(path)==item['sha256'],str(path)
        checks.append(dict(root=str(root),manifest_sha256=sha(root/'LIGHT_MANIFEST.json'),verified_files=len(manifest['files']),unchanged=True))
    for kind in ('navigate','stitch'):
        for item in json.loads((OLD/'results'/f'{kind}_checkpoints.json').read_text()):
            assert sha(item['path'])==item['sha256'];weights.append(dict(path=item['path'],sha256=item['sha256']))
        a=OLD/'runs'/f'formal_{kind}'/'params_0.pkl';b=ROOT/'runs'/f'formal_{kind}'/'params_0.pkl'
        with open(a,'rb') as f:sa=pickle.load(f)
        with open(b,'rb') as f:sb=pickle.load(f)
        assert sa['step']==sb['step']==0
        err=compare(sa['agent'],sb['agent']);assert err==0
        initial.append(dict(kind=kind,A_path=str(a),B_path=str(b),agent_params_optimizer_rng_max_abs=err,fresh_step=0,
            note='agent serialization identical at update0; environment pre-init random scene excluded; full SGD history not inferred'))
    write_json(ROOT/'results/READ_ONLY_ASSETS.json',dict(original_manifests=checks,A_checkpoint_identities=weights,initialization=initial))
    print('READ_ONLY_AND_INITIALIZATION_PASS',flush=True)
if __name__=='__main__':main()
