"""Read-only bounded availability check using the pinned official URL and MD5."""
import urllib.request,hashlib,json,time,pathlib
r=pathlib.Path(__file__).resolve().parents[1];url='http://datasets.polixir.site/neorl2/Pipeline/Pipeline-val.npz';out=dict(url=url,expected_md5='c502f529db1de55ebf37d63e1defca56',status='UNVERIFIED',max_bytes=20000000,epoch=time.time())
try:
 with urllib.request.urlopen(url,timeout=20) as f:
  b=f.read(20000001);out.update(final_url=f.url,bytes=len(b));assert len(b)<=20000000
 out['actual_md5']=hashlib.md5(b).hexdigest();assert out['actual_md5']==out['expected_md5']
 p=r/'sources/Pipeline-val.npz';assert not p.exists();p.write_bytes(b);out.update(status='FETCHED_VERIFIED',sha256=hashlib.sha256(b).hexdigest(),path=str(p))
except Exception as e:out.update(status='ACCESS_FAILED',error=repr(e))
(r/'receipts/public_data_access.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
