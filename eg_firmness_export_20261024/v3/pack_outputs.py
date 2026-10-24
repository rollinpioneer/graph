"""Transport only explicitly selected lightweight exports, excluding caches."""
import sys,json,io,tarfile,base64
from pathlib import Path
ROOT=Path(__file__).resolve().parent
group=sys.argv[1]
assert (ROOT/'summary.json').is_file() and not (ROOT/'STOP_ON_FAILURE').exists()
paths=[]
if group=='meta':
 paths=[p for p in ROOT.iterdir() if p.is_file() and p.suffix in {'.py','.json','.txt'}]
else:
 split,shard=group.split(':');shard=int(shard)
 assert split in ['cal','C3','P2','probe']
 files=sorted((ROOT/'eval').glob(split+'_*.npz'))
 if split=='probe':files=files[shard::8]
 else:assert shard==0
 for p in files:
  key=p.stem;paths.extend([p,ROOT/f'features/{key}.npz',ROOT/f'receipts/eval_{key}.json',ROOT/f'receipts/infer_{key}.json'])
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz') as tar:
 for p in sorted(paths):tar.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
sys.stdout.write(base64.b64encode(buf.getvalue()).decode())
