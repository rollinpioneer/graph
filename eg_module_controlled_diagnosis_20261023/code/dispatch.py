import sys,json,base64
from pathlib import Path
q=Path(__file__).parents[1]/'ssh_queue'
n,name,host,mode,*args=sys.argv[1:]
if mode=='upload':
 p=Path(args[0]);dest=args[1];b=base64.b64encode(p.read_bytes()).decode();cmd=f"echo '{b}' | base64 -d > {dest}"
else:cmd=Path(args[0]).read_text() if mode=='file' else args[0]
(q/f'{n}-{name}.request.json').write_text(json.dumps([dict(host=host,command=cmd)]),encoding='utf-8')
