"""Supplementary CPU residency snapshots; final tick before exit is a lower bound."""
import json
import os
from pathlib import Path
import time

root=Path(__file__).resolve().parent
jobs=json.loads((root/'receipts/JOB_REGISTRY.json').read_text())
ticks=os.sysconf('SC_CLK_TCK')
final={}; timeline=[]
while True:
    active=[]
    for job in jobs:
        label=job['label'];pid=job['pid']
        try:
            stat=Path(f'/proc/{pid}/stat').read_text().split()
            if stat[2]=='Z': continue
            cpu=(int(stat[13])+int(stat[14]))/ticks
            item=dict(label=label,pid=pid,cpu_seconds_lower_bound=cpu,snapshot_unix=time.time())
            final[label]=item; active.append(label)
        except FileNotFoundError: pass
    timeline.append(dict(unix=time.time(),active=active,cpu_seconds={k:v['cpu_seconds_lower_bound'] for k,v in final.items()}))
    (root/'receipts/formal_CPU_ACCOUNTING.json').write_text(json.dumps(dict(final_snapshots=final,exit_resolution_seconds=1,
                                                                         interpretation='last observed process user+system CPU ticks; lower bound, at most about one sampling interval missing',
                                                                         timeline=timeline[::30]),indent=2)+'\n')
    if not active: break
    time.sleep(1)
