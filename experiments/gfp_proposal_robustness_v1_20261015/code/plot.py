import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
R=Path(__file__).resolve().parents[1];d=json.loads((R/'reports/summary.json').read_text());rows=d['states'];labels=['original_actor','resampled_actor','vabc'];a=np.array([[x['new_successes'][k] for k in labels] for x in rows])
fig,ax=plt.subplots(figsize=(9,6.5));im=ax.imshow(a,vmin=0,vmax=7,cmap='Blues',aspect='auto');ax.set_xticks(range(3),['Original Actor','Resampled Actor','VaBC']);ax.set_yticks(range(12),[x['state'] for x in rows],fontsize=8)
for i in range(12):
 for j in range(3):ax.text(j,i,f'{a[i,j]}/7',ha='center',va='center',color='white' if a[i,j]>3 else 'black',fontsize=10)
ax.set_title('Frozen GFP: 300-step success under seven NEW common suffixes',fontsize=11);fig.colorbar(im,ax=ax,label='Success count (out of 7)');fig.text(.5,.01,'Selected states; fixed first actions; local diagnostic, not overall policy success.',ha='center',fontsize=8);fig.tight_layout(rect=[0,.03,1,1]);fig.savefig(R/'reports/new_suffix_successes.png',dpi=160);plt.close(fig)
