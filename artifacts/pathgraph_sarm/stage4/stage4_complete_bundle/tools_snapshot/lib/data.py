import csv,numpy as np
from pathlib import Path
class SupervisionDataset:
    def __init__(self,root,split=None):
        self.root=Path(root); self.rows=list(csv.DictReader(open(self.root/'tables/episode_manifest.csv'))); self.rows=[r for r in self.rows if split is None or r['split_original']==split]
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]; return r,np.load(self.root/r['file'])
