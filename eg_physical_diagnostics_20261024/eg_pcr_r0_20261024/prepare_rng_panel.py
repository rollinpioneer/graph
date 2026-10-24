from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent
src=ROOT/'rngnet/test_graspnet.py';s=src.read_text()
prefix='''import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'rngnet'))
from dataset.base_grasp_dataset import GraspDataset
def input_only(self,idx):
    import numpy as np
    depth=self.get_depth(idx,0.,1.)
    rgb=self.get_rgb(idx,0.,1.)
    return self.to_tensor(np.vstack([depth[None],rgb])), 0, idx, 0., 1.
GraspDataset.__getitem__=input_only
'''
changes={
 'sceneIds = list(range(args.scene_l, args.scene_r))':'sceneIds = [100,106,115,124,132,139,150,157,160,169,176,186]',
 '    SCENE_LIST = test_dataset.scene_list()':'''    selected=[i*256+f for i in range(len(sceneIds)) for f in [0,64,128,192]]
    for key in ['colorpath','depthpath','metapath','grasppath','scenename','frameid']:
        values=getattr(test_dataset,key);setattr(test_dataset,key,[values[i] for i in selected])
    if os.environ.get('R0_PILOT')=='1':
        for key in ['colorpath','depthpath','metapath','grasppath','scenename','frameid']:
            setattr(test_dataset,key,getattr(test_dataset,key)[:1])
    SCENE_LIST = test_dataset.scene_list()''',
 'num_workers=4)':'num_workers=0)',
 "    checkpoint = torch.load(args.checkpoint)":'''    checkpoint = torch.load(args.checkpoint)
    removed=[]
    for name in ['anchor','local']:
        for key in list(checkpoint[name]):
            if key.split('.')[-1] in ['total_ops','total_params']:
                removed.append(name+'.'+key);del checkpoint[name][key]
    print('Removed only profiling counter buffers:',removed)
''',
 "str(batch_idx % 256).zfill(4) + '.npy'":"str(test_dataset.frameid[batch_idx]).zfill(4) + '.npy'",
 '    evaluate(save_folder, multi_scale=args.multi_scale)':'    # Evaluation is separate and restricted to the fixed 48-frame panel.'}
for a,b in changes.items():
 assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
# Remove MAC profiling: it is unrelated to inference and runs before weights load.
start=s.index("    x = torch.randn((48, 64, 64, 6), device='cuda')")
end=s.index('    localnet_params =',start);s=s[:start]+s[end:]
(ROOT/'rngnet_test_panel.py').write_text(prefix+s)
(ROOT/'rng_adapter_receipt.json').write_text(json.dumps(dict(official_test_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),changes=list(changes),labels_used=False,network_forward_and_decode='unchanged',profiling_removed=True),indent=2))
