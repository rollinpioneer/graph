"""Reproduce the frozen EG-B score layer; does not run the backbone or train.

Do not collapse c^T(Wt+b) to one affine map: historical cuDNN TF32
rounds operands before the six-channel convolution and then reduces in FP32.
This fixes cache replay, not the separate pure-affine P2 initialization gate.
"""
import numpy as np
import torch
import torch.nn.functional as F

class FrozenScoreReplay:
    def __init__(self, state, device='cuda:0'):
        self.device=torch.device(device)
        if self.device.type!='cuda':
            raise ValueError('Historical score replay requires CUDA/cuDNN; no silent CPU fallback')
        self.weight=state['grasp_head.conv_score.weight'].detach().to(self.device,torch.float32)
        self.bias=state['grasp_head.conv_score.bias'].detach().to(self.device,torch.float32)
        self.c=torch.tensor([0,.2,.4,.6,.8,1],device=self.device,dtype=torch.float32)[:,None]
    @torch.no_grad()
    def __call__(self, feature320):
        assert feature320.shape==(1024,320) and feature320.dtype==np.float32
        x=torch.from_numpy(feature320[:,256:].T.copy())[None].to(self.device)
        with torch.backends.cudnn.flags(enabled=True,benchmark=False,deterministic=False,allow_tf32=True):
            y=F.conv1d(x,self.weight,self.bias)
            score=(y[0]*self.c).sum(0)
        return score.cpu().numpy()

def reconstruction_metrics(score, reference):
    score=np.asarray(score,dtype=np.float64);reference=np.asarray(reference,dtype=np.float64)
    order=np.argsort(-score,kind='stable');delta=np.diff(reference[order])
    return dict(max_error=float(np.max(abs(score-reference))),exact_candidates=int((score==reference).sum()),
                nonnear_inversions=int((delta>=1e-5).sum()),near_inversions=int(((delta>0)&(delta<1e-5)).sum()),
                order_identical=bool(np.array_equal(order,np.argsort(-reference,kind='stable'))))
