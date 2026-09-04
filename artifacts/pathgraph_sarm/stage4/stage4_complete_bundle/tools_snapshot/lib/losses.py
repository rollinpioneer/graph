import torch
def masked_huber(pred,target,mask=None):
    l=torch.nn.functional.smooth_l1_loss(pred,target,reduction='none'); return (l*mask).sum()/mask.sum().clamp_min(1) if mask is not None else l.mean()
def pairwise_rank(left,right,margin=.1): return torch.relu(margin-(left-right)).mean() if left.numel() else left.sum()*0
