import json,time,sys
from pathlib import Path
import torch
import MinkowskiEngine as ME
from libs.knn.knn_modules import knn
from libs.pointnet2.pointnet2_utils import furthest_point_sample,gather_operation,grouping_operation,cylinder_query
r=Path(__file__).parent; started=time.perf_counter();torch.manual_seed(20261015)
report={'synthetic_only':True,'real_data_smoke_passed':False,'checks':{}}
ref=torch.randn(2,3,17,device='cuda');query=torch.randn(2,3,11,device='cuda')
actual=knn(ref.contiguous(),query.contiguous(),k=1).squeeze(1)-1
expected=torch.cdist(query.transpose(1,2),ref.transpose(1,2)).argmin(-1)
assert torch.equal(actual,expected);report['checks']['knn_vs_cdist']='PASS'
xyz=torch.rand(2,64,3,device='cuda')+.1
idx=furthest_point_sample(xyz,16);minimum=torch.full((2,64),float('inf'),device='cuda');old=torch.zeros(2,dtype=torch.long,device='cuda');out=[]
for i in range(16):
 out.append(old);delta=(xyz-xyz[torch.arange(2,device='cuda'),old][:,None,:]).square().sum(-1);minimum=torch.minimum(minimum,delta);old=minimum.argmax(-1)
assert torch.equal(idx.long(),torch.stack(out,1));report['checks']['fps_vs_reference']='PASS'
f=torch.randn(2,3,64,device='cuda',requires_grad=True);g=gather_operation(f,idx);target=torch.gather(f,2,idx.long()[:,None,:].expand(-1,3,-1));assert torch.equal(g,target)
g.sum().backward();expect_grad=torch.zeros_like(f).scatter_add_(2,idx.long()[:,None,:].expand(-1,3,-1),torch.ones_like(g));assert torch.equal(f.grad,expect_grad);report['checks']['gather_forward_backward']='PASS'
cloud=torch.tensor([[[.001,.002,.001],[-.01,.002,.001],[.08,.001,.001],[0,.07,0]]],device='cuda');center=torch.zeros(1,1,3,device='cuda');rot=torch.eye(3,device='cuda').view(1,1,9)
ci=cylinder_query(.05,-.02,.04,4,cloud,center,rot);assert set(ci.cpu().reshape(-1).tolist()) <= {0,1};assert sorted(ci.cpu().reshape(-1)[:2].tolist())==[0,1]
ff=torch.randn(1,3,4,device='cuda',requires_grad=True);group=grouping_operation(ff,ci);group.sum().backward();expected_grad=torch.zeros_like(ff).scatter_add_(2,ci.long().view(1,1,-1).expand(-1,3,-1),torch.ones(1,3,ci.numel(),device='cuda'));assert torch.equal(ff.grad,expected_grad);report['checks']['cylinder_and_grouping_backward']='PASS'
coords=ME.utils.batched_coordinates([torch.cartesian_prod(torch.arange(4),torch.arange(4),torch.arange(4))]).int().cuda()
features=torch.randn(len(coords),3,device='cuda',requires_grad=True);sp=ME.SparseTensor(features,coordinates=coords);conv=ME.MinkowskiConvolution(3,8,kernel_size=3,dimension=3).cuda();out=conv(sp);loss=out.F.square().mean();loss.backward();assert torch.isfinite(out.F).all() and torch.isfinite(features.grad).all() and torch.isfinite(conv.kernel.grad).all();report['checks']['minkowski_sparse_convolution_forward_backward']='PASS'
torch.cuda.synchronize();report.update(status='PASS',wall_seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),device=torch.cuda.get_device_name())
(r/'receipts/operator_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
