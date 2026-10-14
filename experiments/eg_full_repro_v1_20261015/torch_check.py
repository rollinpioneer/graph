import torch,torchvision,numpy,json
x=torch.randn(128,128,device='cuda',requires_grad=True)
(x@x.T).square().mean().backward()
torch.cuda.synchronize()
assert torch.isfinite(x.grad).all()
print(json.dumps({'torch':torch.__version__,'torchvision':torchvision.__version__,'cuda':torch.version.cuda,'numpy':numpy.__version__,'finite_gradient':True,'device':torch.cuda.get_device_name(),'peak_allocated':torch.cuda.max_memory_allocated()}))
