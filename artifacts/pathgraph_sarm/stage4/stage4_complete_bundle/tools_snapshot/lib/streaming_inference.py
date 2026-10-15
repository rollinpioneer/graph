import torch
class StreamingInference:
    def __init__(self,model,history_steps=32,device='cpu'): self.model=model; self.history_steps=history_steps; self.device=device; self.buf=[]
    def reset(self): self.buf=[]
    def step(self,x):
        self.buf.append(torch.as_tensor(x,dtype=torch.float32)); self.buf=self.buf[-self.history_steps:]; return self.model(torch.stack(self.buf).unsqueeze(0).to(self.device))
