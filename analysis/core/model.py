"""EEGNet-style implementation with participant-level logit pooling."""
import torch
from torch import nn
from torch.nn import functional as F

class EEGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal=nn.Conv2d(1,8,(1,64),bias=False)
        self.bn1=nn.BatchNorm2d(8,eps=1e-3,momentum=.01)
        self.spatial=nn.Conv2d(8,16,(19,1),groups=8,bias=False)
        self.bn2=nn.BatchNorm2d(16,eps=1e-3,momentum=.01)
        self.depth=nn.Conv2d(16,16,(1,16),groups=16,bias=False)
        self.point=nn.Conv2d(16,16,1,bias=False)
        self.bn3=nn.BatchNorm2d(16,eps=1e-3,momentum=.01)
        self.elu1=nn.ELU(); self.elu2=nn.ELU()
        self.pool1=nn.AvgPool2d((1,4)); self.pool2=nn.AvgPool2d((1,8))
        self.drop1=nn.Dropout(.5); self.drop2=nn.Dropout(.5)
        self.head=nn.Linear(400,1)

    def window_logits(self,x):
        x=self.bn1(self.temporal(F.pad(x,(31,32,0,0))))
        x=self.drop1(self.pool1(self.elu1(self.bn2(self.spatial(x)))))
        x=self.point(self.depth(F.pad(x,(7,8,0,0))))
        x=self.drop2(self.pool2(self.elu2(self.bn3(x))))
        return self.head(x.flatten(1)).flatten()

    def forward(self,bags):
        n,w,c,t=bags.shape
        return self.window_logits(bags.reshape(n*w,1,c,t)).reshape(n,w).mean(1)

    def constrain(self):
        with torch.no_grad():
            for weight,limit in [(self.spatial.weight,1.),(self.head.weight,.25)]:
                dims=tuple(range(1,weight.ndim))
                norm=weight.square().sum(dims,keepdim=True).sqrt().clamp_min(1e-12)
                weight.mul_((limit/norm).clamp_max(1))
