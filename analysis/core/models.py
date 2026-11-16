"""Neural classifiers with participant-level logit averaging."""
import torch
from torch import nn
from model import EEGNet

class TemporalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for a,b,k in [(19,16,31),(16,32,15),(32,32,7)]:
            layers.extend([nn.Conv1d(a,b,k,padding=k//2),
                nn.BatchNorm1d(b,eps=.001,momentum=.01), nn.ELU(),
                nn.AvgPool1d(4), nn.Dropout(.5)])
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(32,1)

    def window_logits(self,x):
        return self.head(self.features(x[:,0]).mean(-1)).flatten()

    def forward(self,x):
        n,w,c,t = x.shape
        return self.window_logits(x.reshape(n*w,1,c,t)).reshape(n,w).mean(1)

    def constrain(self):
        pass

def make_model(name):
    return {'eegnet': EEGNet, 'temporal_cnn': TemporalCNN}[name]()
