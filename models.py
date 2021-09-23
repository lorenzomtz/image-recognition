import torch
import torch.nn.functional as F

class MLPClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        # two layer network with activation (ReLU)
        self.linear = torch.nn.Sequential(torch.nn.Linear(3*64*64, 100),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(100, 6))
    
    def forward(self, x):
        # run data through the network
        x_flat = x.view(x.shape[0], -1)
        return self.linear(x_flat)