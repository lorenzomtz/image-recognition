import torch
import torch.nn.functional as F

class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):

        # use cross entropy loss
        return F.cross_entropy(input, target)


class LinearClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # single layer network
        self.linear = torch.nn.Linear(3*64*64, 6)

    def forward(self, x):

        # run data through the layer
        return self.linear(x.view(x.shape[0], -1))

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


model_factory = {
    'linear': LinearClassifier,
    'mlp': MLPClassifier,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r