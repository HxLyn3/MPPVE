import torch
import torch.nn as nn


class MLP(nn.Module):
    """ multi-layer perceptron """
    def __init__(self, input_dim, hidden_dims, output_dim=None, activation=nn.ReLU):
        super(MLP, self).__init__()

        # hidden layers
        dims = [input_dim] + list(hidden_dims)
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_dim, out_dim), activation()]
        
        self.output_dim = dims[-1]
        if output_dim is not None:
            layers += [nn.Linear(dims[-1], output_dim)]
            self.output_dim = output_dim
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class EnsembleLinear(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_ensemble,
        num_elites,
        weight_decay=0.0,
        load_model=False
    ):
        super().__init__()

        self.num_ensemble = num_ensemble
        self.num_elites = num_elites

        self.register_parameter("weight", nn.Parameter(torch.zeros(num_ensemble, input_dim, output_dim)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(num_ensemble, 1, output_dim)))

        nn.init.trunc_normal_(self.weight, std=1/(2*input_dim**0.5))

        self.register_parameter("saved_weight", nn.Parameter(self.weight.detach().clone()))
        self.register_parameter("saved_bias", nn.Parameter(self.bias.detach().clone()))

        if not load_model:
            self.register_parameter("elites", nn.Parameter(torch.tensor(list(range(0, self.num_ensemble))), requires_grad=False))
        else:
            self.register_parameter("elites", nn.Parameter(torch.tensor(list(range(0, self.num_elites))), requires_grad=False))

        self.weight_decay = weight_decay

    def forward(self, x):
        weight = self.weight[self.elites]
        bias = self.bias[self.elites]

        if len(x.shape) == 2:
            x = torch.einsum('ij,bjk->bik', x, weight)
        else:
            x = torch.einsum('bij,bjk->bik', x, weight)

        x = x + bias

        return x

    def set_elites(self, indexes):
        assert len(indexes) <= self.num_ensemble and max(indexes) < self.num_ensemble
        self.register_parameter('elites', nn.Parameter(torch.tensor(indexes), requires_grad=False))
        self.weight.data.copy_(self.saved_weight.data)
        self.bias.data.copy_(self.saved_bias.data)

    def update_save(self, indexes):
        self.saved_weight.data[indexes] = self.weight.data[indexes]
        self.saved_bias.data[indexes] = self.bias.data[indexes]
    
    def reset_elites(self):
        self.register_parameter('elites', nn.Parameter(torch.tensor(list(range(0, self.num_ensemble))), requires_grad=False))
    
    def get_decay_loss(self):
        decay_loss = self.weight_decay * (0.5*((self.weight**2).sum()))
        return decay_loss
