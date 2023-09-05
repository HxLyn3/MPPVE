import torch
import numpy as np
import torch.nn as nn
from components.network import MLP


class Critic(nn.Module):
    """ Q(s,a) """
    def __init__(self, obs_shape, hidden_dims, action_dim, device="cpu"):
        super(Critic, self).__init__()
        self.device = torch.device(device)
        self.backbone = MLP(input_dim=np.prod(obs_shape)+action_dim, hidden_dims=hidden_dims).to(self.device)
        latent_dim = getattr(self.backbone, "output_dim")
        self.last = nn.Linear(latent_dim, 1).to(self.device)

    def forward(self, obs, actions):
        """ return Q(s,a) """
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        net_in = torch.cat([obs, actions], dim=1)
        logits = self.backbone(net_in)
        values = self.last(logits)
        return values
