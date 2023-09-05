import os
import torch
import numpy as np
from copy import deepcopy

from components import ACTOR, CRITIC
from components.dynamics import format_samples_for_training


class MPPVEAgent:
    """ Learning Model-based Predictive Policy with Sequence-value Estimation """
    def __init__(
        self,
        obs_shape, 
        hidden_dims, 
        action_dim,
        action_space,
        dynamics,
        plan_length,
        actor_lr,
        critic_lr,
        batch_size,
        tau=0.005, 
        gamma=0.99, 
        alpha=0.2,
        auto_alpha=True,
        alpha_lr=3e-4,
        target_entropy=-1,
        device="cpu"
    ):
        # actor
        self.actor = ACTOR["prob"](obs_shape, hidden_dims, action_dim, device)

        # critic
        self.valid_plan_length = 1
        self.plan_length = plan_length
        self.critic1 = CRITIC["q"](obs_shape, hidden_dims, action_dim*self.plan_length, device)
        self.critic2 = CRITIC["q"](obs_shape, hidden_dims, action_dim*self.plan_length, device)
        # target critic
        self.critic1_trgt = deepcopy(self.critic1)
        self.critic2_trgt = deepcopy(self.critic2)
        self.critic1_trgt.eval()
        self.critic2_trgt.eval()

        # optimizer
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optim = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optim = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        # env space
        self.obs_dim = np.prod(obs_shape)
        self.action_dim = action_dim
        self.action_space = action_space

        # alpha: weight of entropy
        self._auto_alpha = auto_alpha
        if self._auto_alpha:
            if not target_entropy:
                target_entropy = -np.prod(self.action_space.shape)*self.plan_length
            self._target_entropy = target_entropy*self.plan_length
            self._log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self._alpha = self._log_alpha.detach().exp()
            self._alpha_optim = torch.optim.Adam([self._log_alpha], lr=alpha_lr)
        else:
            self._alpha = alpha

        # dynamics model
        self.dynamics = dynamics

        # other parameters
        self._tau = tau
        self._gamma = gamma
        self._eps = np.finfo(np.float32).eps.item()
        self.batch_size = batch_size
        self.device = device

    def train(self):
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def eval(self):
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()

    def _sync_weight(self):
        """ synchronize weight """
        for trgt, src in zip(self.critic1_trgt.parameters(), self.critic1.parameters()):
            trgt.data.copy_(trgt.data*(1.0-self._tau) + src.data*self._tau)
        for trgt, src in zip(self.critic2_trgt.parameters(), self.critic2.parameters()):
            trgt.data.copy_(trgt.data*(1.0-self._tau) + src.data*self._tau)

    def actor4ward(self, obs, deterministic=False):
        """ forward propagation of actor """
        dist = self.actor(obs)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action)

        action_scale = torch.tensor((self.action_space.high-self.action_space.low)/2, device=self.device)
        squashed_action = torch.tanh(action)
        log_prob = log_prob - torch.log(action_scale*(1-squashed_action.pow(2))+self._eps).sum(-1, keepdim=True)

        return squashed_action, log_prob

    def actor4ward_plan(self, obs, deterministic=False):
        """ forward propagation of actor (planning version) """
        bs = obs.size(0)
        mask = torch.ones((bs, 1), device=self.device)
        plan_actions = torch.zeros((bs, self.plan_length, self.action_dim), device=self.device)
        plan_log_prob = torch.zeros((bs, self.plan_length, 1), device=self.device)
        obs = obs.cpu().detach().numpy()

        for t in range(self.plan_length):
            # plan one step
            obs_torch = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            action, log_prob = self.actor4ward(obs_torch, deterministic)
            plan_actions[:, t] = action*mask.clone().expand(-1, self.action_dim)
            plan_log_prob[:, t] = log_prob*mask.clone()
            if t > self.valid_plan_length:
                plan_actions[:, t] = plan_actions[:, t].detach()
                plan_log_prob[:, t] = plan_log_prob[:, t].detach()

            # imaginary step
            obs, _, done, _ = self.dynamics.step(obs, action.cpu().detach().numpy())
            mask[done.flatten()==1] = 0

        plan_actions = plan_actions.view((bs, -1))
        plan_log_prob = plan_log_prob.sum(1)
        return plan_actions, plan_log_prob

    def act(self, obs, deterministic=False, return_logprob=False):
        """ sample action """
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            action, log_prob = self.actor4ward(obs, deterministic)
            action = action.cpu().detach().numpy()
            log_prob = log_prob.cpu().detach().numpy()

        if return_logprob:
            return action, log_prob
        else:
            return action
    
    def value(self, obs, action):
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            if len(obs.shape) == 1:
                obs = obs.reshape(1, -1)
            na, log_prob = self.actor4ward_plan(obs)
            q1, q2 = self.critic1(obs, na), self.critic2(obs, na)
            value = torch.cat((q1, q2), dim=-1).mean(1).flatten().cpu().numpy()
        return value

    def plan(self, obs, deterministic=False):
        """ planning """
        plan_actions = []

        for _ in range(self.plan_length):
            with torch.no_grad():
                action = self.act(obs, deterministic)
                plan_actions.append(action)
                obs, _, _, _ = self.dynamics.step(obs, action)
        return plan_actions
    
    def rollout_transitions(self, init_transitions, rollout_len):
        if not hasattr(self, "gammas"):   
            self.gammas = self._gamma**np.arange(self.plan_length).reshape((self.plan_length, 1))

        """ rollout to generate {plan_length}-steps transitions """
        obs = init_transitions["s_"][:, -self.obs_dim:]
        transitions = init_transitions

        nstep_transitions = {"s": [], "a": [], "r": [], "s_": [], "done": []}
        for _ in range(rollout_len):
            # imaginary step
            actions = self.act(obs)
            next_obs, rewards, dones, _ = self.dynamics.step(obs, actions)

            # update
            transitions["s"] = np.concatenate((transitions["s"], obs), axis=-1)
            transitions["a"] = np.concatenate((transitions["a"], actions), axis=-1)
            transitions["r"] = np.concatenate((transitions["r"], rewards), axis=-1)
            transitions["s_"] = np.concatenate((transitions["s_"], next_obs), axis=-1)
            transitions["done"] = np.concatenate((transitions["done"], dones), axis=-1)

            # store
            nstep_transitions["s"].append(transitions["s"][:, :self.obs_dim])
            nstep_transitions["a"].append(transitions["a"])
            nstep_transitions["r"].append(transitions["r"].dot(self.gammas))
            nstep_transitions["s_"].append(transitions["s_"][:, -self.obs_dim:])
            nstep_transitions["done"].append(transitions["done"].sum(-1, keepdims=True).clip(None, 1))

            # to next step
            nonterm_mask = (~dones).flatten()
            if nonterm_mask.sum() == 0: break
            obs = next_obs[nonterm_mask]

            # mask
            transitions["s"] = transitions["s"][nonterm_mask, self.obs_dim:]
            transitions["a"] = transitions["a"][nonterm_mask, self.action_dim:]
            transitions["r"] = transitions["r"][nonterm_mask, 1:]
            transitions["s_"] = transitions["s_"][nonterm_mask, self.obs_dim:]
            transitions["done"] = transitions["done"][nonterm_mask, 1:]

        nstep_transitions = {key: np.concatenate(nstep_transitions[key], axis=0) for key in nstep_transitions.keys()}
        return nstep_transitions

    def learn_dynamics(self, transitions):
        """ learn dynamics model """
        inputs, targets = format_samples_for_training(transitions)
        loss = self.dynamics.train(
            inputs,
            targets,
            batch_size=self.batch_size
        )
        return loss["holdout_loss"].item()

    def learn_actor(self, s):
        """ learn predictive policy from {plan_length}-steps transitions """
        s = torch.as_tensor(s, device=self.device)
        # update actor
        na, log_prob = self.actor4ward_plan(s)
        q1, q2 = self.critic1(s, na).flatten(), self.critic2(s, na).flatten()
        actor_loss = (self._alpha*log_prob.flatten() - torch.min(q1, q2)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
        self.actor_optim.step()
        actor_loss = actor_loss.item()

        # update alpha
        if self._auto_alpha:
            log_prob = log_prob.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha*log_prob).mean()
            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()
            alpha_loss = alpha_loss.item()
            self._alpha = self._log_alpha.detach().exp()


        info = {
            "actor_loss": actor_loss
        }

        if self._auto_alpha:
            info["alpha_loss"] = alpha_loss
            info["alpha"] = self._alpha.item()
        else:
            info["alpha"] = self._alpha.item()

        return info

    def learn_critic(self, s, a, r, s_, done):
        """ learn predictive policy from {plan_length}-steps transitions """
        s      = torch.as_tensor(s, device=self.device)
        na     = torch.as_tensor(a, device=self.device)
        nr     = torch.as_tensor(r, device=self.device)
        nth_s_ = torch.as_tensor(s_, device=self.device)
        done   = torch.as_tensor(done, device=self.device)

        # update critic
        q1, q2 = self.critic1(s, na), self.critic2(s, na)
        with torch.no_grad():
            na_, log_prob_ = self.actor4ward_plan(nth_s_)
            nth_q_ = torch.min(
                self.critic1_trgt(nth_s_, na_), 
                self.critic2_trgt(nth_s_, na_)) - self._alpha*log_prob_
            q_trgt = nr + self._gamma**self.plan_length*(1-done)*nth_q_

        critic1_loss = ((q1-q_trgt).pow(2)).mean()
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        critic2_loss = ((q2-q_trgt).pow(2)).mean()
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        # synchronize weight
        self._sync_weight()

        info = {
            "critic_loss": (critic1_loss.item() + critic2_loss.item()) / 2,
            "value": torch.cat((q1, q2), dim=-1).mean(1).mean().item()
        }
        return info
    
    def save_model(self, filepath):
        """ save model """
        # save policy
        state_dict = {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "alpha": self._alpha
        }
        torch.save(state_dict, filepath)
        
        # save dynamics
        dynamics_dir = filepath.split(".pth")[0]
        if not os.path.exists(dynamics_dir):
            os.makedirs(dynamics_dir)
        self.dynamics.save(dynamics_dir)

    def load_model(self, filepath):
        """ load model """
        # load policy
        state_dict = torch.load(filepath, map_location=torch.device(self.device))
        self.actor.load_state_dict(state_dict["actor"])
        self.critic1.load_state_dict(state_dict["critic1"])
        self.critic2.load_state_dict(state_dict["critic2"])
        self._alpha = state_dict["alpha"]
