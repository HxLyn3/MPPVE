import os

import numpy as np
import torch

from utils.scaler import StandardScaler


class Dynamics:
    def __init__(self, model, static_fn):
        self.model = model
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.scaler = StandardScaler()
        self.static_fn = static_fn

    @ torch.no_grad()
    def step(self, obs, action):
        obs_act = np.concatenate([obs, action], axis=-1)
        obs_act = self.scaler.transform(obs_act)
        mean, logvar = self.model(obs_act)
        mean = mean.cpu().numpy()
        logvar = logvar.cpu().numpy()
        mean[..., :-1] += obs
        std = np.sqrt(np.exp(logvar))
        samples = (mean + np.random.normal(size=mean.shape) * std).astype(np.float32)
        next_obses = samples[..., :-1]
        rewards = samples[..., -1:]

        select_indexes = np.random.randint(0, next_obses.shape[0], size=(obs.shape[0]))
        next_obs = next_obses[select_indexes, np.arange(obs.shape[0])]
        reward = rewards[select_indexes, np.arange(obs.shape[0])]
        terminal = self.static_fn.termination_fn(obs, action, next_obs)
        
        return next_obs, reward, terminal, {}
    
    def train(self, inputs, targets, batch_size=256):
        self.model.reset_elites()
        data_size = inputs.shape[0]
        holdout_size = min(int(data_size * 0.2), 5000)
        train_size = data_size - holdout_size
        train_splits, holdout_splits = torch.utils.data.random_split(range(data_size), (train_size, holdout_size))
        train_inputs, train_targets = inputs[train_splits.indices], targets[train_splits.indices]
        holdout_inputs, holdout_targets = inputs[holdout_splits.indices], targets[holdout_splits.indices]

        self.scaler.fit(train_inputs)
        train_inputs = self.scaler.transform(train_inputs)
        holdout_inputs = self.scaler.transform(holdout_inputs)
        holdout_losses = [1e10 for i in range(self.model.num_ensemble)]

        data_idxes = np.random.randint(train_size, size=[self.model.num_ensemble, train_size])
        def shuffle_rows(arr):
            idxes = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[np.arange(arr.shape[0])[:, None], idxes]

        epoch = 0
        cnt = 0
        num_elites = self.model.num_elites

        while True:
            epoch += 1
            self.learn_batch(train_inputs[data_idxes], train_targets[data_idxes], batch_size)
            new_holdout_losses = self.validate(holdout_inputs, holdout_targets)
            holdout_loss = (np.sort(new_holdout_losses)[:num_elites]).mean()

            # shuffle data for each base learner
            data_idxes = shuffle_rows(data_idxes)

            indexes = []
            for i, new_loss, old_loss in zip(range(len(holdout_losses)), new_holdout_losses, holdout_losses):
                improvement = (old_loss - new_loss) / old_loss
                if improvement > 0.01:
                    indexes.append(i)
                    holdout_losses[i] = new_loss
            
            if len(indexes) > 0:
                self.model.update_save(indexes)
                cnt = 0
            else:
                cnt += 1
            
            if cnt >= 5:
                break

        indexes = self.select_elites(holdout_losses)
        self.model.set_elites(indexes)
        return {
            "num_epochs": epoch,
            "elites": indexes,
            "holdout_loss": (np.sort(holdout_losses)[:num_elites]).mean()
        }

    def learn_batch(self, inputs, targets, batch_size):
        self.model.train()
        train_size = inputs.shape[1]
        
        for batch_num in range(int(np.ceil(train_size / batch_size))):
            inputs_batch = inputs[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            targets_batch = targets[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            targets_batch = torch.as_tensor(targets_batch).to(self.model.device)
            
            mean, logvar = self.model(inputs_batch)
            inv_var = torch.exp(-logvar)
            # Average over batch and dim, sum over ensembles.
            mse_loss_inv = (torch.pow(mean - targets_batch, 2) * inv_var).mean(dim=(1, 2))
            var_loss = logvar.mean(dim=(1, 2))
            loss = mse_loss_inv.sum() + var_loss.sum()
            loss = loss + self.model.get_decay_loss()
            loss = loss + 0.01 * self.model.max_logvar.sum() - 0.01 * self.model.min_logvar.sum()

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
    
    @ torch.no_grad()
    def validate(self, inputs, targets):
        self.model.eval()
        targets = torch.as_tensor(targets).to(self.model.device)
        mean, _ = self.model(inputs)
        loss = ((mean - targets) ** 2).mean(dim=(1, 2))
        val_loss = list(loss.cpu().numpy())
        return val_loss
    
    def select_elites(self, metrics):
        pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
        pairs = sorted(pairs, key=lambda x: x[0])
        elites = [pairs[i][1] for i in range(self.model.num_elites)]
        return elites

    def save(self, save_path):
        torch.save(self.model.state_dict(), os.path.join(save_path, "dynamics.pth"))
        self.scaler.save_scaler(save_path)
    
    def load(self, load_path):
        self.model.load_state_dict(torch.load(os.path.join(load_path, "dynamics.pth"), map_location=self.model.device))
        self.scaler.load_scaler(load_path)


def format_samples_for_training(samples):
    obs = samples["s"]
    act = samples["a"]
    next_obs = samples["s_"]
    rew = samples["r"]
    delta_obs = next_obs - obs
    inputs = np.concatenate((obs, act), axis=-1)
    targets = np.concatenate((delta_obs, rew), axis=-1)
    return inputs, targets
