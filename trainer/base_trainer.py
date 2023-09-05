import numpy as np
from tqdm import tqdm

from env import ENV
from utils.logger import Logger, make_log_dirs


class BASETrainer:
    """ base trainer """
    def __init__(self, args):
        # init env
        self.env = ENV[args.env](args.env_name)
        self.env.seed(args.seed)
        self.env.action_space.seed(args.seed)

        self.eval_env = ENV[args.env](args.env_name)
        self.eval_env.seed(args.seed)
        self.eval_env.action_space.seed(args.seed)

        args.obs_shape = self.env.observation_space.shape
        args.action_dim = int(np.prod(self.env.action_space.shape))

        # logger
        log_dirs = make_log_dirs(args.env_name, args.algo, args.seed, vars(args))
        # key: output file name, value: output handler type
        output_config = {
            "consoleout_backup": "stdout",
            "progress": "csv",
            "tb": "tensorboard"
        }
        self.logger = Logger(log_dirs, output_config)
        self.logger.log_hyperparameters(vars(args))

        # running parameters
        self.n_steps = args.n_steps
        self.start_learning = args.start_learning
        self.update_interval = args.update_interval
        self.batch_size = args.batch_size
        self.eval_interval = args.eval_interval
        self.eval_n_episodes = args.eval_n_episodes
        self.render = args.render
        self.device = args.device
        self.seed = args.seed
        self.args = args

    def _warm_up(self):
        """ randomly sample a lot of transitions into buffer before starting learning """
        obs = self.env.reset()

        # step for {self.start_learning} time-steps
        pbar = tqdm(range(self.start_learning), desc="Warming up")
        for _ in pbar:
            action = self.env.action_space.sample()
            next_obs, reward, done, info = self.env.step(action)
            timeout = info.get("TimeLimit.truncated", False)
            self.memory.store(obs, action, reward, next_obs, done, timeout)

            obs = next_obs
            if done: obs = self.env.reset()

        return obs

    def _eval_policy(self):
        """ evaluate policy """
        episode_rewards = []
        for _ in range(self.eval_n_episodes):
            done = False
            episode_rewards.append(0)
            obs = self.eval_env.reset()
            while not done:
                action = self.agent.act(obs, deterministic=True)
                obs, reward, done, _ = self.eval_env.step(action)
                episode_rewards[-1] += reward
        return episode_rewards
    
    def _eval_value_estimation(self):
        """ evaluate value estimation"""
        value_bias_mean, value_bias_std = [], []
        for _ in range(self.eval_n_episodes):
            rewards = []
            log_probs = []
            value_preds = []
            obs = self.eval_env.reset()
            done = False
            while not done:
                action, log_prob = self.agent.act(obs, deterministic=False, return_logprob=True)
                value_preds.append(self.agent.value(obs, action)[0])
                obs, reward, done, info = self.eval_env.step(action)
                rewards.append(reward)
                log_probs.append(log_prob.flatten()[0])
            
            timeout = info.get("TimeLimit.truncated", False)
            returns = []
            if timeout:
                action, log_prob = self.agent.act(obs, deterministic=False, return_logprob=True)
                next_value = self.agent.value(obs, action)[0]
                returns.append(next_value)
                log_probs.append(log_prob.flatten()[0])
            else:
                returns.append(0)
                log_probs.append(0)
            for r in reversed(rewards):
                returns.append(r + self.agent._gamma * (returns[-1] - self.agent._alpha.cpu().item()*log_probs[-1]))
                log_probs.pop()
            
            returns = np.array(list(reversed(returns[1:]))).flatten()
            value_preds = np.array(value_preds).flatten()

            value_bias_mean.append(((value_preds - returns) / (np.abs(returns.mean())+1e-5)).mean())
            value_bias_std.append(((value_preds - returns) / (np.abs(returns.mean())+1e-5)).std())
        
        return {
            "value_bias_mean": np.mean(value_bias_mean),
            "value_bias_std": np.mean(value_bias_std)
        }