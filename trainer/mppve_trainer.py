import numpy as np
from tqdm import tqdm

from mppve import MPPVEAgent
from .base_trainer import BASETrainer
from buffer import ReplayBuffer, ReplayBufferForSeqSampling

from components.dynamics import Dynamics
from components.static_fns import STATICFUNC
from components.dynamics_model import EnsembleDynamicsModel


class MPPVETrainer(BASETrainer):
    """ model-based planning policy learning with multi-step plan-value estimation """
    def __init__(self, args):
        super().__init__(args)

        # init dynamics
        dynamics_model = EnsembleDynamicsModel(
            obs_dim=int(np.prod(args.obs_shape)),
            action_dim=args.action_dim,
            hidden_dims=args.dynamics_hidden_dims,
            num_ensemble=args.n_ensembles,
            num_elites=args.n_elites,
            weight_decays=args.dynamics_weight_decay,
            load_model=False,
            device=args.device
        )
        task = args.env_name.split('-')[0]
        static_fns = STATICFUNC[task]
        dynamics = Dynamics(dynamics_model, static_fns)

        self.model_update_interval = args.model_update_interval
        self.actor_freq = args.actor_freq

        # init mppve-agent
        self.agent = MPPVEAgent(
            obs_shape=args.obs_shape,
            hidden_dims=args.ac_hidden_dims,
            action_dim=args.action_dim,
            action_space=self.env.action_space,
            dynamics=dynamics,
            plan_length=args.plan_length,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            batch_size=args.batch_size,
            tau=args.tau,
            gamma=args.gamma,
            alpha=args.alpha,
            auto_alpha=args.auto_alpha,
            alpha_lr=args.alpha_lr,
            target_entropy=args.target_entropy,
            device=args.device
        )
        self.agent.train()

        # planning actions queue
        self.plan_length = args.plan_length
        self.plan_actions = []

        # init replay buffer
        self.memory = ReplayBufferForSeqSampling(
            args.buffer_size, args.obs_shape, args.action_dim, args.plan_length, args.gamma)

        # create memory to store imaginary transitions
        model_rollout_size = args.rollout_batch_size*args.rollout_schedule[2]
        model_buffer_size = int(model_rollout_size*args.model_retain_steps/args.model_update_interval)
        self.model_memory = ReplayBuffer(
            buffer_size=model_buffer_size,
            obs_shape=args.obs_shape,
            action_dim=args.action_dim*self.plan_length
        )

        # func 4 calculate new rollout length (x->y over steps a->b)
        a, b, x, y = args.rollout_schedule
        self.make_rollout_len = lambda it: int(min(max(x+(it-a)/(b-a)*(y-x), x), y))
        # func 4 calculate new model buffer size
        self.make_model_buffer_size = lambda it: \
            int(args.rollout_batch_size*self.make_rollout_len(it) * \
            args.model_retain_steps/args.model_update_interval)

        # other parameters
        self.model_update_interval = args.model_update_interval
        self.rollout_batch_size = args.rollout_batch_size
        self.real_ratio = args.real_ratio
        self.updates_per_step = args.updates_per_step

    def train(self):
        """ train {args.algo} on {args.env} for {args.n_steps} steps"""

        # init
        obs = self._warm_up()

        pbar = tqdm(range(self.n_steps), desc="Training {} on {}.{} (seed: {})".format(
            self.args.algo.upper(), self.args.env.title(), self.args.env_name, self.seed))

        for it in pbar:
            # update (one-step) dynamics model
            if it%self.model_update_interval == 0:
                transitions = self.memory.sample_all()
                model_loss = self.agent.learn_dynamics(transitions)
                self.agent.valid_plan_length = self.make_rollout_len(it)

                # update imaginary memory
                new_model_buffer_size = self.make_model_buffer_size(it)
                if self.model_memory.capacity != new_model_buffer_size:
                    new_buffer = ReplayBuffer(
                        buffer_size=new_model_buffer_size,
                        obs_shape=self.model_memory.obs_shape,
                        action_dim=self.model_memory.action_dim
                    )
                    old_transitions = self.model_memory.sample_all()
                    new_buffer.store_batch(**old_transitions)
                    self.model_memory = new_buffer

                # rollout
                init_transitions = self.memory.sample_nstep4rollout(self.rollout_batch_size)
                rollout_len = self.make_rollout_len(it)
                fake_transitions = self.agent.rollout_transitions(init_transitions, rollout_len)
                self.model_memory.store_batch(**fake_transitions)

                self.logger.log(f"rollout length: {rollout_len},"+
                      f"model buffer capacity: {new_model_buffer_size},"+
                      f"model buffer size: {self.model_memory.size}")

            # step in env
            action = self.agent.act(obs)
            next_obs, reward, done, info = self.env.step(action)
            timeout = info.get("TimeLimit.truncated", False)
            self.memory.store(obs, action, reward, next_obs, done, timeout)

            obs = next_obs
            if done: obs = self.env.reset(); self.plan_actions = []

            # render
            if self.render: self.env.render()

            # update policy
            if it%self.update_interval == 0:
                real_states = []
                update_num = int(self.update_interval*self.updates_per_step)
                update_cnt = 0
                for _ in range(update_num):
                    # sample transitions
                    real_sample_size = int(self.batch_size*self.real_ratio)
                    fake_sample_size = self.batch_size - real_sample_size
                    real_batch = self.memory.sample_nstep(batch_size=real_sample_size)
                    fake_batch = self.model_memory.sample(batch_size=fake_sample_size)
                    transitions = {key: np.concatenate(
                        (real_batch[key], fake_batch[key]), axis=0) for key in real_batch.keys()}

                    real_states.append(real_batch["s"])

                    # update
                    critic_learning_info = self.agent.learn_critic(**transitions)
                    critic_loss = critic_learning_info["critic_loss"]
                    update_cnt += 1
                    
                    if update_cnt % self.actor_freq == 0:
                        real_states = np.concatenate(real_states, axis=0)
                        actor_learning_info = self.agent.learn_actor(real_states)
                        actor_loss = actor_learning_info["actor_loss"]
                        alpha = actor_learning_info["alpha"]
                        real_states = []

            # evaluate policy
            if it%self.eval_interval == 0:
                episode_rewards = np.mean(self._eval_policy())
                self.logger.logkv("loss/model", model_loss)
                self.logger.logkv("loss/actor", actor_loss)
                self.logger.logkv("loss/critic", critic_loss)
                self.logger.logkv("alpha", alpha)
                self.logger.logkv("eval/episode_rewards", np.mean(episode_rewards))

                value_bias_info = self._eval_value_estimation()
                self.logger.logkv("eval/value_bias_mean", value_bias_info["value_bias_mean"])
                self.logger.logkv("eval/value_bias_std", value_bias_info["value_bias_std"])

                self.logger.set_timestep(it)
                self.logger.dumpkvs()

            pbar.set_postfix(
                alpha=alpha,
                model_loss=model_loss,
                actor_loss=actor_loss, 
                critic_loss=critic_loss, 
                eval_reward=episode_rewards
            )
