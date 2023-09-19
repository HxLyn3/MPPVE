import os
import random
import argparse

import torch
import numpy as np

from config import CONFIG
from trainer.mppve_trainer import MPPVETrainer


def get_args():
    parser = argparse.ArgumentParser(description="DRL")

    # environment settings
    parser.add_argument("--env", type=str, default="mujoco")
    parser.add_argument("--env-name", type=str, default="Hopper-v3")

    # algorithm parameters
    parser.add_argument("--algo", type=str, default="mppve")
    parser.add_argument("--ac-hidden-dims", type=int, nargs='*', default=[256, 256])
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--plan-length", type=int, default=3)
    # (for sac)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", type=bool, default=True)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--target-entropy", type=int, default=-1)

    # replay-buffer parameters
    parser.add_argument("--buffer-size", type=int, default=int(1e6))

    # dynamics-model parameters
    parser.add_argument("--dynamics-hidden-dims", type=int, nargs='*', default=[200, 200, 200, 200])
    parser.add_argument("--dynamics-weight-decay", type=float, nargs='*', default=[2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4])
    parser.add_argument("--n-ensembles", type=int, default=7)
    parser.add_argument("--n-elites", type=int, default=5)
    parser.add_argument("--rollout-batch-size", type=int, default=int(1e5))
    parser.add_argument("--rollout-schedule", type=int, nargs='*', default=[int(2e4), int(5e4), 1, 4])
    parser.add_argument("--model-update-interval", type=int, default=250)
    parser.add_argument("--model-retain-steps", type=int, default=1000)
    parser.add_argument("--real-ratio", type=float, default=0.05)

    # running parameters
    parser.add_argument("--n-steps", type=int, default=int(1e5))
    parser.add_argument("--start-learning", type=int, default=int(5e3))
    parser.add_argument("--update-interval", type=int, default=1)
    parser.add_argument("--updates-per-step", type=int, default=20)
    parser.add_argument("--actor-freq", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-interval", type=int, default=int(1e3))
    parser.add_argument("--eval-n-episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    return args

def main():
    args = vars(get_args())
    config = CONFIG[args["env_name"].split('-')[0]]
    for k, v in config.items():
        args[k] = v
    args = argparse.Namespace(**args)

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    trainer = MPPVETrainer(args)
    trainer.train()

if __name__ == "__main__":
    main()
