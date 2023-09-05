from config.invertedpendulum import inverted_pendulum_config
from config.hopper import hopper_config
from config.swimmer import swimmer_config
from config.walker2d import walker2d_config
from config.halfcheetah import halfcheetah_config
from config.ant import ant_config
from config.humanoid import humanoid_config


CONFIG = {
    "InvertedPendulum": inverted_pendulum_config,
    "Hopper": hopper_config,
    "Swimmer": swimmer_config,
    "Walker2d": walker2d_config,
    "HalfCheetah": halfcheetah_config,
    "AntTruncatedObs": ant_config,
    "HumanoidTruncatedObs": humanoid_config
}