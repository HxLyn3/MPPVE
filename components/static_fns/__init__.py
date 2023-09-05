from .hopper import StaticFns as HopperStaticFns
from .swimmer import StaticFns as SwimmerStaticFns
from .walker2d import StaticFns as Walker2dStaticFns
from .halfcheetah import StaticFns as HalfcheetahStaticFns
from .inverted_pendulum import StaticFns as InvertedPendulumFns
from .ant_truncated_obs import StaticFns as AntTruncatedObsStaticFns
from .humanoid_truncated_obs import StaticFns as HumanoidTruncatedObsStaticFns

STATICFUNC = {
    "Hopper": HopperStaticFns,
    "Swimmer": SwimmerStaticFns,
    "Walker2d": Walker2dStaticFns,
    "HalfCheetah": HalfcheetahStaticFns,
    "InvertedPendulum": InvertedPendulumFns,
    "AntTruncatedObs": AntTruncatedObsStaticFns,
    "HumanoidTruncatedObs": HumanoidTruncatedObsStaticFns
}