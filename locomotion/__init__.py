import brax

from .envs import BarkourStraightEnv

print("Registering new environment: BarkourStraightEnv")
brax.envs.register_environment('barkour_straight', BarkourStraightEnv)
