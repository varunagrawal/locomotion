"""
Train Barkour model via Soft Actor Critic (SAC) in an environment with no obstacle.

The environment initializes the robot at a random start point
normally distributed around a mean and the robot has to
reach the same goal point.

python scripts/train_barkour_straight_sac.py
"""

import functools
from pathlib import Path

import jax
from brax import envs
from brax.io import model
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac

from locomotion.envs import domain_randomize
from locomotion.utilities import Progress


def main():
    """Main training code."""

    env_name = 'barkour_straight'
    env = envs.get_environment(env_name)

    num_timesteps = int(1e4)

    train_fn = functools.partial(
        sac.train,
        num_timesteps=num_timesteps,
        episode_length=env.eps_length,
        num_envs=1,
        learning_rate=3e-4,
        discounting=0.99,
        batch_size=256,
        num_evals=10,
        normalize_observations=True,
        min_replay_size=1000,
        max_replay_size=num_timesteps,
        network_factory=sac_networks.make_sac_networks,
        randomization_fn=domain_randomize,
    )

    # Reset environments since internals may be overwritten by tracers from the
    # domain randomization function.
    env = envs.get_environment(env_name)
    eval_env = envs.get_environment(env_name)

    save_path = Path("results") / "train_barkour_straight_sac"

    # Create save_path if it doesn't exits
    save_path.mkdir(parents=True, exist_ok=True)

    progress = Progress(num_timesteps=num_timesteps,
                        save_path=save_path / 'graph')
    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=progress,
        eval_env=eval_env,
    )

    # Save and reload params.
    model_path = save_path / 'mjx_brax_quadruped_policy'
    model.save_params(model_path, params)
    print(f"Loading model from {model_path}")
    params = model.load_params(model_path)

    # Visualize trained policy
    eval_env = envs.get_environment(env_name)

    # initialize the state
    rng = jax.random.PRNGKey(0)

    inference_fn = make_inference_fn(params)


if __name__ == "__main__":
    main()
