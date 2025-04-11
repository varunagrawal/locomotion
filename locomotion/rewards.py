import jax
from brax import base, math
from brax.base import Motion, Transform
from jax import numpy as jnp
from ml_collections import config_dict


def get_barkour_config():
    """Returns reward config for barkour quadruped environment."""

    def get_default_rewards_config():
        default_config = config_dict.ConfigDict(
            dict(
                # The coefficients for all reward terms used for training. All
                # physical quantities are in SI units, if no otherwise specified,
                # i.e. joint positions are in rad, positions are measured in meters,
                # torques in Nm, and time in seconds, and forces in Newtons.
                scales=config_dict.ConfigDict(
                    dict(
                        # Specify the cofficients of each reward function.
                        # The final reward will be computed as reward = coeff * reward_function
                        # e.g.: 'tracking_lin_vel': 1.0,

                        # Goal reaching
                        goal_reaching=10.0,
                        # Tracking rewards are computed using exp(-delta^2/sigma)
                        # sigma can be a hyperparameters to tune.
                        # Track the base x-y velocity (no z-velocity tracking.)
                        tracking_lin_vel=1.5,

                        # Track the angular velocity along z-axis, i.e. yaw rate.
                        tracking_ang_vel=0.8,
                        # Below are regularization terms, we roughly divide the
                        # terms to base state regularizations, joint
                        # regularizations, and other behavior regularizations.
                        # Penalize the base velocity in z direction, L2 penalty.
                        lin_vel_z=0.0,
                        # Penalize the base roll and pitch rate. L2 penalty.
                        ang_vel_xy=-0.05,
                        # Penalize non-zero roll and pitch angles. L2 penalty.
                        orientation=-5.0,
                        # L2 regularization of joint torques, |tau|^2.
                        torques=-0.0002,
                        # Penalize the change in the action and encourage smooth
                        # actions. L2 regularization |action - last_action|^2
                        action_rate=-0.01,
                        # Encourage long swing steps.  However, it does not
                        # encourage high clearances.
                        feet_air_time=0.0,
                        # Encourage no motion at zero command, L2 regularization
                        # |q - q_default|^2.
                        stand_still=0.0,  # -0.5,
                        # Penalizing foot slipping on the ground.
                        foot_slip=-0.0,  # -0.1
                        # Early termination penalty.
                        termination=-1.0,
                    )),
                # Tracking reward = exp(-error^2/sigma).
                tracking_sigma=0.25,
                goal_reaching_sigma=0.5))
        return default_config

    default_config = config_dict.ConfigDict(
        dict(rewards=get_default_rewards_config(), ))

    return default_config


class BaseRewards:
    """Base class for all rewarder classes."""

    def __init__(self, reward_config):
        self._reward_config = reward_config

    def get_config(self):
        """Accessor for the rewards configuration."""
        return self._reward_config


class QuadrupedRewards(BaseRewards):
    """Class with common quadruped rewards as seen in Rudin et. al. and similar papers."""

    def _reward_goal_reaching(self, goal_loc: jax.Array, x: Transform,
                              torso_idx: int):
        """Reward for getting closer to the goal location"""
        pos = x.pos[torso_idx - 1, :2]
        dist = jnp.linalg.norm(goal_loc - pos)
        dist_rew = jnp.exp(-dist**2 /
                           self._reward_config.rewards.goal_reaching_sigma)

        return dist_rew

    def _reward_lin_vel_z(self, xd: Motion) -> jax.Array:
        """Penalize z axis base linear velocity"""
        return jnp.square(xd.vel[0, 2])

    def _reward_ang_vel_xy(self, xd: Motion) -> jax.Array:
        """Penalize xy axes base angular velocity"""
        return jnp.sum(jnp.square(xd.ang[0, :2]))

    def _reward_orientation(self, x: Transform) -> jax.Array:
        """Penalize non flat base orientation"""
        up = jnp.array([0.0, 0.0, 1.0])
        rot_up = math.rotate(up, x.rot[0])
        return jnp.sum(jnp.square(rot_up[:2]))

    def _reward_torques(self, torques: jax.Array) -> jax.Array:
        """Penalize torques"""
        return jnp.sqrt(jnp.sum(jnp.square(torques))) + jnp.sum(
            jnp.abs(torques))

    def _reward_action_rate(self, act: jax.Array,
                            last_act: jax.Array) -> jax.Array:
        """Penalize changes in actions"""
        return jnp.sum(jnp.square(act - last_act))

    def _reward_tracking_ang_vel(self, commands: jax.Array, x: Transform,
                                 xd: Motion) -> jax.Array:
        """Tracking of angular velocity commands (yaw)"""
        base_ang_vel = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
        ang_vel_error = jnp.square(commands[2] - base_ang_vel[2])
        return jnp.exp(-ang_vel_error /
                       self._reward_config.rewards.tracking_sigma)

    def _reward_feet_air_time(self, air_time: jax.Array,
                              first_contact: jax.Array,
                              commands: jax.Array) -> jax.Array:
        """Reward air time."""
        rew_air_time = jnp.sum((air_time - 0.1) * first_contact)
        rew_air_time *= (math.normalize(commands[:2])[1]
                         > 0.05)  # no reward for zero command
        return rew_air_time

    def _reward_foot_slip(self, pipeline_state: base.State,
                          contact_filt: jax.Array, feet_site_id,
                          lower_leg_body_id) -> jax.Array:
        """Get velocities at feet which are offset from lower legs"""
        pos = pipeline_state.site_xpos[feet_site_id]  # feet position
        feet_offset = pos - pipeline_state.xpos[lower_leg_body_id]
        # pytype: enable=attribute-error
        offset = base.Transform.create(pos=feet_offset)
        foot_indices = lower_leg_body_id - 1  # we got rid of the world body
        foot_vel = offset.vmap().do(pipeline_state.xd.take(foot_indices)).vel

        # Penalize large feet velocity for feet that are in contact with the ground.
        return jnp.sum(
            jnp.square(foot_vel[:, :2]) * contact_filt.reshape((-1, 1)))

    def _reward_termination(self, done: jax.Array, step: jax.Array,
                            eps_length: int) -> jax.Array:
        """Reward if end of episode."""
        return done & (step < eps_length)


class BarkourRewards(QuadrupedRewards):
    """
    Reward functions for the Barkour Env.
    You can also refer the MJX tutorial's BarkourEnv class code.
    https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb
    e.g.: def _reward_reward_name(self, args)
    """

    def compute_rewards(self, state, pipeline_state, obs, action, torso_idx,
                        command_vel, first_contact, joint_angles,
                        default_joint_angles, contact_filt_cm, feet_site_id,
                        lower_leg_body_id, eps_length, done):
        """Compute all the rewards"""
        x, xd = pipeline_state.x, pipeline_state.xd

        rewards = {
            # Specify reward here in the following format:
            # 'reward_name': self._reward_reward_name(args),
            # e.g.: 'tracking_lin_vel': self._reward_tracking_lin_vel(state.info['command'], x, xd),
            # NOTE: the "reward_name" should be the same as
            # the key in the reward_config.rewards.scales
            'goal_reaching':
            self._reward_goal_reaching(state.info['goal_loc'],
                                       x,
                                       torso_idx=torso_idx),
            'tracking_lin_vel':
            self._reward_tracking_lin_vel(state.info['command'], x, xd),
            'tracking_ang_vel':
            self._reward_tracking_ang_vel(state.info['command'], x, xd),
            'lin_vel_z':
            self._reward_lin_vel_z(xd),
            'ang_vel_xy':
            self._reward_ang_vel_xy(xd),
            'orientation':
            self._reward_orientation(x),
            'torques':
            self._reward_torques(pipeline_state.qfrc_actuator),
            'action_rate':
            self._reward_action_rate(action, state.info['last_act']),
            'feet_air_time':
            self._reward_feet_air_time(
                state.info['feet_air_time'],
                first_contact,
                state.info['command'],
            ),
            'stand_still':
            self._reward_stand_still(state.info['command'], joint_angles,
                                     default_joint_angles),
            'foot_slip':
            self._reward_foot_slip(pipeline_state,
                                   contact_filt_cm,
                                   feet_site_id=feet_site_id,
                                   lower_leg_body_id=lower_leg_body_id),
            'termination':
            self._reward_termination(done,
                                     state.info['step'],
                                     eps_length=eps_length),
        }
        return rewards

    def _reward_tracking_lin_vel(self, commands: jax.Array, x: Transform,
                                 xd: Motion) -> jax.Array:
        # Tracking of linear velocity commands (xy axes)
        local_vel = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
        lin_vel_error = jnp.sum(jnp.square(commands[:2] - local_vel[:2]))
        lin_vel_reward = jnp.exp(-lin_vel_error /
                                 self._reward_config.rewards.tracking_sigma)
        return lin_vel_reward

    def _reward_stand_still(self, commands: jax.Array, joint_angles: jax.Array,
                            default_joint_angles: jax.Array) -> jax.Array:
        # Penalize motion at zero commands
        return jnp.sum(jnp.abs(joint_angles - default_joint_angles)) * (
            math.normalize(commands[:2])[1] < 0.1)


class BarkourStraightRewards(QuadrupedRewards):
    """
    Reward functions for the BarkourStraight environment.
    """

    def compute_rewards(self, state, pipeline_state, obs, action, torso_idx,
                        command_vel, first_contact, joint_angles,
                        default_joint_angles, contact_filt_cm, feet_site_id,
                        lower_leg_body_id, eps_length, done):
        """Compute all the rewards"""
        x, xd = pipeline_state.x, pipeline_state.xd

        rewards = {
            # Specify reward here in the following format:
            # 'reward_name': self._reward_reward_name(args),
            # e.g.: 'tracking_lin_vel': self._reward_tracking_lin_vel(state.info['command'], x, xd),
            # NOTE: the "reward_name" should be the same as
            # the key in the reward_config.rewards.scales
            'goal_reaching':
            self._reward_goal_reaching(state.info['goal_loc'],
                                       x,
                                       torso_idx=torso_idx),
            # Varun: Taken from the BarkourEnv class in the DeepMind Mujoco tutorial
            'tracking_lin_vel':
            self._reward_tracking_lin_vel(x,
                                          xd,
                                          state.info['goal_loc'],
                                          command_vel=command_vel,
                                          torso_idx=torso_idx),
            'tracking_ang_vel':
            self._reward_tracking_ang_vel(state.info['command'], x, xd),
            'lin_vel_z':
            self._reward_lin_vel_z(xd),
            'ang_vel_xy':
            self._reward_ang_vel_xy(xd),
            'orientation':
            self._reward_orientation(x),
            'torques':
            self._reward_torques(pipeline_state.qfrc_actuator),
            'action_rate':
            self._reward_action_rate(action, state.info['last_act']),
            'feet_air_time':
            self._reward_feet_air_time(
                state.info['feet_air_time'],
                first_contact,
                state.info['command'],
            ),
            'stand_still':
            self._reward_stand_still(state.info['command'], joint_angles,
                                     state.info['goal_loc'],
                                     x.pos[torso_idx - 1, :2],
                                     default_joint_angles),
            'termination':
            self._reward_termination(done,
                                     state.info['step'],
                                     eps_length=eps_length),
            'foot_slip':
            self._reward_foot_slip(pipeline_state,
                                   contact_filt_cm,
                                   feet_site_id=feet_site_id,
                                   lower_leg_body_id=lower_leg_body_id),
        }
        return rewards

    def _reward_goal_satisfying(self, goal_loc: jax.Array,
                                robot_pose: jax.Array) -> jax.Array:
        """Reward for being at the goal location"""
        return jnp.exp(
            -jnp.square(jnp.linalg.norm(goal_loc - robot_pose)) /
            self._reward_config.rewards.goal_satisfying_sigma)  # tight sigma

    def _reward_tracking_lin_vel(self, x: Transform, xd: Motion,
                                 goal_loc: jax.Array, command_vel: jax.Array,
                                 torso_idx: int) -> jax.Array:
        # Tracking of linear velocity commands (xy axes)
        robot_pose = x.pos[torso_idx - 1, :2]
        u = (goal_loc - robot_pose) / jnp.linalg.norm(goal_loc - robot_pose)
        vel_coeff = 1 - jnp.exp(
            -jnp.square(jnp.linalg.norm(goal_loc - robot_pose)) /
            self._reward_config.rewards.tracking_sigma)
        local_vel = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
        lin_vel_error = jnp.sum(
            jnp.square(vel_coeff * command_vel * u - local_vel[:2]))
        lin_vel_reward = jnp.exp(-lin_vel_error /
                                 self._reward_config.rewards.tracking_sigma)
        return lin_vel_reward

    def _reward_stand_still(self, commands: jax.Array, joint_angles: jax.Array,
                            goal_loc: jax.Array, robot_pose: jax.Array,
                            default_joint_angles: jax.Array) -> jax.Array:
        # Penalize motion at zero commands
        vel_coeff = 1 - jnp.exp(
            -jnp.square(jnp.linalg.norm(goal_loc - robot_pose)) /
            self._reward_config.rewards.tracking_sigma)
        return jnp.sum(jnp.abs(joint_angles - default_joint_angles)) * (
            math.normalize(vel_coeff * commands[:2])[1] < 0.1)
