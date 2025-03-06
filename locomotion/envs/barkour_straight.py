"""Barkour environment class with no obstacles."""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, List, Sequence

import jax
import mujoco
import numpy as np
from brax import base, math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
from fill.envs.add_obstacles import add_rand_loc
from fill.rewards import get_barkour_config
from fill.rewards.barkour import BarkourStraightRewards
from jax import numpy as jnp


def domain_randomize(sys, rng):
    """Randomizes the mjx.Model."""

    @jax.vmap
    def rand(rng):
        _, key = jax.random.split(rng, 2)
        # friction
        friction = jax.random.uniform(key, (1, ), minval=0.6, maxval=1.4)
        friction = sys.geom_friction.at[:, 0].set(friction)
        # actuator
        _, key = jax.random.split(key, 2)
        gain_range = (-5, 5)
        param = jax.random.uniform(
            key, (1, ), minval=gain_range[0],
            maxval=gain_range[1]) + sys.actuator_gainprm[:, 0]
        gain = sys.actuator_gainprm.at[:, 0].set(param)
        bias = sys.actuator_biasprm.at[:, 1].set(-param)
        return friction, gain, bias

    friction, gain, bias = rand(rng)

    in_axes = jax.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace({
        'geom_friction': 0,
        'actuator_gainprm': 0,
        'actuator_biasprm': 0,
    })

    sys = sys.tree_replace({
        'geom_friction': friction,
        'actuator_gainprm': gain,
        'actuator_biasprm': bias,
    })

    return sys, in_axes


def get_absolute_path(path: str):
    """Return absolute path of the file given by `path`."""
    p = Path(__file__).parent / path
    return p.absolute()


BARKOUR_ROOT_PATH = epath.Path(
    get_absolute_path('mujoco_menagerie/google_barkour_vb'))


class BarkourStraightEnv(PipelineEnv):
    """
    A simple environment with markings in a straight line.
    The objective is to get a quadruped to walk in a straight line to a goal point
    from a randomly selected start point.
    """

    def __init__(
            self,
            obs_noise: float = 0.05,
            action_scale: float = 0.3,
            kick_vel: float = 0.05,
            rewarder=BarkourStraightRewards(get_barkour_config()),
            **kwargs,
    ):
        # barkour update
        path = BARKOUR_ROOT_PATH / 'barkour_vb_mjx.xml'

        tree = ET.parse(path)
        root = tree.getroot()

        # Define the mapping of body names to new geom names
        new_names = {
            "leg_front_left": "foot_front_left",
            "leg_hind_left": "foot_hind_left",
            "leg_front_right": "foot_front_right",
            "leg_hind_right": "foot_hind_right",
        }

        # Iterate over all body elements
        for body in root.findall(".//body"):
            body_name = body.get("name")
            if body_name in new_names:
                # Find all geom elements within this body with the class "bkvb/collision/foot"
                for geom in body.findall(
                        ".//geom[@class='bkvb/collision/foot']"):
                    # Update the name attribute of the geom
                    geom.set("name", new_names[body_name])
        tree.write(BARKOUR_ROOT_PATH / 'barkour_vb_mjx_geom_feet_name.xml')

        # scene update
        path = epath.Path(BARKOUR_ROOT_PATH / 'scene_mjx.xml')

        tree = ET.parse(path)
        root = tree.getroot()

        # Find the <include> element and update its 'file' attribute
        include_tag = root.find('include')
        if include_tag is not None:
            include_tag.set('file', 'barkour_vb_mjx_geom_feet_name.xml')

        # Find the worldbody element
        worldbody = root.find('worldbody')
        self.terrain_res = 0.01

        self.end_goal_dist = 6
        self.starting_offset = 3

        self.spawn_locs, self.left_bottom_edge_coord, self.terrain_map = \
            self.init_environment(worldbody)

        # Save the modified MJCF file
        tree.write(BARKOUR_ROOT_PATH / 'scene_mjx_terrain.xml')

        # re-assign path to new xml file
        path = BARKOUR_ROOT_PATH / 'scene_mjx_terrain.xml'

        self.command_vel = 1.0
        self._dt = 0.02  # this environment is 50 fps
        total_time = 2 * self.starting_offset / self.command_vel + 2
        self.eps_length = int(total_time / self._dt)
        sys = mjcf.load(path)
        sys = sys.tree_replace({'opt.timestep': 0.004})

        # override menagerie params for smoother policy
        sys = sys.replace(
            dof_damping=sys.dof_damping.at[6:].set(0.5239),
            actuator_gainprm=sys.actuator_gainprm.at[:, 0].set(35.0),
            actuator_biasprm=sys.actuator_biasprm.at[:, 1].set(-35.0),
        )

        n_frames = kwargs.pop('n_frames', int(self._dt / sys.opt.timestep))
        super().__init__(sys, backend='mjx', n_frames=n_frames)

        self.rewarder = rewarder
        self.reward_config = self.rewarder.get_config()
        # set custom from kwargs
        for k, v in kwargs.items():
            if k.endswith('_scale'):
                self.reward_config.rewards.scales[k[:-6]] = v

        self._torso_idx = mujoco.mj_name2id(sys.mj_model,
                                            mujoco.mjtObj.mjOBJ_BODY.value,
                                            'torso')
        self._action_scale = action_scale
        self._obs_noise = obs_noise
        self._init_q = jnp.array(sys.mj_model.keyframe('home').qpos)
        self._default_pose = sys.mj_model.keyframe('home').qpos[7:]
        self.lowers = jnp.array([-0.7, -1.0, 0.05] * 4)
        self.uppers = jnp.array([0.52, 2.1, 2.1] * 4)
        feet_site = [
            'foot_front_left',
            'foot_hind_left',
            'foot_front_right',
            'foot_hind_right',
        ]
        feet_site_id = [
            mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, f)
            for f in feet_site
        ]
        self.l_fr_foot_id = mujoco.mj_name2id(sys.mj_model,
                                              mujoco.mjtObj.mjOBJ_GEOM.value,
                                              'foot_front_left')
        self.l_hd_foot_id = mujoco.mj_name2id(sys.mj_model,
                                              mujoco.mjtObj.mjOBJ_GEOM.value,
                                              'foot_hind_left')
        self.r_fr_foot_id = mujoco.mj_name2id(sys.mj_model,
                                              mujoco.mjtObj.mjOBJ_GEOM.value,
                                              'foot_front_right')
        self.r_hd_foot_id = mujoco.mj_name2id(sys.mj_model,
                                              mujoco.mjtObj.mjOBJ_GEOM.value,
                                              'foot_hind_right')
        self.foot_geom_ids = jnp.array([
            self.l_fr_foot_id, self.l_hd_foot_id, self.r_fr_foot_id,
            self.r_hd_foot_id
        ])
        assert not any(id_ == -1 for id_ in feet_site_id), 'Site not found.'
        self._feet_site_id = np.array(feet_site_id)
        lower_leg_body = [
            'lower_leg_front_left',
            'lower_leg_hind_left',
            'lower_leg_front_right',
            'lower_leg_hind_right',
        ]
        lower_leg_body_id = [
            mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, l)
            for l in lower_leg_body
        ]
        assert not any(id_ == -1
                       for id_ in lower_leg_body_id), 'Body not found.'
        self._lower_leg_body_id = np.array(lower_leg_body_id)
        self._foot_radius = 0.0175
        self._nv = sys.nv

        self.curriculum_level = 0  # this should be constant

    def get_height(self, x, y):
        """Get terrain height at location (x, y)"""
        map_x = x - self.left_bottom_edge_coord[0]
        map_y = y - self.left_bottom_edge_coord[1]
        map_x_pix = (map_x / self.terrain_res).astype(int)
        map_y_pix = (map_y / self.terrain_res).astype(int)
        x_out_of_range = ((map_x_pix < 0) |
                          (map_x_pix >= self.terrain_map.shape[1]))
        y_out_of_range = ((map_y_pix < 0) |
                          (map_y_pix >= self.terrain_map.shape[0]))
        return jnp.where(x_out_of_range | y_out_of_range, 0,
                         self.terrain_map[map_y_pix, map_x_pix])

    def init_environment(self, worldbody):
        """
        Initialize environment to spawn the robot at randomly distributed locations.
        """
        return add_rand_loc(worldbody, self.end_goal_dist)

    def update_spawn_location(self, state):
        """Samples different locations for the robot spawn point."""
        current_level = jnp.mod(state.info['new_level'] + 1,
                                self.spawn_locs.shape[0])
        state.info['new_level'] = current_level

        new_spawn_loc = self.spawn_locs[current_level, :]
        state.info['new_spawn_loc'] = new_spawn_loc
        return state

    def process_state_after_step(self, state: State) -> State:
        """
        Process the state to update the curriculum after an env step is taken.
        """
        # update spawn location
        state = self.update_spawn_location(state)
        return state

    def reset(self, rng: jax.Array) -> State:  # pytype: disable=signature-mismatch
        """
        This reset function is only called in the beginning of the training,
        not during the training.
        """
        # This is should look like for one env. It is vmapped in `ppo``.
        rng, _ = jax.random.split(rng)
        # set new goal location
        spawn_loc = self.spawn_locs[self.curriculum_level]
        goal_loc = jnp.array(
            [spawn_loc[0] + 2 * self.starting_offset, spawn_loc[1]])

        pipeline_state = self.pipeline_init(self._init_q, jnp.zeros(self._nv))

        state_info = {
            'rng': rng,
            'last_act': jnp.zeros(12),
            'last_vel': jnp.zeros(12),
            'command': jnp.array([self.command_vel, 0, 0]),
            'last_contact': jnp.zeros(4, dtype=bool),
            'feet_air_time': jnp.zeros(4),
            'rewards': {
                k: 0.0
                for k in self.reward_config.rewards.scales.keys()
            },
            'step': 0,
            'current_level':
            self.curriculum_level,  # change when reset happens
            'goal_loc': goal_loc,  # change when reset happens
            'new_goal_loc':
            goal_loc,  # updated every step and used in curriculum update when reset
            'new_spawn_loc':
            spawn_loc,  # updated every step and used in curriculum update when reset
            'new_level': self.
            curriculum_level,  # updated every step and used in curriculum update when reset
            'not_done': jnp.array(1.0),
        }

        obs_history = jnp.zeros(15 * 31)  # store 15 steps of history
        obs = self._get_obs(pipeline_state, state_info, obs_history)
        reward, done = jnp.zeros(2)
        metrics = {'total_dist': 0.0}
        for k in state_info['rewards']:
            metrics[k] = state_info['rewards'][k]
        state = State(pipeline_state, obs, reward, done, metrics, state_info)
        return state

    def physics_step(self, state, action):
        """Physics step"""
        motor_targets = self._default_pose + action * self._action_scale
        motor_targets = jnp.clip(motor_targets, self.lowers, self.uppers)
        pipeline_state = self.pipeline_step(state.pipeline_state,
                                            motor_targets)
        return pipeline_state

    def compute_foot_data(self, state: State, pipeline_state: State):
        """Helper to compute foot data"""
        # foot contact data based on z-position
        foot_pos = pipeline_state.site_xpos[self._feet_site_id]
        foot0_terrain_h = self.get_height(foot_pos[0, 0], foot_pos[0, 1])
        foot1_terrain_h = self.get_height(foot_pos[1, 0], foot_pos[1, 1])
        foot2_terrain_h = self.get_height(foot_pos[2, 0], foot_pos[2, 1])
        foot3_terrain_h = self.get_height(foot_pos[3, 0], foot_pos[3, 1])
        feet_terrain_h = jnp.array([
            foot0_terrain_h, foot1_terrain_h, foot2_terrain_h, foot3_terrain_h
        ])
        foot_contact_z = foot_pos[:, 2] - feet_terrain_h - self._foot_radius
        contact = foot_contact_z < 1e-3  # a mm or less off the floor
        contact_filt_mm = contact | state.info['last_contact']
        contact_filt_cm = (foot_contact_z < 3e-2) | state.info['last_contact']
        first_contact = (state.info['feet_air_time'] > 0) * contact_filt_mm
        return contact, contact_filt_cm, contact_filt_mm, first_contact

    def step(self, state: State, action: jax.Array) -> State:  # pytype: disable=signature-mismatch
        # In env's step function, consider it is one env.
        rng, _ = jax.random.split(state.info['rng'], 2)

        pipeline_state = self.physics_step(state, action)

        x = pipeline_state.x

        # observation data
        obs = self._get_obs(pipeline_state, state.info, state.obs)
        joint_angles = pipeline_state.q[7:]
        joint_vel = pipeline_state.qd[6:]

        contact, contact_filt_cm, contact_filt_mm, first_contact = self.compute_foot_data(
            state, pipeline_state)

        state.info['feet_air_time'] += self.dt

        # done if joint limits are reached or robot is falling
        up = jnp.array([0.0, 0.0, 1.0])
        done = jnp.dot(math.rotate(up, x.rot[self._torso_idx - 1]), up) < 0
        done |= jnp.any(joint_angles < self.lowers)
        done |= jnp.any(joint_angles > self.uppers)
        done |= pipeline_state.x.pos[self._torso_idx - 1, 2] < 0.18

        # reward
        rewards = self.rewarder.compute_rewards(
            state, pipeline_state, obs, action, self._torso_idx,
            self.command_vel, first_contact, joint_angles, self._default_pose,
            contact_filt_cm, self._feet_site_id, self._lower_leg_body_id,
            self.eps_length, done)
        rewards = {
            k: v * self.reward_config.rewards.scales[k]
            for k, v in rewards.items()
        }
        reward = jnp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        # state management
        state.info['last_act'] = action
        state.info['last_vel'] = joint_vel
        state.info['feet_air_time'] *= ~contact_filt_mm
        state.info['last_contact'] = contact
        state.info['rewards'] = rewards
        state.info['step'] += 1
        state.info['rng'] = rng
        state.info['not_done'] = jnp.invert(done).astype(float)

        state = self.process_state_after_step(state)

        # reset the step counter when done
        state.info['step'] = jnp.where(
            done | (state.info['step'] > self.eps_length), 0,
            state.info['step'])

        # log total displacement as a proxy metric
        state.metrics['total_dist'] = math.normalize(x.pos[self._torso_idx -
                                                           1])[1]
        state.metrics.update(state.info['rewards'])

        done = jnp.float32(done)
        state = state.replace(pipeline_state=pipeline_state,
                              obs=obs,
                              reward=reward,
                              done=done)
        return state

    def _get_obs(
        self,
        pipeline_state: base.State,
        state_info: dict[str, Any],
        obs_history: jax.Array,
    ) -> jax.Array:
        inv_torso_rot = math.quat_inv(pipeline_state.x.rot[0])
        local_rpyrate = math.rotate(pipeline_state.xd.ang[0], inv_torso_rot)

        obs = jnp.concatenate(
            [  # if observation changes, be careful when reset in AutoReset class.
                jnp.array([local_rpyrate[2]]) * 0.25,  # yaw rate
                math.rotate(jnp.array([0, 0, -1]),
                            inv_torso_rot),  # projected gravity
                pipeline_state.q[7:] - self._default_pose,  # motor angles
                state_info['last_act'],  # last action
                pipeline_state.x.pos[self._torso_idx - 1, :2] -
                state_info['goal_loc'],  # difference to goal location
            ])

        # clip, noise
        obs = jnp.clip(obs, -100.0,
                       100.0) + self._obs_noise * jax.random.uniform(
                           state_info['rng'], obs.shape, minval=-1, maxval=1)
        # stack observations through time
        obs = jnp.roll(obs_history, obs.size).at[:obs.size].set(obs)

        return obs

    def render(self,
               trajectory: List[base.State],
               camera: str | None = None) -> Sequence[np.ndarray]:
        camera = camera or 'track'
        return super().render(trajectory, camera=camera)
