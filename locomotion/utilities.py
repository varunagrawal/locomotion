"""Module with progress helper"""

from datetime import datetime
from typing import List

from matplotlib import pyplot as plt


class Progress:
    """Helper class to display training progress."""

    def __init__(self, num_timesteps, save_path=None, show=False) -> None:
        self.num_timesteps = num_timesteps

        self.save_path = save_path
        if self.save_path:
            self.save_path.mkdir(parents=True, exist_ok=True)

        self.x_data: List[int] = []
        self.y_data: List[float] = []
        self.ydataerr: List[float] = []
        self.times = [datetime.now()]

        self.show = show
        self.save_id = 0

    def __call__(self, num_steps, metrics):
        self.times.append(datetime.now())
        self.x_data.append(num_steps)
        self.y_data.append(float(metrics['eval/episode_reward']))
        self.ydataerr.append(float(metrics['eval/episode_reward_std']))

        plt.xlim([0, self.num_timesteps * 1.25])

        plt.xlabel('# environment steps')
        plt.ylabel('reward per episode')
        plt.title(f'y={self.y_data[-1]:.3f}')

        plt.errorbar(self.x_data, self.y_data, yerr=self.ydataerr)

        if self.show:
            plt.show()
        else:
            if self.save_path:

                plt.savefig(self.save_path / f"{self.save_id}")
                self.save_id += 1


def add_rand_loc(worldbody, args, end_goal_dist=0):
    """
    Initialize robot to a location normally distributed around the start point,
    and add markers via obstacles of height 0.001m.
    """
    assert args is not None
    # Add boxes
    num_loc = args['num_loc']
    mean = args['mean']
    cov = args['cov'] * np.eye(2)
    num_cp = int(end_goal_dist) + 1

    spawn_locs = np.random.multivariate_normal(mean, cov, num_loc)
    spawn_locs = jnp.array(spawn_locs, dtype=float)

    terrain_res = args['terrain_res']
    obstacle_width = 0.25
    obstacle_length = 0.25
    height = 0.001

    map_width_pixels = int(obstacle_width / terrain_res)
    map_length_pixels = int(obstacle_length / terrain_res)
    terrain_map = jnp.zeros((map_length_pixels, map_width_pixels))
    left_bottom_edge_coord = [-obstacle_width / 2, -obstacle_length / 2]
    for s in range(num_cp):
        x_range = [
            int((s - obstacle_width / 2 - left_bottom_edge_coord[0]) /
                terrain_res),
            int((s + obstacle_width / 2 - left_bottom_edge_coord[0]) /
                terrain_res)
        ]
        y_range = [
            int((-obstacle_length / 2 - left_bottom_edge_coord[1]) /
                terrain_res),
            int((obstacle_length / 2 - left_bottom_edge_coord[1]) /
                terrain_res)
        ]
        # terrain map
        terrain_map = terrain_map.at[y_range[0]:y_range[1],
                                     x_range[0]:x_range[1]].set(height)
        # add boxes
        body = add_box('box', [s, 0, height / 2],
                       [obstacle_width / 2, obstacle_length / 2, height / 2])
        worldbody.append(body)

    return spawn_locs, left_bottom_edge_coord, terrain_map
