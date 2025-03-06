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
