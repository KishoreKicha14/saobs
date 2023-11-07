from gym.spaces import discrete
import gym
from gym import spaces
import numpy as np

class sanwo(gym.Env):


    def __init__(self):
        # Define the action space
        # action is the angle of rotation (-π to π)
        n_actions = 8
        self.action_space = gym.spaces.Discrete(n_actions)
		# You can map discrete actions to actual actions (e.g., angles in radians)
        self.action_mapping = {
    0: -np.pi/4,    # Up
    1: 0.0,         # Right
    2: np.pi/4,     # Down
    3: np.pi/2,     # Left
    4: 3*np.pi/4,   # Up-Left
    5: np.pi,       # Up-Right
    6: -3*np.pi/4,  # Down-Left
    7: -np.pi       # Down-Right
}

        low = np.array([-4.60450186, -4.60450186, -4.60450186, -4.60450186, -4.60450186, -4.60450186], dtype=np.float32)
        # Define the observation space
        # The observation space has 10 dimensions:
        # 1. Agent x-coordinate
        # 2. Agent y-coordinate
		# 3. target x-coordinate
        # 4. target y-coordinate
        high = np.array([4.60450186, 4.60450186, 4.60450186, 4.60450186, 4.60450186, 4.60450186], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)