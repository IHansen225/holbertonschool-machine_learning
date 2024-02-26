#!/usr/bin/env python3

import gym

def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
        load the FrozenLakeEnv environment
    """
    return gym.make('FrozenLake-v1', map_name=map_name, desc=desc,
                    is_slippery=is_slippery)
