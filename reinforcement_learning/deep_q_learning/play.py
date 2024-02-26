#!/usr/bin/env python3

from keras.models import Sequential
import gym
from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy

env = gym.make('Breakout-v0')
input_shape = (1,) + env.observation_space.shape
nb_actions = env.action_space.n
mdl = Sequential()
mdl.add(Flatten(input_shape=input_shape))
mdl.add(Dense(128, activation='relu'))
mdl.add(Dense(64, activation='relu'))
mdl.add(Dense(nb_actions, activation='linear'))
p = GreedyQPolicy()
agent = DQNAgent(mdl=mdl, nb_actions=nb_actions, memory=None, nb_steps_warmup=1000, target_mdl_update=1e-2, policy=p)
agent.load_weights('policy.h5')
agent.test(env, nb_episodes=5, visualize=True)
