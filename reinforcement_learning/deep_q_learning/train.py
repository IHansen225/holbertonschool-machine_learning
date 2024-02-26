#!/usr/bin/env python3

import gym
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy

env = gym.make('Breakout-v0')
input_shape = (1,) + env.observation_space.shape
nb_actions = env.action_space.n
mdl = Sequential()
mdl.add(Flatten(input_shape=input_shape))
mdl.add(Dense(128, activation='relu'))
mdl.add(Dense(64, activation='relu'))
mdl.add(Dense(nb_actions, activation='linear'))
memory = SequentialMemory(limit=100000, window_length=1)
p = EpsGreedyQPolicy(eps=0.1)
agent = DQNAgent(model=mdl, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000, target_model_update=1e-2, policy=p)
agent.compile(Adam(lr=1e-3), metrics=['mae'])
agent.fit(env, nb_steps=10000, visualize=False, verbose=2)
agent.save_weights('policy.h5', overwrite=True)
