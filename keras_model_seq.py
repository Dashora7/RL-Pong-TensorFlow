# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 21:18:58 2019

@author: Aimee
"""
""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import gym

# import necessary modules from keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# used for saving and loading
from datetime import datetime
from tensorflow.keras import callbacks
import os

# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
gamma = 0.99 # discount factor for reward
resume = False # resume from previous checkpoint?

# model initialization
D = 80 * 80 # input dimensionality: 80x80 grid

# creates a generic neural network architecture
model = Sequential()

# hidden layer takes a pre-processed frame as input, and has 200 units
model.add(Dense(units=H,input_dim=D, activation='relu', kernel_initializer='glorot_uniform'))

# output layer
model.add(Dense(units=1, activation='sigmoid', kernel_initializer='RandomNormal'))

# compile the model using traditional Machine Learning losses and optimizers
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Methods from Karpathy
def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

def discount_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    r = np.array(r)
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    discounted_r -= np.mean(discounted_r) #normalizing the result
    discounted_r /= np.std(discounted_r) #idem
    return discounted_r

# gym initialization
env = gym.make("Pong-v0")
observation = env.reset()
prev_input = None

# initialization of variables used in the main loop
xs, ys, rewards = [],[],[]
running_reward = None
reward_sum = 0
episode_number = 1

# Macros
UP_ACTION = 2
DOWN_ACTION = 3

"""
# add a callback tensorboard object to visualize learning
log_dir = './log' + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
tbCallBack = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)
"""

# load pre-trained model if exist
if (resume and os.path.isfile('Keras\\SmartPong_weights.h5')):
    print("loading previous weights")
    model.load_weights('SmartPong_weights.h5')
    
n_steps = 1
round_n = 1
# main loop
while (True):

    # preprocess the observation, set input as difference between images
    cur_input = prepro(observation)
    x = cur_input - prev_input if prev_input is not None else np.zeros(80 * 80)
    prev_input = cur_input
    
    # forward the policy network and sample action according to the proba distribution
    proba = model.predict(np.expand_dims(x, axis=1).T)
    action = UP_ACTION if np.random.uniform() < proba else DOWN_ACTION
    y = 1 if action == 2 else 0 # 0 and 1 are our labels

    # log the input and label to train later
    xs.append(x)
    ys.append(y)

    # do one step in our environment
    observation, reward, done, info = env.step(action)
    rewards.append(reward)
    reward_sum += reward
    
    n_steps+=1
    
    if reward == -1:
        print("Round %d: it took %d time steps, L" % (round_n, n_steps))
    elif reward == +1:
        print("Round %d: it took %d time steps,  dub" % (round_n, n_steps))
    if reward != 0:
        round_n += 1
        n_steps = 0
            
    # end of an episode
    if done:
        print('At the end of episode', episode_number, 'the total reward was :', reward_sum)
        
        # increment episode number
        episode_number += 1
        
        # training
        model.fit(x=np.vstack(xs), y=np.vstack(ys), verbose=0, sample_weight=discount_rewards(rewards, gamma))
        """verbose=1, callbacks=[tbCallBack], """
        
        # Saving the weights used by our model
        if episode_number % batch_size == 0:    
            print ('Saving model...')
            model.save_weights('SmartPong_weights' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.h5')
        
        # Log the reward
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print ('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
        
        # Reinitialization
        xs, ys, rewards = [],[],[]
        observation = env.reset()
        reward_sum = 0
        prev_input = None
        round_n = 1