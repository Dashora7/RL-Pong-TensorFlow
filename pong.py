import numpy as np
import gym

from keras_model import Net


UP_ACTION = 2
DOWN_ACTION = 3
action_dict = {DOWN_ACTION: 0, UP_ACTION: 1}

#Initialization variables
path = 'C:\\Users\\nrdas\\Downloads\\SADE_AI\\TFRL\\checks'
batch_num = 10
render = False
decay = 0.99
H = 200
learning_rate = 0.005
D = (80, 80)

#If you are resuming
resume = False

#make network
network = Net(D, H, 'tanh', learning_rate, path)

if resume:
    network.load_checkpoint()    

def discount_rewards(rewards, discount_factor):
    discounted_rewards = np.zeros_like(rewards)
    for t in range(len(rewards)):
        discounted_reward_sum = 0
        discount = 1
        for k in range(t, len(rewards)):
            discounted_reward_sum += rewards[k] * discount
            discount *= discount_factor
            if rewards[k] != 0:
                # Don't count rewards from subsequent rounds
                break
        discounted_rewards[t] = discounted_reward_sum
    return discounted_rewards

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

env = gym.make('Pong-v0')

batch_tup = []
running_reward = None
episode_n = 1

while True:
    print("Episode %d" % episode_n)

    done = False
    ep_reward_sum = 0

    round_n = 1

    prev_observation = prepro(env.reset())
    action = env.action_space.sample()
    observation, _, _, _ = env.step(action)
    observation = prepro(observation)
    n_steps = 1

    while not done:
        
        if render:
            env.render()
        
        
        difference_frame = observation - prev_observation
        prev_observation = observation
        aprob = network.forward_pass(difference_frame)[0]
        
        if np.random.uniform() < aprob:
            action = UP_ACTION
        else:
            action = DOWN_ACTION

        observation, reward, done, _ = env.step(action)
        observation = prepro(observation)
        ep_reward_sum += reward
        n_steps += 1

        tup = (difference_frame, action_dict[action], reward)
        batch_tup.append(tup)

        if reward == -1:
            print("Round %d: it took %d time steps, L" % (round_n, n_steps))
        elif reward == +1:
            print("Round %d: it took %d time steps,  dub" % (round_n, n_steps))
        if reward != 0:
            round_n += 1
            n_steps = 0

    print("Episode %d finished after %d rounds" % (episode_n, round_n))
        
    running_reward = ep_reward_sum if running_reward is None else running_reward * decay + ep_reward_sum * (1-decay)
        
    print("Resetting env. Episode reward was %.3f; running mean is %.3f" \
        % (ep_reward_sum, running_reward))

    if episode_n % batch_num == 0:
        states, actions, rewards = zip(*batch_tup)
        rewards = discount_rewards(rewards, decay)
        rewards -= np.mean(rewards)
        rewards /= np.std(rewards)
        batch_tup = list(zip(states, actions, rewards))
        network.train(batch_tup)
        batch_tup = []

    if episode_n % 100 == 0:
        network.save_checkpoint()

    episode_n += 1

network.closeSess()