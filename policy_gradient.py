
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical



import numpy as np
import gym
import gym_ple
from PIL import Image
from scipy import misc
import time
from environment import Envir


batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?
render = False
max_episodes = 1000
batch_size = 1

D = 100 * 100 # input dimensionality: 80x80 grid

criterion = nn.CrossEntropyLoss()



def main():

# model initialization

    # hyperparameters

    #CONV_NET PARAMETERS



      # Batch History

    env = Envir('FlappyBird-v0',False)

    for e in range(0, max_episodes):

        env.step()



        if e >= 0 and e % batch_size == 0:

            epdlogp = np.vstack(env.log_probs)
            discounted_epr = discount_rewards(np.vstack(env.rewards))
                # standardize the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)
            epdlogp *= discounted_epr
            for log_prob, reward in zip(env.log_probs, rewards):
                policy_loss.append(-log_prob * reward)

        env.reset()

def discount_rewards(r):

    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

main()
