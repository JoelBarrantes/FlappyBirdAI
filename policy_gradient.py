import torch
import torch.nn as nn
import numpy as np
from environment import Envir
import torch.optim as optim

batch_size = 10 # every how many episodes to do a param update?



D = 100 * 100 # input dimensionality: 80x80 grid

criterion = nn.CrossEntropyLoss()



def main():

# model initialization

    # hyperparameters

    #CONV_NET PARAMETERS



      # Batch History

    env = Envir('FlappyBird-v0',False)

    optimizer = optim.RMSprop(env.policy.parameters(), lr=learning_rate, weight_decay=decay_rate)

    for e in range(0, max_episodes):

        env.step()


        discounted_epr = discount_rewards(np.vstack(env.rewards))
            # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)


        policy_loss = []
        for log_prob, reward in zip(env.log_probs, discounted_epr):
            policy_loss.append(-log_prob * reward)

        optimizer.zero_grad()

        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()

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
