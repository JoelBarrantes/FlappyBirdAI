import argparse
import gym
import numpy as np
import time
import cv2
import gym_ple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions import Categorical
from itertools import count
from scipy import misc
from skimage import measure


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 1)')
args = parser.parse_args()


env = gym.make('FlappyBird-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)

learning_rate = 1e-4
#learning_rate = 0.0085
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False

# resume from previous checkpoint?
batch_size = 1
render = True
max_episodes = 1000
date_time = time.strftime("%Y.%m.%d-%H.%M.%S")
model_path = 'Model-'+date_time+'.pt'
model_load = "Model-2018.05.30-01.13.48.pt"


class Policy1(nn.Module):

    # cnn_layers is a list of lists that represent a layer. i.e. [[input_ch, out_ch, kernel_size, stride, padding], [.....]] is a list with two convolutional layers.
    def __init__(self):
        super(Policy1, self).__init__()
        self.conv1 = nn.Conv2d(4,10,3,stride=1,padding=0)
        self.conv2 = nn.Conv2d(10,5,3,stride=1,padding=0)
        #self.conv3 = nn.Conv2d(10,5,3,stride=1,padding=0)
  #      self.conv4 = nn.Conv2d(32,32,3,stride=1,padding=0)

        self.fc1= nn.Linear(845, 64)
       # self.fc2= nn.Linear(256, 64)
        self.fc2= nn.Linear(64, 2)


        self.saved_log_probs = []
        self.rewards = []
        self.states = []
        self.running_reward = 0

    def forward(self, x):

        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        #x = F.relu((self.conv3(x)))
     #   x = F.relu(F.max_pool2d(self.conv4(x),2))
        x = x.view(-1, 845)

        x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        # x = F.relu(self.fc2(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x


class Policy(torch.nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        self.conv = torch.nn.Sequential()
        self.conv.add_module("conv_1", torch.nn.Conv2d(4, 32, kernel_size=3))
        self.conv.add_module("maxpool_1", torch.nn.MaxPool2d(kernel_size=2))
        self.conv.add_module("relu_1", torch.nn.ReLU())
        self.conv.add_module("conv_2", torch.nn.Conv2d(32, 32, kernel_size=3))
        self.conv.add_module("dropout_2", torch.nn.Dropout())
        self.conv.add_module("maxpool_2", torch.nn.MaxPool2d(kernel_size=2))
        self.conv.add_module("relu_2", torch.nn.ReLU())

        self.fc = torch.nn.Sequential()
        self.fc.add_module("fc1", torch.nn.Linear(320*10, 50))
        self.fc.add_module("relu_3", torch.nn.ReLU())
        self.fc.add_module("dropout_3", torch.nn.Dropout())
        self.fc.add_module("fc2", torch.nn.Linear(50, 2))
        self.fc.add_module("softmax_1", torch.nn.Softmax())

        self.saved_log_probs = []
        self.rewards = []
        self.states = []
        self.running_reward = 0

    def forward(self, x):
        x = self.conv.forward(x)
        x = x.view(-1, 320*10)
        return self.fc.forward(x)



if resume:
    policy = torch.load(model_load)
else:
    policy = Policy1()

criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(policy.parameters(), lr=learning_rate)
eps = np.finfo(np.float32).eps.item()

def select_action(state):
    state = torch.from_numpy(state).float()
    state.requires_grad=True
    probs = policy(state)#The output of the model
    m = Categorical(probs)
    action = m.sample() #The "Fake label"
    policy.saved_log_probs.append(-m.log_prob(action))
    #policy.saved_log_probs.append(criterion(probs, action))#Cross entropy loss
    return action

def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

    rewards = policy.rewards

    optimizer.zero_grad()
    for loss, reward, state in zip(policy.saved_log_probs,rewards, policy.states):

    #    state = torch.from_numpy(state).float()
    #    state.requires_grad=True
    #    probs = policy(state)#The output of the model
    #    m = Categorical(probs)
    #    action = m.sample() #The "Fake label"
    #    policy_loss = criterion(probs, action) * reward
    #    policy_loss.backward()
        loss_t = loss * reward
        policy_loss.append(loss_t)


    policy_loss = torch.cat(policy_loss).sum()
    print(policy_loss)
    policy_loss.backward()
    #optimizer.step()
    optimizer.step()
    del policy.states[:]
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def _preprocess_observation(observation):

    observation = observation[0:404, :]
    red, green, blue = observation[:,:,0], observation[:,:,1], observation[:,:,2]
    mask =((red == 83) & (green == 56 ) & (blue == 70 )) | (red == 73 ) & (green == 36 ) & (blue == 85) | \
          (red == 84) & (green == 56 ) & (blue == 71) | (red == 80) & (green == 67) &  ( blue == 97)
    observation[:,:,:3][mask] = [0,0,0]

    grayscale_observation = observation.mean(2)

    ret,thresh = cv2.threshold(grayscale_observation,1,255,0)
    image = measure.label(thresh,connectivity=1)
    image[image == 1] = 255
    image[image < 255 ] = 0

    return misc.imresize((255 - image.astype(np.uint8)), (60, 60))

def main():
    threshold = 0
    running_reward = policy.running_reward
    for i_episode in count(1):
        state = env.reset()
        state = _preprocess_observation(state)
        state = state.reshape(1, 1, state.shape[0], state.shape[1])
        episode_length = 0
        states = []
        states.append(state)

        state_acc = state
        for t in range(10000):  # Don't infinite loop while learning

            if t<=3:
                action = np.random.randint(0,2)
            else:
                action = select_action(state)

            state, reward, done, _ = env.step(action)

            if reward < 0:
                reward = -5
            if reward > 1:
                reward = 1

            state = _preprocess_observation(state)
            state = state.reshape(1, 1, state.shape[0], state.shape[1])

            if t==3:
                s_t1 = np.append(state, states[len(states)-1], axis=1)

                s_t2 = np.append(s_t1, states[len(states)-2], axis=1)

                s_t3 = np.append(s_t2, states[len(states)-3], axis=1)
                state = s_t3

                policy.states.append(state)
                policy.rewards.append(reward)

            if t>3:
                s_t1 = np.append(state, states[len(states)-1][:,:3,:,:], axis=1)
                state = s_t1
                policy.rewards.append(reward)
                policy.states.append(state)

            states.append(state.copy())


            #misc.imsave('output/'+str(t)+"-"+str(i_episode)+'.png',state)
            #time.sleep(0.3)
            if render:
                env.render()

            if t>=3:
                episode_length +=1
                if reward<0:
                    old_threshold = threshold
                    threshold = episode_length - 20
                    if threshold<old_threshold:
                        threshold = old_threshold
                    policy.rewards[threshold:]=[-1]*len(policy.rewards[threshold:])
                    threshold = episode_length


                if reward>0:


                    #threshold = episode_length - 20
                    # if threshold<0:
                    #    threshold = 0
                    policy.rewards[threshold:]= [1]*len(policy.rewards[threshold:])
                    threshold = episode_length


            if done:
                break


        if i_episode>0 and  i_episode%batch_size==0:
            threshold = 0
            running_reward = running_reward + episode_length
            policy.running_reward = running_reward
            finish_episode()
            if i_episode % args.log_interval == 0:

                torch.save(policy, model_path)

                print('Episode {}\tEpisode length: {:5d}\tAverage length: {:.2f}'.format(
                    i_episode, episode_length, running_reward/i_episode+1))



if __name__ == '__main__':
    main()



