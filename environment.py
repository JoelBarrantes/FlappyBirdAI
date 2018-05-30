import gym
import time
import numpy as np
import cv2
import torch
import gym_ple

from torch.autograd import Variable
from torch.distributions import Categorical
from scipy import misc
from skimage import measure
from policy import Policy



ACTION_SPACE = {'Pong-v0': [0, 2, 3],  # NONE, UP and DOWN.
                'FlappyBird-v0': [0, 1], #DO NOTHING, PRESS THE BUTTON
                'Breakout-v0': [1, 2, 3]
                }




conv_layers =  [[1,5,5,1,0]
                ,[5,5,5,1,0]
                ,[5,10,3,1,0]
                 ]


fc_layers = [[1*10*10*10, 32]
             ,[32,2]
             #,[]
            ]



class Envir():

    def __init__(self, name, render):
        self.environment = gym.make(name)
        self.action_space = ACTION_SPACE[name]
        self.reset()
        self.policy = Policy(conv_layers, fc_layers, False).to('cpu')
        self.render = render

    def reset(self):
        """Resets the environment."""

        self.done = False
        self.episode_reward = 0
        self.episode_length = 0
        self.state = _preprocess_observation(self.environment.reset())
        self.episode_start_time = time.time()
        self.episode_run_time = 0

        self.observations = []
        self.rewards = []
        self.actions = []
        self.log_probs = []

    def step(self):


        if self.done:
            return 1

        state =  torch.from_numpy(self.state).float().unsqueeze(0)
        probs = self.policy(Variable(state.unsqueeze_(0)))
        m = Categorical(probs)
        action = m.sample()
        self.log_probs.append(m.log_prob(action))
        action = action.data.numpy().astype(int)[0]


        if action not in self.action_space:
            raise ValueError('Action "{}" is invalid. Valid actions: {}.'.format(action, self.action_space))

        observation, reward, self.done, info = self.environment.step(action)

        observation = _preprocess_observation(observation)
        if self.episode_length < 7 :
            self.state = observation
            self.observations.append(self.state)

        else:
            y = self.observations[self.episode_length- 7] - 127
            y[y<0] = 0
            self.state = (observation + y)
            self.state[self.state > 255] = 255
            self.observations.append(self.state)



        if self.render:
            self.render()
        self.actions.append(action)


        self.observations.append(self.state)
        self.rewards.append(reward)

        self.episode_reward += reward
        self.episode_length += 1


        self.episode_run_time = time.time() - self.episode_start_time


    def render(self):
        """Draws the environment."""

        self.environment.render()

    def sample_action(self):
        """Samples a random action."""
        x = np.random.randint(0, 11)
        if x == 10:
            return 0
        return 1

        return np.random.choice(self.action_space)

    def get_state(self):
        """Gets the current state.
        Returns:
            An observation (47x47x1 tensor with float32 values between 0 and 1).
        """

        return self.state



































def _preprocess_observation(observation):
    """Transforms the specified observation into a 47x47x1 grayscale image.
    Returns:
        A 47x47x1 tensor with float32 values between 0 and 1.
    """

    # Transform the observation into a grayscale image with values between 0 and 1. Use the simple
    # np.mean method instead of sophisticated luminance extraction techniques since they do not seem
    # to improve training.

    observation = observation[0:450, :]
    red, green, blue = observation[:,:,0], observation[:,:,1], observation[:,:,2]
    mask =((red == 83) & (green == 56 ) & (blue == 70 )) | (red == 73 ) & (green == 36 ) & (blue == 85) | \
          (red == 84) & (green == 56 ) & (blue == 71) | (red == 80) & (green == 67) &  ( blue == 97)
    observation[:,:,:3][mask] = [0,0,0]


   # observation[np.where((observation==[83,56,70]).all(axis=2) |(observation==[73,36,85]).all(axis=2)
   #                   | (observation==[84,56,71]).all(axis=2) | (observation==[80,67,97]).all(axis=2))] = [0,0,0]

    #84,56,71
    #80,67,97
 #   black_observation[np.where((black_observation!=[0,0,0]).all(axis=2))] = [255,255,255]

    grayscale_observation = observation.mean(2)

    ret,thresh = cv2.threshold(grayscale_observation,1,255,0)
    image = measure.label(thresh,connectivity=1)
    image[image == 1] = 255
    image[image < 255 ] = 0

    #thresh=thresh/255

    # thresh = np.pad(thresh, [(1,0),(1,1)],mode='constant', constant_values = 0)
    #thresh =  thresh.astype(np.uint8)
    #thresh = scipy.ndimage.binary_fill_holes(1-thresh)
    #im2, contours, hierarchy = cv2.findContours(255-thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(thresh, [contours], -1 , (0,255,0), thickness = -1)
    #return np.pad((1-thresh[1:,1:thresh.shape[1]-1].astype(np.uint8))*255,[(0,20),(0,0)], mode='constant', constant_values = 255)
    return misc.imresize((255 - image.astype(np.uint8)), (100, 100)).astype(int)

