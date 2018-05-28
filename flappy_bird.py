

import numpy as np
import gym
import gym_ple
from PIL import Image
from scipy import misc
import time

c = 0
i = 0

while (True):


    env.step(env.sample_action())


   # misc.imsave('output/i-'+str(c)+'.png',env.get_state())
    #c+=1

   # time.sleep(0.3)

    i+=1
