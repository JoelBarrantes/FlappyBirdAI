
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical


class Policy(nn.Module):

    # cnn_layers is a list of lists that represent a layer. i.e. [[input_ch, out_ch, kernel_size, stride, padding], [.....]] is a list with two convolutional layers.
    def __init__(self, cnn_layers, nn_layers, bias):
        super(Policy, self).__init__()

        self.conv_layers = [] #Conv layers


        for i in range(0, len(cnn_layers)):

            in_ch = cnn_layers[i][0]
            out_ch = cnn_layers[i][1]
            kernel_size = cnn_layers[i][2]
            stride = cnn_layers[i][3]
            padding = cnn_layers[i][4]
            self.conv_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False))

        self.fc_layers = [] #Fully connected layers
        self.n_in = nn_layers[0][0]
        for i in range(0, len(nn_layers)):
            in_size = nn_layers[i][0]
            out_size = nn_layers[i][1]
            self.fc_layers.append( nn.Linear(in_size, out_size))

    def forward(self, x):

        for layer in self.conv_layers:
            x = F.relu(F.max_pool2d(layer(x),2))

        x = x.view(-1, self.n_in)

        for i  in range(0,len(self.fc_layers)):
            if i == len(self.fc_layers) - 1:
                x = self.fc_layers[i](x)
                x = F.softmax(x, dim= 1)
            else:
                x = F.relu(self.fc_layers[i](x))
        return x
