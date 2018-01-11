import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class ClassificationCNN(nn.Module):
    
    """
    A PyTorch implementation of a three-layer convolutional network
    with the following architecture:
    conv - relu - 2x2 max pool - fc - dropout - relu - fc
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(1, 256, 256), num_filters=32, kernel_size=7,
                 stride=1, weight_scale=0.001, pool=2, stride_pool=2, hidden_dim=100,
                 num_classes=14, dropout=0.0):
        """
        Initialize a new network.
        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data.
        - num_filters: Number of filters to use in the convolutional layer.
        - kernel_size: Size of filters to use in the convolutional layer.
        - hidden_dim: Number of units to use in the fully-connected hidden layer-
        - num_classes: Number of scores to produce from the final affine layer.
        - stride: Only for convolutional layer
        - weight_scale: Scale for the convolution weights initialization-
        - pool: The size of the max pooling window.
        - dropout: Probability of an element to be zeroed.
        """
        super(ClassificationCNN, self).__init__()
        
        channels, height, width = input_dim

        ############################################################################
        # TODO: Initialize the necessary layers to resemble the ThreeLayerCNN      #
        # architecture  from the class docstring. In- and output features should   #
        # not be hard coded which demands some calculations especially for the     #
        # input of the first fully convolutional layer. The convolution should use #
        # "some" padding which can be derived from the kernel size and its weights #
        # should be scaled. Layers should have a bias if possible.                 #
        ############################################################################
        # define parameters
        # if stride = 1 try to preserve same input size after conv layer,
        # otherwise we don't care if input gets scaled down
        pad = 0
        if stride == 1:
            pad = int((kernel_size - 1) / 2)
        conv_out_width = int(1 + (width - kernel_size + 2 * pad) / stride)
        conv_out_height = int(1 + (height - kernel_size + 2 * pad) / stride)
        out_pool_width = int(1 + (conv_out_width - pool) / pool)
        out_pool_height = int(1 + (conv_out_height - pool) / pool)
        # input features of first fc layer
        lin_input = int(num_filters * out_pool_height * out_pool_width)
        # this way we can easily access them in forward/backward pass
        self.input_dim = input_dim
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.weight_scale = weight_scale
        self.pool = pool
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout

        self.conv = nn.Conv2d(in_channels=channels, out_channels=num_filters, kernel_size=kernel_size,
                              stride=stride, padding=pad, bias=True)
        self.conv.weight.data = weight_scale * self.conv.weight.data  # weight scale
        # init.xavier_normal(self.conv.weight, gain=np.sqrt(2))
        # init.constant(self.conv.bias, 0.001)
        self.fc1 = nn.Linear(in_features=156800, out_features=1024)
        # init.xavier_normal(self.fc1.weight, gain=np.sqrt(2))
        # init.constant(self.fc1.bias, 0.001)
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        # init.xavier_normal(self.fc2.weight, gain=np.sqrt(2))
        # init.constant(self.fc2.bias, 0.001)
        self.fc3 = nn.Linear(in_features=512, out_features=num_classes)
        
        #self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=4, padding=2)
        #self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=4, padding=2)
        #self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding=2)
        #self.conv4 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, padding=2)
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=2, bias=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=2, bias=True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=2, bias=True)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=2, bias=True)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=2, bias=True)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=2, bias=True)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.
        Inputs:
        - x: PyTorch input Variable
        """

        ############################################################################
        # TODO: Chain our previously initialized convolutional neural network      #
        # layers to resemble the architecture drafted in the class docstring.      #
        # Have a look at the Variable.view function to make the transition from    #
        # convolutional to fully connected layers.                                 #
        ############################################################################
        
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv4(x)), kernel_size=2)
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(F.relu(self.conv6(x)), kernel_size=2)
        
        (_, C, H, W) = x.data.size()
        x = x.view(-1, C * H * W)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        
        # print('before conv', x.data.size())
        ##x = self.conv(x)
        # print('after conv', x.data.size())
        ##x = F.relu(F.max_pool2d(x, kernel_size=self.pool))
        # print('after max pool', x.size())
        ##(_, C, H, W) = x.data.size()
        ##x = x.view(-1, C * H * W)
        # print('after view', x.data.size())
        ##x = F.relu(F.dropout(self.fc1(x), self.dropout))
        # print('after fc1', x.size())
        ##x = self.fc2(x)
        # print('after fc2', x.size())
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return x
    
    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".
        Inputs:
        - path: path string
        """
        print ('Saving model... %s' % path)
        torch.save(self, path)