import torch
import torch.nn as nn
from torch.distributions.distribution import Distribution
from base import Neural_Network


class Conv(nn.Module):
    """Simlifed NN for ease of code"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, conv = nn.Conv1d, activation == nn.ReLU, batch_norm = False):
        """Initialize an Convolutional model given a dictionary of parameter.s

        Params
        =====
        * **in_channels** (int) --- input channels
        * **out_channels** (int) --- output channels
        * **kernel_size** (int) --- feature map size
        * **stride** (int) --- stride length
        * **padding** (int) --- padding
        * **conv** (function) --- convulational function that is to be used
        * **activation** (function) --- activation function that is to be used
        * **batch_norm** (boolean) --- whether batch normalization is required
        """

        super(Conv, self).__init__()
        self.conv = conv(
            in_channels, 
            out_channels,
            kernel_size = kernel_size, 
            stride = stride,
            padding = padding, 
            bias = False
        ) 
        
        self.bn = None if batch_norm == False else nn.BatchNorm1d(
            out_channels,
            eps = 0.001, # value found in tensorflow
            momentum = 0.1, # default pytorch value
            affine = True
        )

        self.activation = activation(inplace = True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        x = self.activation(x)
        return x



class ConvVAE(Neural_Network):
    """Convolutional Variational Auto Encoder"""

    def __init__(self, name, type_, params):
        """Initialize an Neural Network model given a dictionary of parameters.

        Params
        =====
        * **name** (string) --- name of the model
        * **type** (string) --- type of the model
        * **params** (dict-like) --- a dictionary of parameters
        """
        super(ConvVAE, self).__init__()
        self.name = name
        self.type = type_
        self.z_size = params['z_size']
        self.batch_size = params['batch_size']
        self.learning_rate = params['learning_rate']
        self.kl_tolerance = params['kl_tolerance']
        self.params = params
        
        # The Encoder
        self.encode = nn.Sequential(
            Conv(3, 32, 4, 2, 0, conv = nn.Conv2d, activation = nn.ReLU, batch_norm = False),
            Conv(32, 64, 4, 2, 0, conv = nn.Conv2d, activation = nn.ReLU, batch_norm = False)
            Conv(64, 128, 4, 2, 0, conv = nn.Conv2d, activation = nn.ReLU, batch_norm = False)
            Conv(128, 256, 4, 2, 0, conv = nn.Conv2d, activation = nn.ReLU, batch_norm = False)
        )
    
        self.enc_fc_mu = nn.Linear(2 * 2 * 256, self.z_size)
        self.enc_fc_logvar = nn.Linear(2 * 2 * 256, self.z_size)
        self.epsilon = Distribution((self.batch_size, self.z_size))

        # The Decoder
        self.decode = nn.Sequential(
            Conv(256, 128, 5, 2, 0, conv = nn.ConvTranspose2d, activation = nn.ReLU, batch_norm = False),
            Conv(128, 64, 5, 2, 0, conv = nn.ConvTranspose2d, activation = nn.ReLU, batch_norm = False)
            Conv(64, 32, 6, 2, 0, conv = nn.ConvTranspose2d, activation = nn.ReLU, batch_norm = False)
            Conv(32, 3, 6, 2, 0, conv = nn.ConvTranspose2d, activation = nn.Sigmoid, batch_norm = False)
        )

        self.dec_fc = nn.Linear(2000, 1024)

    def forward(self, x):
        x = self.encode(x)
        x = x.reshape((-1, 2 * 2 * 256))
        mu = self.enc_fc_mu(x)
        logvar = self.enc_fc_logvar(x)
        sigma = torch.exp(logvar / 2.0)
        z = mu + sigma * self.epsilon
        x = self.dec_fc(z)
        x = x.reshape((-1, 1, 1, 1024))
        x = self.decode(x)
        return x

    def encode(self, x):
        """Encodes observation"""
        with torch.no_grad():
            x = self.encode(x)
            x = x.reshape((-1, 2 * 2 * 256))
            mu = self.enc_fc_mu(x)
            logvar = self.enc_fc_logvar(x)
        sigma = torch.exp(logvar / 2.0)
        z = mu + sigma * self.epsilon
        return z

    def train(self):
        """Trains the Conv VAE"""
        

