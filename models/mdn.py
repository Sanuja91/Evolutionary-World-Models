import torch, os, cv2
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.distributions.distribution import Distribution
import matplotlib.pyplot as plt
from base import Neural_Network

class MDNRNN(Neural_Network):
    """A Mixed Density Network using an LSTM Core"""

    def __init__(self, name, params, load_model):
        """Initialize an Neural Network model given a dictionary of parameters.

        Params
        =====
        * **name** (string) --- name of the model
        * **params** (dict-like) --- a dictionary of parameters
        """

        super(MDNRNN, self).__init__(name, params, load_model)
        self.name = name
        self.type = 'MDN RNN'
        if load_model != False:
            self.load_model(load_model)
        else:
            self.params = params
        self.z_size = self.params['z_size']
        self.n_hidden = self.params['n_hidden']
        self.n_gaussians = self.params['n_gaussians']
        self.n_layers = self.params['n_layers']
        self.batch_size = self.params['batch_size']
        self.device = self.get_device()
        
        self.lstm = nn.LSTM(self.z_size, self.n_hidden, self.n_layers, batch_first = True)
        self.fc1 = nn.Linear(self.n_hidden, self.n_gaussians * self.z_size)
        self.fc2 = nn.Linear(self.n_hidden, self.n_gaussians * self.z_size)
        self.fc3 = nn.Linear(self.n_hidden, self.n_gaussians * self.z_size)
        
        if load_model != False:
            self.load_state_dict(self.weights)

    def get_mixture_coef(self, y):
        rollout_length = y.size(1)
        pi, mu, sigma = self.fc1(y), self.fc2(y), self.fc3(y)
        
        pi = pi.view(-1, rollout_length, self.n_gaussians, self.z_size)
        mu = mu.view(-1, rollout_length, self.n_gaussians, self.z_size)
        sigma = sigma.view(-1, rollout_length, self.n_gaussians, self.z_size)
        
        pi = F.softmax(pi, 2)
        sigma = torch.exp(sigma)
        return pi, mu, sigma
        
        
    def forward(self, x, h):
        # Forward propagate LSTM
        y, (h, c) = self.lstm(x, h)
        pi, mu, sigma = self.get_mixture_coef(y)
        return (pi, mu, sigma), (h, c)
    
    def init_hidden(self):
        return (
            torch.zeros(self.n_layers, self.batch_size, self.n_hidden).to(self.device),
            torch.zeros(self.n_layers, self.batch_size, self.n_hidden).to(self.device)
        )


def loss_function(y, pi, mu, sigma):
    y = y.unsqueeze(2)
    m = torch.distributions.Normal(loc = mu, scale = sigma)
    loss = torch.exp(m.log_prob(y))
    loss = torch.sum(loss * pi, dim = 2)
    loss = -torch.log(loss)
    return loss.mean()

def train(model, epochs, log_interval):
    """Trains the MDN RNN"""

    for epoch in range(epochs):
        # Set initial hidden and cell states
        hidden = model.init_hidden()

        for i in range(0, z.size(1) - seqlen, seqlen):
            # Get mini-batch inputs and targets
            inputs = z[:, i:i+seqlen, :]
            targets = z[:, (i+1):(i+1)+seqlen, :]

            # Forward pass
            hidden = detach(hidden)
            (pi, mu, sigma), hidden = model(inputs, hidden)
            loss = loss_function(targets, pi, mu, sigma)

            # Backward and optimize
            model.zero_grad()
            loss.backward()
            # clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        if epoch % log_interval == 0:
            print ('Epoch [{}/{}], Loss: {:.4f}'
                   .format(epoch, epochs, loss.item()))

if __name__ == "__main__":
    params = {
        'z_size' : 128,
        'n_hidden' : 256,
        'batch_size' : 200,
        'learning_rate' : 0.0001,
        'n_gaussians' : 5,
        'n_layers' : 1,
        'batch_norm' : True
    }

    model = MDNRNN('Alpha', params, False)
    train(model, 100, 500)


