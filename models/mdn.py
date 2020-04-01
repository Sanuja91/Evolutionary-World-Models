import torch, os, cv2
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.distributions.distribution import Distribution
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
from models.base import Neural_Network

class MDN_RNN(Neural_Network):
    """A Mixed Density Network using an LSTM Core"""

    def __init__(self, name, params, load_model):
        """Initialize an Neural Network model given a dictionary of parameters.

        Params
        =====
        * **name** (string) --- name of the model
        * **params** (dict-like) --- a dictionary of parameters
        """

        super(MDN_RNN, self).__init__(name, params, load_model)
        self.name = name
        self.type = 'MDN RNN'
        if load_model != False:
            self.load_model(load_model)
        else:
            self.params = params
        self.z_size = self.params['z_size']
        self.action_size = self.params['action_size']
        self.n_hidden = self.params['n_hidden']
        self.n_gaussians = self.params['n_gaussians']
        self.n_layers = self.params['n_layers']
        self.seq_len = self.params['seq_len']
        self.learning_rate = self.params['learning_rate']
        self.grad_clip = self.params['grad_clip']
        self.batch_size = self.params['batch_size']
        self.device = self.get_device()
        
        self.lstm = nn.LSTM(self.z_size + self.action_size, self.n_hidden, self.n_layers, batch_first = True)
        self.fc1 = nn.Linear(self.n_hidden, self.n_gaussians * (self.z_size + self.action_size))
        self.fc2 = nn.Linear(self.n_hidden, self.n_gaussians * (self.z_size + self.action_size))
        self.fc3 = nn.Linear(self.n_hidden, self.n_gaussians * (self.z_size + self.action_size))
        
        if load_model != False:
            self.load_state_dict(self.weights)
        
        print(self)

    def get_mixture_coef(self, y):
        rollout_length = y.size(1)
        pi, mu, sigma = self.fc1(y), self.fc2(y), self.fc3(y)
        
        pi = pi.view(-1, rollout_length, self.n_gaussians, self.z_size + self.action_size)
        mu = mu.view(-1, rollout_length, self.n_gaussians, self.z_size + self.action_size)
        sigma = sigma.view(-1, rollout_length, self.n_gaussians, self.z_size + self.action_size)
        
        pi = F.softmax(pi, 2)
        sigma = torch.exp(sigma)
        return pi, mu, sigma
        
        
    def forward(self, x, h):
        y, (h, c) = self.lstm(x, h)
        pi, mu, sigma = self.get_mixture_coef(y)
        return (pi, mu, sigma), (h, c)
    
    def init_hidden(self, batch_size):
        return (
            torch.zeros(self.n_layers, batch_size, self.n_hidden).to(self.device),
            torch.zeros(self.n_layers, batch_size, self.n_hidden).to(self.device)
        )


def loss_function(y, pi, mu, sigma):
    y = y.unsqueeze(2)
    m = torch.distributions.Normal(loc = mu, scale = sigma)
    loss = torch.exp(m.log_prob(y))
    loss = torch.sum(loss * pi, dim = 2)
    loss = -torch.log(loss)
    return loss.mean()

def train_mdn(model, epochs, log_interval):
    """Trains the MDN RNN"""

    model.train()
    model = model.to(model.device)
    optimizer = optim.Adam(model.parameters(), lr = model.learning_rate)
    z = torch.load('data\\inputs\\tensors\\zs.pt').float()
    actions = torch.load('data\\inputs\\tensors\\actions.pt').float()

    z = torch.cat((z, actions), dim = 1).unsqueeze(0)
    count = 0
    # for epoch in range(epochs):
    #     # Set initial hidden and cell states
    #     hidden = model.init_hidden(1)

    #     for i in range(0, z.size(1) - model.seq_len, model.seq_len):
    #         # Get mini-batch inputs and targets
    #         inputs = z[ : , i : i + model.seq_len, : ]
    #         targets = z[ : , (i + 1) : (i + 1) + model.seq_len, : ]

    #         print(inputs.shape)

    #         # Forward pass
    #         hidden = (hidden[0].detach(), hidden[1].detach())
    #         (pi, mu, sigma), hidden = model(inputs, hidden)
    #         loss = loss_function(targets, pi, mu, sigma)

    #         # Backward and optimize
    #         model.zero_grad()
    #         loss.backward()
    #         clip_grad_norm_(model.parameters(), model.grad_clip)
    #         optimizer.step()
    #         count += 1
    #         if count % log_interval == 0:
    #             print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch, epochs, loss.item()))
    
    # Creates batches
    inputs = []
    targets = []
    for i in range(0, z.size(1) - model.seq_len, model.seq_len):
        # Get mini-batch inputs and targets
        inputs.append(z[ : , i : i + model.seq_len, : ])
        targets.append(z[ : , (i + 1) : (i + 1) + model.seq_len, : ])

    if model.batch_size > len(inputs): 
        model.batch_size = len(inputs)
    else:
        raise Exception('Code for handling larger batch sizes not written.. Easy fix')
    
    inputs = torch.stack(inputs).squeeze(1)
    targets = torch.stack(targets).squeeze(1)
    print(inputs.shape)

    for epoch in range(epochs):
        # Set initial hidden and cell states
        hidden = model.init_hidden(model.batch_size)

        # Forward pass
        hidden = (hidden[0].detach(), hidden[1].detach())
        (pi, mu, sigma), hidden = model(inputs, hidden)
        loss = loss_function(targets, pi, mu, sigma)

        # Backward and optimize
        model.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), model.grad_clip)
        optimizer.step()
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch, epochs, loss.item()))
    model.save_model()