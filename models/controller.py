import torch
import torch.nn as nn
from cma import CMAEvolutionStrategy
from torch.multiprocessing import Process, Queue

from models.base import Neural_Network

class Controller(Neural_Network):
    """A Controller using Fully Connected Layers"""

    def __init__(self, name, params, load_model):
        """Initialize an Neural Network model given a dictionary of parameters.

        Params
        =====
        * **name** (string) --- name of the model
        * **params** (dict-like) --- a dictionary of parameters
        """
        super(Controller, self).__init__(name, params, load_model)
        self.name = name
        self.type = 'Controller'
        if load_model != False:
            self.load_model(load_model)
        else:
            self.params = params
        self.z_size = self.params['z_size']
        self.n_hidden = self.params['n_hidden']
        self.action_size = self.params['action_size']
        self.device = self.get_device()
        
        self.fc = nn.Linear(self.z_size + self.n_hidden, self.action_size)

        if load_model != False:
            self.load_state_dict(self.weights)
        
        print(self, "\n\n")

    def forward(self, *inputs):
        cat_in = torch.cat(inputs, dim = 1)
        return self.fc(cat_in)

def evolve_controller(model, vae, mdn, evolve, epochs, log_interval):
    """Evolves the controller"""

    
    


