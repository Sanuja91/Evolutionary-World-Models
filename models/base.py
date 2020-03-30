import torch, os
import torch.nn as nn
from abc import ABCMeta, abstractmethod
from time import time
from glob import glob

class Neural_Network(nn.Module):
    '''Base layer for all neural network models'''
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, name, type, params):
        '''Initialize an Agent object given a dictionary of parameters.
        
        Params
        ======
        * **name** (string) --- name of the model
        * **type** (string) --- type of the model
        * **params** (dict-like) --- a dictionary of parameters
        '''
        pass


    def load_model(self, file_name):
        '''Loads the models parameters and weights'''
        
        path = f'outputs\\checkpoints\\{self.type}\\{self.name}'

        if file_name is None:
            list_of_files = glob(path) 
            latest_file = max(list_of_files, key = os.path.getctime)
            _, file_name = os.path.split(latest_file)
        
        path = f'outputs\\checkpoints\\{self.type}\\{self.name}\\{file_name}'
            
        checkpoint = torch.load(path, map_location = lambda storage, loc: storage)
        self.params = checkpoint['params']
        self.load_state_dict(checkpoint['weights'])


    def save_model(self):
        '''Loads the models parameters and weights'''
        
        checkpoint = {
            'params' : self.params,
            'weights' : self.state_dict()
        }
        
        path = f'outputs\\checkpoints\\{self.type}\\{self.name}'
        if not os.path.exists(path):
            os.makedirs(path)
        
        path += f'\\{str(time)}'
        torch.save(checkpoint, path)


        
        




