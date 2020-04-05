import torch, os
import torch.nn as nn
import numpy as np
from abc import ABCMeta, abstractmethod
from time import time
from copy import deepcopy

class Evolution():
    """Base layer for evolution strategies"""
    __metaclass__ = ABCMeta

    def __init__(self, env, agent, interact, params):
        """Initialize an Evolution class given a dictionary of parameters.

        Params
        ======
        * **env** (gym) --- gym environment
        * **agent** (Neural Network) --- agent
        * **interact** (function) --- function to allow interaction between agent and environment to return cumulative rewards
        * **params** (dict-like) --- a dictionary of parameters
        """
        
        self.env = env
        self.agent = agent
        self.interact = interact
        self.params = params
        self.num_agents = params['num_agents']
        self.runs = params['runs']
        self.timesteps = params['timesteps']
        self.top_parents = params['top_parents']
        self.generations = params['generations']
        self.mutation_power = params['mutation_power']

    @abstractmethod
    def reproduce(sorted_parent_indexes, elite_index):
        """An abstract function where you take the parent agents and replace them with their children"""

        pass
        
    
    def generate_random_agents(self):
        """Creates X random agents"""

        agents = []
        for _ in range(self.num_agents):
            agent = deepcopy(self.agent)
            for param in agent.parameters():
                param.requires_grad = False
            agents.append(agent)
        return agents
        
    def calculate_fitness(self):
        """Calculates the average fitness score of each agent after several runs"""

        fitness_scores = []
        for agent in self.agents:
            fitness_scores_ = []
            for _ in range(self.runs):
                fitness_scores_.append(self.interact(agent, self.env, self.params))
            fitness_scores.append(sum(fitness_scores_) / self.runs)
        return fitness_scores

    def mutate(self, agent):
        """Mutates the weights of the agent"""

        agent = deepcopy(agent)
        for param in agent.parameters():
            if len(param.shape) == 4: # weights of Conv2D layer
                for i0 in range(param.shape[0]):
                    for i1 in range(param.shape[1]):
                        for i2 in range(param.shape[2]):
                            for i3 in range(param.shape[3]):
                                param[i0][i1][i2][i3] += self.mutation_power * np.random.randn()

            elif len(param.shape) == 2: # weights of Linear layer
                for i0 in range(param.shape[0]):
                    for i1 in range(param.shape[1]):
                        param[i0][i1] += self.mutation_power * np.random.randn()

            elif len(param.shape) == 1: # biases of linear layer or conv layer
                for i0 in range(param.shape[0]):
                    param[i0] += self.mutation_power * np.random.randn()
            
            else:
                raise Exception('Mutation failed for layer! Layer type does not exist')

        return agent

    def evolve(self):
        """Evolve the agents"""

        # disable gradients as we will not use them
        torch.set_grad_enabled(False)

        self.agents = self.generate_random_agents()

        for generation in range(self.generations):
            fitness_scores = self.calculate_fitness() 
            if self.top_parents < len(fitness_scores):
                top_agents_idxs = sorted(range(len(fitness_scores)), key = lambda sub: fitness_scores[sub])[-self.top_parents : ] 
            else: 
                top_agents_idxs = sorted(range(len(fitness_scores)), key = lambda sub: fitness_scores[sub])
            
            top_fitness_scores = [fitness_scores[x] for x in top_agents_idxs]
            print(f'Generation {generation} | Mean Fitness Scores : {np.mean(fitness_scores)} | Mean of top {self.top_parents} : {np.mean(top_fitness_scores)}')
            print(f'Top {self.top_parents} Scores {top_fitness_scores}')

            # kill all agents, and replace them with their children
            self.agents = self.reproduce(top_agents_idxs)

class Simple_Asexual_Evolution(Evolution):
    """Evolution strategy for Car Racing Environment"""

    def reproduce(self, top_agents_idxs):
        """Simple Asexual reproduction where you mutate the parents genes without mixing"""
        
        elite_index = top_agents_idxs[0]

        child_agents = []

        # first take selected parents from sorted_parent_indexes and generate N-1 children
        for i in range(self.num_agents - 1):
            agent_idx = top_agents_idxs[np.random.randint(len(top_agents_idxs))]
            child_agents.append(self.mutate(self.agents[agent_idx]))

        # add elite agent
        child_agents.append(self.agents[elite_index])

        return child_agents

class Elite_Evolution(Evolution):
    """Evolution strategy for Car Racing Environment"""

    def reproduce(self, top_agents_idxs):
        """Mutate the genes of the elite only"""
        
        elite_index = top_agents_idxs[0]

        child_agents = []

        # first take selected parents from sorted_parent_indexes and generate N-1 children
        for i in range(self.num_agents - 1):
            agent_idx = top_agents_idxs[np.random.randint(len(top_agents_idxs))]
            child_agents.append(self.mutate(self.agents[elite_index]))

        # add elite agent
        child_agents.append(self.agents[elite_index])

        return child_agents