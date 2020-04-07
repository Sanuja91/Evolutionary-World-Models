import torch, os, gym
import torch.nn as nn
import numpy as np
from abc import ABCMeta, abstractmethod
from time import time
from copy import deepcopy
import ray, gym

class Evolution():
    """Base layer for evolution strategies"""
    __metaclass__ = ABCMeta

    def __init__(self, agent, interact, params):
        """Initialize an Evolution class given a dictionary of parameters.

        Params
        ======
        * **env** (gym) --- gym environment
        * **agent** (Neural Network) --- agent
        * **interact** (function) --- function to allow interaction between agent and environment to return cumulative rewards
        * **params** (dict-like) --- a dictionary of parameters
        """

        self.agent = agent
        self.interact = interact
        self.params = params
        self.gym = params['gym']
        self.num_agents = params['num_agents']
        self.runs = params['runs']
        self.timesteps = params['timesteps']
        self.top_parents = params['top_parents']
        self.generations = params['generations']
        self.mutation_power = params['mutation_power']
        self.parallel = params['parallel']
        

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

        return [self.fitness_test(agent, self.gym, self.interact, self.runs, self.params) for agent in self.agents]
    
    def calculate_fitness_parallel(self, ray):
        """Calculates the average fitness score of each agent after several runs in parallel"""
        
        return ray.get([self.fitness_test.remote(self, agent, self.gym, self.interact, self.runs, self.params) for agent in self.agents])

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

        if os.name != 'nt' and self.parallel:
            import ray
            ray.init()
        else:
            env = gym.make(self.gym)

        for generation in range(self.generations):
            if os.name != 'nt' and self.parallel:
                fitness_scores = self.calculate_fitness_parallel(ray)
            else:
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
            

    @ray.remote
    def fitness_test(self, agent, gym_, interact, runs, params):
        """Runs a fitness test for an agent"""
        env = gym.make(gym_)
        fitness_scores_ = []
        for _ in range(runs):
            fitness_scores_.append(interact(agent, env, params))
        return sum(fitness_scores_) / runs