import torch, os
import torch.nn as nn
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

        # super(Evolution, self).__init__(env, agent, interact, params)
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

        for param in agent.parameters():
            if len(param.shape) == 4: # weights of Conv2D
                for i0 in range(param.shape[0]):
                    for i1 in range(param.shape[1]):
                        for i2 in range(param.shape[2]):
                            for i3 in range(param.shape[3]):
                                param[i0][i1][i2][i3] += self.mutation_power * np.random.randn()

            elif len(param.shape) == 2: # weights of linear layer
                for i0 in range(param.shape[0]):
                    for i1 in range(param.shape[1]):
                        param[i0][i1] += self.mutation_power * np.random.randn()

            elif len(param.shape) == 1: # biases of linear layer or conv layer
                for i0 in range(param.shape[0]):
                    param[i0] += self.mutation_power * np.random.randn()

        return agent

    # def add_elite(agents, sorted_parent_indexes, elite_index = None, only_consider_top_n = 10):

    #     candidate_elite_index = sorted_parent_indexes[:only_consider_top_n]

    #     if(elite_index is not None):
    #         candidate_elite_index = np.append(candidate_elite_index,[elite_index])

    #     top_score = None
    #     top_elite_index = None

    #     for i in candidate_elite_index:
    #         score = return_average_score(agents[i],runs=5)
    #         print("Score for elite i ", i, " is ", score)

    #         if(top_score is None):
    #             top_score = score
    #             top_elite_index = i
    #         elif(score > top_score):
    #             top_score = score
    #             top_elite_index = i

    #     print("Elite selected with index ",top_elite_index, " and score", top_score)

    #     child_agent = copy.deepcopy(agents[top_elite_index])
    #     return child_agent

    def evolve(self):
        """Evolve the agents"""

        # disable gradients as we will not use them
        torch.set_grad_enabled(False)

        self.agents = self.generate_random_agents()

        for generation in range(self.generations):

            fitness_scores = self.calculate_fitness() 
            top_fitness_scores = fitness_scores.sort(reverse = True)[ : self.top_limit]
            print(f'Generation {generation} | Mean Fitness Scores : {np.mean(fitness_scores)} | Mean of top {self.top_limit} : {np.mean(top_fitness_scores)}')
            print(f'Top {self.top_limit} Scores {top_fitness_scores}')

            # kill all agents, and replace them with their children
            self.agents = self.reproduce(fitness_scores)


    # def evolve():
    #     actions = 2 #2 actions possible: left or right

    #     # disable gradients as we will not use them
    #     torch.set_grad_enabled(False)

    #     # initialize N number of agents
    #     self.agents = self.generate_random_agents()

    #     # How many top agents to consider as parents
    #     top_limit = 20

    #     # run evolution until X generations
    #     generations = 1000

    #     elite_index = None

    #     for generation in range(generations):

    #         # return fitness_scores of agents
    #         rewards = deploy_agents_n_times(agents, 3) #return average of 3 runs

    #         # sort by rewards
    #         sorted_parent_indexes = np.argsort(rewards)[::-1][:top_limit] #reverses and gives top values (argsort sorts by ascending by default) https://stackoverflow.com/questions/16486252/is-it-possible-to-use-argsort-in-descending-order
    #         print("")
    #         print("")

    #         top_rewards = []
    #         for best_parent in sorted_parent_indexes:
    #             top_rewards.append(rewards[best_parent])

    #         print("Generation ", generation, " | Mean rewards: ", np.mean(rewards), " | Mean of top 5: ",np.mean(top_rewards[:5]))
    #         #print(rewards)
    #         print("Top ",top_limit," scores", sorted_parent_indexes)
    #         print("Rewards for top: ",top_rewards)

    #         # setup an empty list for containing children agents
    #         children_agents, elite_index = return_children(agents, sorted_parent_indexes, elite_index)

    #         # kill all agents, and replace them with their children
    #         agents = children_agents


class Simple_Asexual_Evolution(Evolution):
    """Evolution strategy for Car Racing Environment"""

    def reproduce(self, fitness_scores):
        """Simple Asexual reproduction where you mutate the parents genes without mixing"""
        
        # identify fittest agents for becoming parents
        sorted_parent_indexes = np.argsort(fitness_scores)[ : : -1 ][ : self.top_limit] # reverses and gives top values (argsort sorts by ascending by default) https://stackoverflow.com/questions/16486252/is-it-possible-to-use-argsort-in-descending-order
        top_fitness_scores = []
        for parent in sorted_parent_indexes:
            top_fitness_scores.append(fitness_scores[parent])
        elite_index = fitness_scores.index(max(fitness_scores))

        children_agents = []

        # first take selected parents from sorted_parent_indexes and generate N-1 children
        for i in range(len(agents) - 1):
            selected_agent_index = sorted_parent_indexes[np.random.randint(len(sorted_parent_indexes))]
            children_agents.append(self.mutate(self.agents[selected_agent_index]))

        #now add one elite
        elite_child = add_elite(agents, sorted_parent_indexes, elite_index)
        children_agents.append(elite_child)
        elite_index = len(children_agents) - 1 #it is the last one

        return children_agents, elite_index