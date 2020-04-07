import numpy as np
from evolve.base import Evolution
from random import randint, random
from copy import deepcopy

class Simple_Asexual_Evolution(Evolution):
    """Evolution strategy for Car Racing Environment"""

    def reproduce(self, top_agents_idxs, fitness_scores):
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

    def reproduce(self, top_agents_idxs, fitness_scores):
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


class Natural_Evolution(Evolution):
    """Evolution strategy for Car Racing Environment"""

    def reproduce(self, top_agents_idxs, fitness_scores):
        """Traditional Human Evolution"""
        
        elite_index = top_agents_idxs[0]
        adjusted_fitness_scores = [x + min(fitness_scores) for x in fitness_scores]  # adjust so negative numbers do not distort sum
        selection_ratios = [x / sum(adjusted_fitness_scores) for x in adjusted_fitness_scores] 

        children = []

        # first take selected parents from sorted_parent_indexes and generate N-1 children
        for i in range(self.num_agents - 1):
            print(len(top_agents_idxs), len(selection_ratios))
            (parent_a_idx, parent_b_idx) = np.random.choice(top_agents_idxs, 2, p = selection_ratios)
            parent_a = self.agents[parent_a_idx]
            parent_b = self.agents[parent_b_idx]

            gene_split_ratio = random()
            gene_split_ratio = [gene_split_ratio, 1 - gene_split_ratio]
            child = self.copulate(parent_a, parent_b, gene_split_ratio)
            children.append(child)

        # add elite agent
        children.append(self.agents[elite_index])

        return children

    def copulate(self, parent_a, parent_b, gene_split_ratio):
        """Mix the gene pools by predefined ratios"""
        child = deepcopy(parent_a)
        parent_a_params = list(parent_a.parameters())
        parent_b_params = list(parent_b.parameters())

        for param_idx, param in enumerate(child.parameters()):
            if len(param.shape) == 4: # weights of Conv2D layer
                for i0 in range(param.shape[0]):
                    for i1 in range(param.shape[1]):
                        for i2 in range(param.shape[2]):
                            for i3 in range(param.shape[3]):
                                param[i0][i1][i2][i3] = parent_a_params[param_idx][i0][i1][i2][i3] * gene_split_ratio[0] + parent_b_params[param_idx][i0][i1][i2][i3] * gene_split_ratio[1]
                                if random() < self.mutation_chance:
                                    param[i0][i1][i2][i3] += self.mutation_power * np.random.randn()

            elif len(param.shape) == 2: # weights of Linear layer
                for i0 in range(param.shape[0]):
                    for i1 in range(param.shape[1]):
                        param[i0][i1] == parent_a_params[param_idx][i0][i1] * gene_split_ratio[0] + parent_b_params[param_idx][i0][i1] * gene_split_ratio[1]
                        if random() < self.mutation_chance:
                            param[i0][i1] += self.mutation_power * np.random.randn()

            elif len(param.shape) == 1: # biases of linear layer or conv layer
                for i0 in range(param.shape[0]):
                    param[i0] == parent_a_params[param_idx][i0] * gene_split_ratio[0] + parent_b_params[param_idx][i0] * gene_split_ratio[1]
                    if random() < self.mutation_chance:
                        param[i0] += self.mutation_power * np.random.randn()
        return child