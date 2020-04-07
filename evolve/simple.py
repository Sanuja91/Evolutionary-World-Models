from evolve.base import Evolution

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


class Natural_Evolution(Evolution):
    """Evolution strategy for Car Racing Environment"""

    def reproduce(self, top_agents_idxs):
        """Traditional Human Evolution"""

        elite_index = top_agents_idxs[0]
        
        exit()

        child_agents = []

        # first take selected parents from sorted_parent_indexes and generate N-1 children
        for i in range(self.num_agents - 1):
            agent_idx = top_agents_idxs[np.random.randint(len(top_agents_idxs))]
            child_agents.append(self.mutate(self.agents[elite_index]))

        # add elite agent
        child_agents.append(self.agents[elite_index])

        return child_agents