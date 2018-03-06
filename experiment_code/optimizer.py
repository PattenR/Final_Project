"""
Class that holds a genetic algorithm for evolving a network.
Credit:
    A lot of those code was originally inspired by:
    http://lethain.com/genetic-algorithms-cool-name-damn-simple/
"""
from functools import reduce
from operator import add
import random
from network import Network

class Optimizer():
    """Class that implements genetic algorithm for MLP optimization."""

    def __init__(self, nn_param_choices, retain=0.4,
                 random_select=0.1, mutate_chance=0.2):
        """Create an optimizer.
        Args:
            nn_param_choices (dict): Possible network paremters
            retain (float): Percentage of population to retain after
                each generation
            random_select (float): Probability of a rejected network
                remaining in the population
            mutate_chance (float): Probability a network will be
                randomly mutated
        """
        self.mutate_chance = mutate_chance
        self.random_select = random_select
        self.retain = retain
        self.nn_param_choices = nn_param_choices

    def create_population(self, count):
        """Create a population of random networks.
        Args:
            count (int): Number of networks to generate, aka the
                size of the population
        Returns:
            (list): Population of network objects
        """
        pop = []
        for _ in range(0, count):
            # Create a random network.
            network = Network(self.nn_param_choices)
            network.create_random()

            # Add the network to our population.
            pop.append(network)

        return pop

    @staticmethod
    def fitness(network):
        """Return the accuracy, which is our fitness function."""
        return 1/network.mal_accuracy

    def grade(self, pop):
        """Find average fitness for a population.
        Args:
            pop (list): The population of networks
        Returns:
            (float): The average accuracy of the population
        """
        summed = reduce(add, (self.fitness(network) for network in pop))
        return summed / float((len(pop)))

    def breed(self, mother, father):
        """Make two children as parts of their parents.
        Args:
            mother (dict): Network parameters
            father (dict): Network parameters
        Returns:
            (list): Two network objects
        """
        #we need to make sure we don't take the param choice, we need whether the connection was actually there or not
        children = []
        for _ in range(2):

            child = {}

            # Loop through the parameters and pick params for the kid.
#            for param in self.nn_param_choices:
#                child[param] = random.choice(
#                    [mother.network[param], father.network[param]]
#                )
#            print("mother.network[0]")
#            print(mother.network["0"])
#            print("mother.network[1]")
#            print(mother.network["1"])
#            print("mother.network[2]")
#            print(mother.network["2"])
            child = mother.network # to give it the right shape to be filled in
#            for i in range(len(mother.network["0"])):
#                for j in range(len(mother.network["0"][i])):
#                    child["0"][i][j] = random.choice([mother.network["0"][i][j], father.network["0"][i][j]])
#
#            for i in range(len(mother.network["1"])):
#                for j in range(len(mother.network["1"][i])):
#                    child["1"][i][j] = random.choice([mother.network["1"][i][j], father.network["1"][i][j]])
#
#            for i in range(len(mother.network["2"])):
#                for j in range(len(mother.network["2"][i])):
#                    child["2"][i][j] = random.choice([mother.network["2"][i][j], father.network["2"][i][j]])

#            for i in range(len(mother.network["0"])):
#                child["0"][i] = random.choice([mother.network["0"][i], father.network["0"][i]])
#
#            for i in range(len(mother.network["1"])):
#                child["1"][i] = random.choice([mother.network["1"][i], father.network["1"][i]])
#
#            for i in range(len(mother.network["2"])):
#                child["2"][i] = random.choice([mother.network["2"][i], father.network["2"][i]])

            # take more stucture!
            child["0"] = random.choice([mother.network["0"], father.network["0"]])
            child["1"] = random.choice([mother.network["1"], father.network["1"]])
            child["2"] = random.choice([mother.network["2"], father.network["2"]])
            

            # Now create a network object.
            network = Network(self.nn_param_choices)
            network.create_set(child)

            # Randomly mutate some of the children.
            if self.mutate_chance > random.random():
                network = self.mutate(network)

            children.append(network)

        return children

    def mutate(self, network):
        """Randomly mutate one part of the network.
        Args:
            network (dict): The network parameters to mutate
        Returns:
            (Network): A randomly mutated network object
        """
        #slightly more complex mutation procedure required!
        mutation_rate = 0.02
        p = int(100*(1-mutation_rate))
        q = int(100*mutation_rate)
#
#
        for i in range(len(network.network["0"])):
            for j in range(len(network.network["0"][i])):
                list = [network.network["0"][i][j]]*p +[1 - network.network["0"][i][j]]*q
                network.network["0"][i][j] = random.choice(list)

        for i in range(len(network.network["1"])):
            for j in range(len(network.network["1"][i])):
                list = [network.network["1"][i][j]]*p +[1 - network.network["1"][i][j]]*q
                network.network["1"][i][j] = random.choice(list)

        for i in range(len(network.network["2"])):
            for j in range(len(network.network["2"][i])):
                list = [network.network["2"][i][j]]*p +[1 - network.network["2"][i][j]]*q
                network.network["2"][i][j] = random.choice(list)

        # Choose a random key.
#        print(self.nn_param_choices)
#        mutation = random.choice(list(self.nn_param_choices.keys()))
#
#        # Mutate one of the params.
#        network.network[mutation] = random.choice(self.nn_param_choices[mutation])

        return network

    def evolve(self, pop):
        """Evolve a population of networks.
        Args:
            pop (list): A list of network parameters
        Returns:
            (list): The evolved population of networks
        """
        # Get scores for each network.
        graded = [(self.fitness(network), network) for network in pop]

        # Sort on the scores.
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]

        # Get the number we want to keep for the next gen.
        retain_length = int(len(graded)*self.retain)

        # The parents are every network we want to keep.
#        parents = graded[:retain_length]
        parent = graded[:1]
        parent = parent[0]

        # For those we aren't keeping, randomly keep some anyway.
#        for individual in graded[retain_length:]:
#            if self.random_select > random.random():
#                parents.append(individual)

        # Now find out how many spots we have left to fill.
#        parents_length = len(parents)
#        desired_length = len(pop) - parents_length
        desired_length = len(pop) - 1
        children = []
        parents = [parent]

        # Add children, which are bred from two remaining networks.
        while len(children) < desired_length:
            child = {}
            child = parent.network
            network = Network(self.nn_param_choices)
            network.create_set(child)
            network = self.mutate(network)
            children.append(network)
            # Get a random mom and dad.
#            male = random.randint(0, parents_length-1)
#            female = random.randint(0, parents_length-1)

            # Assuming they aren't the same network...
#            if male != female:
#                male = parents[male]
#                female = parents[female]
#
#                # Breed them.
#                babies = self.breed(male, female)
#
#                # Add the children one at a time.
#                for baby in babies:
#                    # Don't grow larger than desired length.
#                    if len(children) < desired_length:
#                        children.append(baby)

        parents.extend(children)

        return parents
