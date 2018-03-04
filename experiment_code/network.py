
"""Class that represents the network to be evolved."""
import random
import logging
from train import train_network
NET_SIZE = 512
class Network():
    """Represent a network and let us operate on it.
        Currently only works for an MLP.
        """
    
    def __init__(self, nn_param_choices=None):
        """Initialize our network.
            Args:
            nn_param_choices (dict): Parameters for the network, includes:
            nb_neurons (list): [64, 128, 256]
            nb_layers (list): [1, 2, 3, 4]
            activation (list): ['relu', 'elu']
            optimizer (list): ['rmsprop', 'adam']
            """
        self.accuracy = 0.
        self.mal_accuracy = 0
        self.nn_param_choices = nn_param_choices
        self.network = {}  # (dic): represents MLP network parameters
    
    def create_random(self):
        """Create a random network."""
        
#        for key in self.nn_param_choices:
#            self.network[key] = random.choice(self.nn_param_choices[key])
        #LAYER 1 -- BEWARE THIS IS HARD CODED FOR NETWORK WITH 2 LAYERS SIZE NET_SIZE
        self.network["0"] = [[float(random.getrandbits(1)) for i in range(NET_SIZE)] for j in range(784)]
        #LAYER 2
        self.network["1"] = [[float(random.getrandbits(1)) for i in range(NET_SIZE)] for j in range(NET_SIZE)]
        #LAYER 3
        self.network["2"] = [[float(random.getrandbits(1)) for i in range(10)] for j in range(NET_SIZE)]

#        self.network["0"] = [[float(1) for i in range(NET_SIZE)] for j in range(784)]
#        #LAYER 2
#        self.network["1"] = [[float(1) for i in range(NET_SIZE)] for j in range(NET_SIZE)]
#        #LAYER 3
##        self.network["2"] = [[float(0) for i in range(10)] for j in range(NET_SIZE)]
#        self.network["2"] = [[float(1) for i in range(10)] for j in range(NET_SIZE)]

    def create_set(self, network):
        """Set network properties.
        Args:
        network (dict): The network parameters
        """
        self.network = network
    
    def get_conns(self):
        conns = 0
        for i in range(len(self.network["0"])):
            for j in range(len(self.network["0"][i])):
                conns += self.network["0"][i][j]
    
        for i in range(len(self.network["1"])):
            for j in range(len(self.network["1"][i])):
                conns += self.network["1"][i][j]
        
        for i in range(len(self.network["2"])):
            for j in range(len(self.network["2"][i])):
                conns += self.network["2"][i][j]
                    
        return conns
            
    def train(self, dataset):
        """Train the network and record the accuracy.
        Args:
        dataset (str): Name of dataset to use.
        """
        print("training!")
#        if self.accuracy == 0.:
        self.accuracy, self.mal_accuracy = train_network(self.network, dataset)
        return self.accuracy, self.mal_accuracy

    def print_network(self):
        """Print out a network."""
        logging.info(self.network)
        logging.info("Network accuracy: %.2f%%" % (self.accuracy * 100))
