import numpy as np
from optimizer import Optimizer
import logging

def gen_population(generations, population, nn_param_choices):
    optimizer = Optimizer(nn_param_choices) #choices are unimportant
    networks = optimizer.create_population(population)
    # Evolve the generation.
    for i in range(generations):
        logging.info("***Doing generation %d of %d***" %
                     (i + 1, generations))
            
         # Train and get accuracy for networks.
        accuracys = []
	F = open("results_sorted.txt", "w")
        accuracys_mal = []
        for net in networks:
#            print("training in perm")
#            print(net.network)
            acc, acc_mal = net.train("mnist")
            print("Net accuracy:%.2f\n" % acc)
#            print(acc)
            print("Net accuracy mal:%.2f\n" % acc_mal)
#            print(acc_mal)
            accuracys.append(acc)
            accuracys_mal.append(acc_mal)
         
        # Get the average accuracy for this generation.
        average_accuracy = np.mean(accuracys)
        average_accuracy_mal = np.mean(accuracys_mal)

        # Print out the average accuracy each generation.
        F.write("Generation average: %.2f\n" % (average_accuracy * 100))
        F.write("Generation average mal: %.2f\n" % (average_accuracy_mal * 100))
#         logging.info('-'*80)

         # Evolve, except on the last iteration.
        if i != generations - 1:
             # Do the evolution.
            networks = optimizer.evolve(networks)

    # Sort our final population.
    networks = sorted(networks, key=lambda x: 1/x.mal_accuracy, reverse=True)
    for net in networks[:5]:
        F.write("acc")
        F.write('{}'.format(net.accuracy))
        F.write("\n")
        F.write("acc_mal")
        F.write('{}'.format(net.mal_accuracy))
        F.write("\n")
        F.write("setup")
        F.write('{}'.format(net.network))
            F.write("\n")
    # Print out the top 5 networks.
#    print_networks(networks[:5])

def main():
    generations = 10  # Number of times to evole the population.
    population = 10  # Number of networks in each generation.
#    choice_arr = [5, 10, 15, 20, 25, 30, 35]
#    nn_param_choices = {
#        'nb_neurons_1': choice_arr,
#        'nb_neurons_2': choice_arr,
#        'nb_neurons_3': choice_arr,
#    }
#    choice_arr = all the choices
#    nn_param_choices = {
#        'nb_neurons_1': choice_arr,
#        'nb_neurons_2': choice_arr,
#        'nb_neurons_3': choice_arr,
#    }
    nn_param_choices = {} #not used for current experiemnt, refactoring needed
    gen_population(generations, population, nn_param_choices)

main()
