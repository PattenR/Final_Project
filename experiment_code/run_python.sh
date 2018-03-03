#!/bin/bash -login
BATCH --nodes=1

#SBATCH -t 0-02:00 # Runtime in D-HH:MM                                                                                                              
#SBATCH -p gpu # Partition to submit to                                                                                                              
#SBATCH --mem=15000
#SBATCH -J  testing_tensorflow    # name    

#SBATCH --gres=gpu:2
#SBATCH --job-name=gpujob
#SBATCH -o hostname_%j.out # File to which STDOUT will be written                                                                                    
#SBATCH -e hostname_%j.err # File to which STDERR will be written                                                                                    
#SBATCH --mail-type=ALL # Type of email notification- BEGIN,END,FAIL,ALL                                                                             
#SBATCH --mail-user=rp14964@bristol.ac.uk # Email to which notifications will be sent   

module add libs/tensorflow/1.2

srun python permutations.py
