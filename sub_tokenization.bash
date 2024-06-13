#!/bin/bash
## now loop through the above array

#SBATCH --time=4-00:10:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=20G   # memory per CPU core
#SBATCH -J "tokenization"   # job name
#SBATCH --mail-user=liangxu@caltech.edu   # email address


## /SBATCH -p general # partition (queue)
## /SBATCH -o slurm.%N.%j.out # STDOUT
## /SBATCH -e slurm.%N.%j.err # STDERR


python BPEgenome.py

