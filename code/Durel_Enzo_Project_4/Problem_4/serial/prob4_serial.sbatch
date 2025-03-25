#!/bin/bash
#SBATCH --partition=oucspdn_gpu
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --chdir=/home/oucspdn025/ou_pdn_p4/code/Durel_Enzo_Project_4
#SBATCH --output=/home/oucspdn025/ou_pdn_p4/code/Durel_Enzo_Project_4/Problem_4/serial/prob4_serial_%J_stdout.txt
#SBATCH --error=/home/oucspdn025/ou_pdn_p4/code/Durel_Enzo_Project_4/Problem_4/serial/prob4_serial_%J_stderr.txt
#SBATCH --time=00:30:00
#SBATCH --job-name=prob4_serial

#################################################

./convolution_maxpooling_serial 2048 2048 ../../test_data/mat_input.csv result_prob4_serial.csv time_prob4_serial.csv

