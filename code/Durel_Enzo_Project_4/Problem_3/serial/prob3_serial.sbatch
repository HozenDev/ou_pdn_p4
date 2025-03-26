#!/bin/bash
#SBATCH --partition=oucspdn_gpu
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --chdir=/home/oucspdn025/ou_pdn_p4/code/Durel_Enzo_Project_4/Problem_3/serial
#SBATCH --output=/home/oucspdn025/ou_pdn_p4/code/Durel_Enzo_Project_4/Problem_3/serial/prob3_serial_%J_stdout.txt
#SBATCH --error=/home/oucspdn025/ou_pdn_p4/code/Durel_Enzo_Project_4/Problem_3/serial/prob3_serial_%J_stderr.txt
#SBATCH --time=00:05:00
#SBATCH --job-name=prob3_serial
#SBATCH --mail-user=Enzo.B.Durel-1@ou.edu
#SBATCH --mail-type=ALL

#################################################

./convolution_serial 2048 2048 ../../../test_data/mat_input.csv result_prob3_serial.csv time_prob3_serial.csv

