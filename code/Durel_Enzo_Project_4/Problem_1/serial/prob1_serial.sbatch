#!/bin/bash
#SBATCH --partition=oucspdn_gpu
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --chdir=/home/oucspdn025/ou_pdn_p4/code/Durel_Enzo_Project_4
#SBATCH --output=/home/oucspdn025/ou_pdn_p4/code/Durel_Enzo_Project_4/Problem_1/serial/prob1_serial_%J_stdout.txt
#SBATCH --error=/home/oucspdn025/ou_pdn_p4/code/Durel_Enzo_Project_4/Problem_1/serial/prob1_serial_%J_stderr.txt
#SBATCH --time=00:30:00
#SBATCH --job-name=prob1_serial
#SBATCH --mail-user=Enzo.B.Durel-1@ou.edu
#SBATCH --mail-type=ALL

#################################################

module purge
module load CUDA/10.1.243-GCC-8.3.0

make run
