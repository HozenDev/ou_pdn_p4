#!/bin/bash
# Choose the partition that you will use.
# You can choose: oucspdn_gpu or debug_gpu or gpu 
#
#
#SBATCH --partition=oucspdn_gpu
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=P4-1_gpu
#SBATCH --time=00:05:00
#SBATCH --job-name=P4-1
#SBATCH --chdir=/home/oucspdn025/ou_pdn_p4/code/Durel_Enzo_Project_4/Problem_1
#SBATCH --output=/home/oucspdn025/ou_pdn_p4/code/Durel_Enzo_Project_4/Problem_1/_P4-1_%J_stdout.txt
#SBATCH --error=/home/oucspdn025/ou_pdn_p4/code/Durel_Enzo_Project_4/Problem_1/_P4-1_%J_stderr.txt

#################################################

module purge
module load CUDA/10.1.243-GCC-8.3.0

# Add commands here
make run
