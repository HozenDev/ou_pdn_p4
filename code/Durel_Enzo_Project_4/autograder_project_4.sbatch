#!/bin/bash
# Choose the partition that you will use.
# You can choose: oucspdn_gpu or gpu 
#
#SBATCH --partition=oucspdn_gpu
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=auto
#SBATCH --time=00:30:00
#
# --------------------- Change the items below here! ---------------------
#
#SBATCH --mail-user=Enzo.B.Durel-1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/oucspdn025/ou_pdn_p4/code/Durel_Enzo_Project_4
#SBATCH --output=/home/oucspdn025/ou_pdn_p4/code/Durel_Enzo_Project_4/results/out_autograding.txt
#SBATCH --error=/home/oucspdn025/ou_pdn_p4/code/Durel_Enzo_Project_4/results/err_autograding.txt

#################################################

source /home/oucspdnta/activate_pdn_env.sh
module load CUDA/10.1.243-GCC-8.3.0

# To run all the problems and get all results
# python3 -u autograder_project_4.py

# To run just problem 1's autograder
python3 -u autograder_problem_4_1.py

# To run just problem 2's autograder
# python3 -u autograder_problem_4_2.py

# To run just problem 3's autograder
# python3 -u autograder_problem_4_3.py

# To run just problem 4's autograder
# python3 -u autograder_problem_4_4.py
