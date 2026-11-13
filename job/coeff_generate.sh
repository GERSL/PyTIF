#!/bin/bash
#SBATCH -J coeff
#SBATCH --partition=priority			# specify partition
#SBATCH --account=zhz18039			# specify the priority account 
#SBATCH --constraint='epyc128' # Target the AMD Epyc node architecture
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --array 1-450
#SBATCH --mem-per-cpu=10G   
#SBATCH -o log/%x-out-%A_%4a.out
#SBATCH -e log/%x-err-%A_%4a.err


#NOTE: This job is to arrage cores in My Matlab function. nocona quanah
#The variable $SGE_TASK_ID is the ID for this task. 
#The variable $SGE_TASK_FIRST is the ID for the first task.
#The variable $SGE_TASK_LAST is the ID for the last task.
#This script is a companion script for submitmjobs for MATLAB apps
#It demonstrates how the passed "task" can be used to make each of
#The ntasks to perform a different task or use different data. If your
#app is a function m-file, "task" needs to be passed as input argument.
#If your app is a script m-file, "task" is automatically available
#because it shares the same workspace as myscript.
# IMPORTANT: DONOT indent any of the below statements  quanah nocona


. "/home/shq19004/miniconda3/etc/profile.d/conda.sh"  # startup conda
conda activate tifpy315

echo $SLURMD_NODENAME # display the node name
cd ../

python coeff_generate.py --ji=$SLURM_ARRAY_TASK_ID --jn=$SLURM_ARRAY_TASK_MAX

echo 'Finished!'
exit

#SBATCH --partition=priority			# specify partition
#SBATCH --account=zhz18039			# specify the priority account 
#SBATCH --constraint='epyc128' # Target the AMD Epyc node architecture


#SBATCH --partition=general
#SBATCH --constraint='epyc128' # Target the AMD Epyc node architecture


#SBATCH --partition=general-gpu
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=120G