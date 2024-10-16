#!/bin/bash -l
#SBATCH --job-name=dummy_train
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="lex.janssens@live.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --partition=testing
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --array=1-5
#SBATCH --partition=cpu-medium

# Source the Python virtual environment
# source $HOME/ARL-SEAC/bin/activate
cd $HOME/ARL-SEAC/seac

conda activate NEW_SEAC

echo "[$SHELL] #### Starting Python train SEAC"
echo "[$SHELL] ## This is $SLURM_JOB_USER on $HOSTNAME and this job has the ID $SLURM_JOB_ID"
# get the current working directory

export CWD=$(pwd)
echo "[$SHELL] ## current working directory: "$CWD
echo "[$SHELL] ## current python version: "$(python --version)
echo "[$SHELL] ## python from path: "$(which python)
echo "[$SHELL] ## list of packages: "$(conda list)

# Run the file
echo "[$SHELL] ## Run script"
python train.py with env_name=Foraging-10x10-3p-3f-v2 time_limit=25 run_dir="TEST${SLURM_ARRAY_TASK_ID}"

echo "[$SHELL] ## Script finished"

echo "[$SHELL] #### Finished Python test. Have a nice day"
