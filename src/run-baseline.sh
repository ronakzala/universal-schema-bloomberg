#!/bin/bash
#
#SBATCH --job-name=train_roll_call_106_0.1
#SBATCH --output=log_files/res_gpu_109.txt  # output file
#SBATCH -e log_files/res_gpu_109.err        # File to which STDERR will be written
#SBATCH --partition=1080ti-long # Partition to submit to 
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=01-03:00:00         # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=4000    # Memory in MB per cpu allocated

module add cuda90/toolkit/9.0.176
python baseline_model.py --datafile ../data/109_no_eval.hdf5 --classifier nn_embed_m_nocv --eta 0.1 --nepochs 10 --dp 10
