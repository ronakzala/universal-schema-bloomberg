#!/bin/bash
#
#SBATCH --job-name=filter_politicians
#SBATCH --output=filter-log.txt  # STDOUT outfile
#SBATCH -e filter-log.err
#SBATCH --partition=longq    # Partition to submit to 
#
#SBATCH --ntasks=1
#SBATCH --time=02-01:00:00
#SBATCH --mem-per-cpu=4000

python filter_by_names.py
