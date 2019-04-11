#!/bin/bash
#
#SBATCH --job-name=preprocess
#SBATCH --output=loading.txt  # STDOUT outfile
#SBATCH -e loading.err
#SBATCH --partition=longq    # Partition to submit to 
#
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=2000

python preprocess_articles.py --path /mnt/nfs/work1/mccallum/smysore/data/concretely_annotated_nyt/data/comms --loadwords True --loaddict True
