#!/bin/bash
#
#SBATCH --job-name=preprocess
#SBATCH --output=log_files/glove.txt  # STDOUT outfile
#SBATCH -e log_files/glove.err
#
#SBATCH --ntasks=1
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=2000

python preprocess_articles.py --path /mnt/nfs/work1/mccallum/smysore/data/concretely_annotated_nyt/data/comms --loadwords True --loaddict True
