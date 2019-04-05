#!/bin/bash

# sbatch -p longq --output="../output/test/joblog" --job-name test --ntasks=1 --cpus-per-task=10 --mem=100G filter-script.sh
source activate cs696
name=$1
python filter_by_names_seq.py --output-dir=$name