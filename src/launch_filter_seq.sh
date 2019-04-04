#!/bin/bash

name=$1
jname=$2
partition=$3

#sbatch -p $partition --ntasks=1 --cpus-per-task=10 --mem=50G --output="$name/joblog" --job-name $jname filter-script.sh $name

sbatch -p $partition --ntasks=1 --cpus-per-task=10 --mem=20G filter_seq_script.sh $name --output="$name/joblog"  --job-name $jname

#sbatch -p $partition --output="$name/joblog" --ntasks=1 --cpus-per-task=10 --mem=20G filter-script.sh $name 
