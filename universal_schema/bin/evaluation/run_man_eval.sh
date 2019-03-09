#!/usr/bin/env bash
# Parse command line args.
while getopts ":a:e:d:r:" opt; do
    case "$opt" in
        a) action=$OPTARG ;;
        e) experiment=$OPTARG ;;
        d) dataset=$OPTARG ;;
        r) run_name=$OPTARG ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            exit 1
            ;;
    esac
done
# Make sure required arguments are passed.
if [[ "$action" == '' ]] || [[ "$experiment" == '' ]] || [[ "$dataset" == '' ]]; then
    echo "Must specify action (-a): nearest_ents/nearest_ops/nearest_cols/nearest_rows/nearest_roles/print_scores/ent2type/print_rankedents"
    echo "Must specify dataset (-d): ms500k/anyt"
    echo "Must specify experiment (-e): naryus/typevs/typentvs/ltentvs/mementvs/shmementvs/entvs/rnentvs"
    exit 1
fi
if [[ "$run_name" == '' ]] && [[ "$action" != 'ent2type' ]]; then
    echo "Must specify run name from model_runs directory (-r)."
    exit 1
fi

op2count_path="$CUR_PROJ_DIR/datasets_proc"
int_mapped_path="$CUR_PROJ_DIR/datasets_proc/${dataset}/${experiment}"
run_dir="$CUR_PROJ_DIR/model_runs/${dataset}/${experiment}/${run_name}"

log_dir="$CUR_PROJ_DIR/logs/evaluation"
mkdir -p "$log_dir"
script_name="man_eval"
source_path="$CUR_PROJ_DIR/experiments/src/evaluation"


if [[ $action == 'nearest_cols' ]] || [[ $action == 'nearest_rows' ]]; then
    log_file="${run_dir}/${script_name}-${action}-${experiment}_logs.txt"
    cmd="python2 -u $source_path/$script_name.py $action --int_mapped_path $int_mapped_path --run_path $run_dir"
    echo $cmd | tee ${log_file}
    eval $cmd 2>&1 | tee -a ${log_file}
elif [[ $action == 'nearest_ents' ]] || [[ $action == 'nearest_ops' ]]; then
    log_file="${run_dir}/${script_name}-${action}-${experiment}_logs.txt"
    cmd="python2 -u $source_path/$script_name.py $action
        --int_mapped_path $int_mapped_path \
        --op2count_path $op2count_path \
        --run_path $run_dir"
    echo $cmd | tee ${log_file}
    eval $cmd 2>&1 | tee -a ${log_file}
elif [[ $action == 'nearest_roles' ]]; then
    log_file="${run_dir}/${script_name}-${action}-${experiment}_logs.txt"
    cmd="python2 -u $source_path/$script_name.py $action --int_mapped_path $int_mapped_path \
        --run_path $run_dir \
        --experiment $experiment \
        --dataset $dataset"
    echo $cmd | tee ${log_file}
    eval $cmd 2>&1 | tee -a ${log_file}
elif [[ $action == 'print_scores' ]]; then
    log_file="${run_dir}/${script_name}-${action}-${experiment}_logs.txt"
    # Print high scoring pairs.
    cmd="python2 -u $source_path/$script_name.py $action --run_path $run_dir --thresh_str high --thresh_score 0.7"
    echo $cmd | tee ${log_file}
    eval $cmd 2>&1 | tee -a ${log_file}
    echo '' | tee -a ${log_file}
    # Print low scoring pairs.
    cmd="python2 -u $source_path/$script_name.py $action --run_path $run_dir --thresh_str low --thresh_score 0.3"
    echo $cmd | tee -a ${log_file}
    eval $cmd 2>&1 | tee -a ${log_file}
elif [[ $action == 'print_rankedents' ]]; then
    # Print high scoring pairs.
    cmd="python2 -u $source_path/$script_name.py $action --run_path $run_dir --experiment $experiment"
    echo $cmd
    eval $cmd
elif [[ $action == 'ent2type' ]]; then
    if [[ $experiment != 'ltentvs' ]]; then
        echo "Expects experiment: ltentvs. Got $experiment."
        exit 1
    fi
    # Create the ent2types dict.
    cmd="python2 -u $source_path/$script_name.py $action \
        --int_mapped_path $int_mapped_path \
        --ent2type_path $op2count_path"
    # Pipe this out if you care about it.
    echo $cmd
    eval $cmd
elif [[ $action == 'nearest_fine_ents' ]]; then
    # Find nearest finer entities.
    log_file="${run_dir}/${script_name}-${action}-${experiment}_logs.txt"
    cmd="python2 -u $source_path/$script_name.py $action
        --int_mapped_path $int_mapped_path \
        --op2count_path $op2count_path \
        --run_path $run_dir \
        --experiment $experiment"
    echo $cmd | tee ${log_file}
    eval $cmd 2>&1 | tee -a ${log_file}
else
    echo "Unknown action"
    exit 1
fi