#!/usr/bin/env bash
echo "Starting the pre-processing script !"

# Parse command line args.
while getopts ":a:e:d:" opt; do
    case "$opt" in
        a) action=$OPTARG ;;
        e) experiment=$OPTARG ;;
        d) dataset=$OPTARG ;;
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
if [[ "$action" == '' ]] || [[ "$dataset" == '' ]] || [[ "$experiment" == '' ]]; then
    echo "Must specify action (-a): int_map/readable_neg"
    echo "Must specify dataset (-d): anyt/freebase/fbanyt"
    echo "Must specify experiment (-e): latfeatus"
    exit 1
fi

# $CUR_PROJ_DIR is a environment variable; manually set outside the script.
#CUR_PROJ_DIR="/Users/ronakzala/696ds/universal-schema-bloomberg/universal_schema"
log_dir="$CUR_PROJ_DIR/logs/learning"
mkdir -p $log_dir

script_name="nn_preproc"
splits_path="$CUR_PROJ_DIR/datasets_proc/${dataset}/${experiment}"
source_path="$CUR_PROJ_DIR/src/learning"

if [[ $action == 'int_map' ]]; then
    log_file="${log_dir}/${script_name}-${action}-${dataset}-${experiment}-full_logs.txt"
    cmd="python2 -u $source_path/$script_name.py
    $action \
    --in_path $splits_path \
    --experiment $experiment \
    --size full
    --dataset $dataset"
    echo ${cmd} | tee ${log_file}
    eval ${cmd} 2>&1 | tee -a ${log_file}
elif [[ $action == 'readable_neg' ]]; then
    # Create shuffled negative data examples for each split.
    train_file="$splits_path/train.json"
    dev_file="$splits_path/dev.json"
    test_file="$splits_path/test.json"
    neg_data_path="$splits_path/neg"
    mkdir -p "$neg_data_path"
    shuf "$train_file" > "$neg_data_path/train-shuf.json"
    echo "Created: $neg_data_path/train-shuf.json"
    shuf "$dev_file" > "$neg_data_path/dev-shuf.json"
    echo "Created: $neg_data_path/dev-shuf.json"
    shuf "$test_file" > "$neg_data_path/test-shuf.json"
    echo "Created: $neg_data_path/test-shuf.json"

    # Create readable negs.
    log_file="${log_dir}/${script_name}-${action}-${dataset}-${experiment}-full_logs.txt"
    cmd="python2 -u $source_path/$script_name.py
    $action \
    --experiment $experiment \
    --in_path $splits_path \
    --dataset $dataset"
    echo ${cmd} | tee ${log_file}
    eval ${cmd} 2>&1 | tee -a ${log_file}

    # Print out the number of lines in the file right after so that errors made due
    # to file appending may be looked at and fixed.
    wc -l "$splits_path"/*-neg.json | tee -a ${log_file}
    # Get rid of the shuffled files.
    rm -r "$neg_data_path"
    echo "Removed: $neg_data_path/*.json"
fi
