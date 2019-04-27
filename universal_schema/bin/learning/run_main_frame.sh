#!/usr/bin/env bash
use_toy=false
# Parse command line args.
while getopts ":a:e:d:g:s:r:t" opt; do
    case "$opt" in
        a) action=$OPTARG ;;
        e) experiment=$OPTARG ;;
        d) dataset=$OPTARG ;;
        # Train on a specific GPU if specified. Check with nvidia-smi and pick a free one.
        g) gpu_id=$OPTARG;;
        s) suffix=$OPTARG ;;
        r) run_name=$OPTARG ;;
        # If command line switch active then use toy data.
        t) use_toy=true ;;
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
# If the gpuid wasnt passed then it assume that the one set by default is trustworthy.
if [[ $gpu_id != '' ]]; then
    export CUDA_VISIBLE_DEVICES="$gpu_id"
fi
# Make sure required arguments are passed.
if [[ "$action" == '' ]] || [[ "$dataset" == '' ]] || [[ "$experiment" == '' ]] || [[ "$suffix" == '' ]]; then
    echo "Must specify action (-a): run_saved/train_model"
    echo "Must specify experiment (-e): latfeatus"
    echo "Must specify dataset (-d): freebase/anyt/fbanyt"
    echo "Must a meaningful suffix to add to the run directory (-s)."
    exit 1
fi
if [[ "$action" == 'run_saved' ]] && [[ "$run_name" == '' ]]; then
    echo "Must specify dir name of trained model (-r)."
    exit 1
fi

# Getting a random seed as described in:
# https://www.gnu.org/software/coreutils/manual/html_node/Random-sources.html
get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero 2>/dev/null
}

# Source hyperparameters from a config file.
#CUR_PROJ_DIR="/Users/ronakzala/696ds/universal-schema-bloomberg/universal_schema"
hparam_config="$CUR_PROJ_DIR/config/models_config/${dataset}/${experiment}-fixed.conf"
source "$hparam_config"

run_time=`date '+%Y_%m_%d-%H_%M_%S'`
int_mapped_path="$CUR_PROJ_DIR/datasets_proc/${dataset}/${experiment}"

 Create shuffled copies of the dataset; one for each epoch.
if [[ $action == 'train_model' ]]; then
    train_file="$int_mapped_path/train-im-full.json"
    train_neg_file="$int_mapped_path/train-neg-im-full.json"
    dev_file="$int_mapped_path/dev-im-full.json"
    test_file="$int_mapped_path/test-im-full.json"
    shuffled_data_path="$int_mapped_path/shuffled_data"
    mkdir -p "$shuffled_data_path"
    # Create a subset of training examples.
    temp_train="$shuffled_data_path/train-im-full-subset.json"
    head -n "$train_size" "$train_file" > "$temp_train"
    temp_train_neg="$shuffled_data_path/train-neg-im-full-subset.json"
    head -n "$train_size" "$train_neg_file" > "$temp_train_neg"
    # Shuffle this subset file.
    train_file="$temp_train"
    train_neg_file="$temp_train_neg"
    for (( i=0; i<$epochs; i+=1 )); do
        # Shuffle the negative and positive file with the same random seed
        # so individual examples are aligned.
        randomseed=$RANDOM
        fname="$shuffled_data_path/train-im-full-$i.json"
        shuf --random-source=<(get_seeded_random $randomseed) "$train_file" --output="$fname"
        echo "Created: $fname"
        fname="$shuffled_data_path/train-neg-im-full-$i.json"
        shuf --random-source=<(get_seeded_random $randomseed) "$train_neg_file" --output="$fname"
        echo "Created: $fname"
    done
fi

script_name="main_frame"
source_path="$CUR_PROJ_DIR/src/learning"

# Train the model.
if [[ $action == 'train_model' ]]; then
    run_name="${experiment}-${run_time}-${suffix}"
    run_path="$CUR_PROJ_DIR/model_runs/${dataset}/${experiment}/${run_name}"
    log_file="$run_path/train_run_log.txt"
    mkdir -p "$run_path"
    # Base command line call for all models.
    cmd="python2 -u $source_path/$script_name.py  train_model \
            --model_name $experiment \
            --int_mapped_path $int_mapped_path \
            --dataset $dataset \
            --run_path $run_path \
            --train_size $train_size --dev_size $dev_size --test_size $test_size \
            --bsize $bsize --epochs $epochs --lr $lr\
            --decay_by $decay_by --decay_every $decay_every --es_check_every $es_check_every\
            --use_toy $use_toy"
    # Additional args needed by each model. # TODO: What is this used for?(rdim)
    if [[ $experiment == 'latfeatus' ]]; then
        cmd="$cmd --rdim $rdim --dropp $dropp --argdim $argdim --lstm_comp $lstm_comp"
    fi
    eval $cmd 2>&1 | tee -a ${log_file}
elif [[ $action == 'run_saved' ]]; then
    run_path="$CUR_PROJ_DIR/model_runs/${dataset}/${experiment}/${run_name}"
    log_file="$run_path/inference_run_log.txt"
    cmd="python2 -u $source_path/$script_name.py  run_saved_model \
        --model_name $experiment \
        --int_mapped_path $int_mapped_path \
        --run_path $run_path \
        --use_toy $use_toy
        --dataset $dataset"
    eval $cmd 2>&1 | tee ${log_file}
fi