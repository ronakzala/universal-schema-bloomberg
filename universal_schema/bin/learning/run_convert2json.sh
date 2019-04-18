#!/usr/bin/env bash
echo "Starting the convert2json script"
# Parse command line args.
while getopts ":p:r:t:d:" opt; do
    case "$opt" in
        p) parse_type=$OPTARG ;;
        r) relationship_id_file=$OPTARG ;;
        t) train_split=$OPTARG ;;
        d) dev_split=$OPTARG ;;
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

source_path="$CUR_PROJ_DIR/src/learning"
data_source_path="$CUR_PROJ_DIR/datasets_proc/freebase/latfeatus"
script_name="convert2json"

if [[ $parse_type == 'split' ]]; then
    # Shuffling the input - relationship to entity file
    cd $data_source_path
    shuf "$relationship_id_file" > "$relationship_id_file-shuf"
    echo "Created: $relationship_id_file-shuf"
    num_of_lines=$(< "$relationship_id_file" wc -l)

    # Clean the stale train,dev and test files
    rm -f train*
    rm -f dev*
    rm -f test*
    echo "Cleaned the dataset directory"
fi


cmd="python2 $source_path/$script_name.py
    $action \
    --parse_type $parse_type\
    --relationship_id_file $relationship_id_file \
    --train_split $train_split \
    --dev_split $dev_split \
    --num_of_lines $num_of_lines "
echo ${cmd}
eval ${cmd}

#eval python $source_path/$script_name.py -e $entity_id_file -r $relationship_id_file -t $train_split -d $dev_split -n $num_of_lines