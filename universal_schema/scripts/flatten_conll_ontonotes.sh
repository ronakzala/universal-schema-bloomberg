#!/usr/bin/env bash
# Flatten the conll ontonotes newswire files into a single directory with all
# the articles under one directory.
# Parse command line args.
while getopts ":s:" opt; do
    case "$opt" in
        s) suffix=$OPTARG ;;
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

if [[ $suffix == '' ]]; then
    conll_formatted_ontonotes="/iesl/canvas/smysore/data/conll-formatted-ontonotes-5.0/raw"
    middle='/data/english/annotations/nw/wsj/'
    # Make the top level destination directory:
    top_level_dest="/iesl/canvas/smysore/data/conll-formatted-ontonotes-5.0/raw_flat"
    extension='gold_conll'
else # suffix can be: sdeps or sdeps_colcc
    conll_formatted_ontonotes="/iesl/canvas/smysore/data/conll-formatted-ontonotes-5.0/raw_${suffix}"
    middle='/nw/wsj/'
    # Make the top level destination directory:
    top_level_dest="/iesl/canvas/smysore/data/conll-formatted-ontonotes-5.0/raw_flat_${suffix}"
    extension='gold_conll.combined'
fi
mkdir -p "$top_level_dest"
for split in "dev" "test" "train"; do
    echo "Flattening: $split"
    dest="$top_level_dest"/"$split"
    mkdir -p "$dest"
    cp "$conll_formatted_ontonotes/$split/$middle"/*/*."$extension" "$dest"
    # Rename the gold_conll.combined to be .gold_conll in the dest for the sdeps and collapsed versions.
    if [[ "$suffix" != '' ]]; then
        # https://unix.stackexchange.com/a/19656/85507
        for f in "$dest"/*.gold_conll.combined; do
            mv -- "$f" "${f%.gold_conll.combined}.gold_conll"
        done
    fi
    echo "Wrote to: $dest"
done