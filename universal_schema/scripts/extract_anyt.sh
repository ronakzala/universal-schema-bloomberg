#!/usr/bin/env bash
# Unpack all the tar.gz archives of the concretely annotated NYT documents.
tar_dir=$1
# Make directory for extracted comm files.
mkdir "$tar_dir/comms"
for cfile in `ls "$tar_dir"/*".tar.gz"`; do
    echo "Extracting: $cfile"
    tar -xzf "$cfile" -C "$tar_dir/comms"
done