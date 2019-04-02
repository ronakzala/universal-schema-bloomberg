#!/bin/bash

declare -i var=0
cat unique_congress2.txt | while read line
do
        grep -i -w $line /mnt/nfs/scratch1/smysore/freebase/freebase_names >> filtered_names.txt
        var=$var+1
        echo $var
done
