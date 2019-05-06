#!/bin/bash

declare -i var=0
cat filtered_relations_direct.txt | while read line
do
    #grep -i -w $line /mnt/nfs/scratch1/smysore/freebase/freebase-two-entities.mtx >> filtered_relations_direct.txt
        IFS=' ' read -a relation <<< "${line}"

        #echo ${relation[0]}

        c1=$( fgrep -c -w "${relation[0]}" filtered_names.txt )
        c2=$( fgrep -c -w "${relation[1]}" filtered_names.txt )
        #echo $c1

        if (( $c1 > 0 )) && (( $c2 == 0 )); then
                echo "${relation[1]}" >> onehop_names.txt
        elif (( $c1 == 0 )) && (( $c2 > 0 )); then
                echo "${relation[0]}" >> onehop_names.txt
        fi

        var=$var+1
        if ! (( $var % 1000 )) ; then
                echo $var
        fi
done
