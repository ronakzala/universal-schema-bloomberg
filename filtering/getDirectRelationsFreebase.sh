#!/bin/bash

declare -i var=0
cat filtered_names.txt | while read line
do
	#grep -i -w $line /mnt/nfs/scratch1/smysore/freebase/freebase_names >> filtered_names.txt
	IFS=' ' read -a relation <<< "${line}"
	grep -i -w ${relation[0]} freebase-two-entities.mtx >> filtered_relations_direct.txt
	var=$var+1
	if ! (( $var % 1000 )) ; then
		echo $var
	fi
done
