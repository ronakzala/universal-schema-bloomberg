#!/bin/bash

declare -i var=0
cat ids_uniq.txt | while read line
do
	#grep -i -w $line /mnt/nfs/scratch1/smysore/freebase/freebase-two-entities.mtx >> filtered_relations_direct.txt
	IFS='' read -a ids <<< "${line}"

	#echo ${relation[0]}

	c1=$( fgrep -c -w "${ids}" filtered_names.txt )
	#echo $c1

	if (( $c1 == 0 )); then
		fgrep -w "${ids}" /mnt/nfs/scratch1/smysore/freebase/freebase_names >> code_mapping.txt
	fi

	var=$var+1
	if ! (( $var % 1000 )) ; then
		echo $var
	fi
done