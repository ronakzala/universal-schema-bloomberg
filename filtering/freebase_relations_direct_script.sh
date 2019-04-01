#!/bin/bash

#332283759

declare -i var=0
cat freebase-two-entities.mtx | while read line
do
	#grep -i -w $line /mnt/nfs/scratch1/smysore/freebase/freebase-two-entities.mtx >> filtered_relations_direct.txt
	IFS=' ' read -a relation <<< "${line}"
	relation[0]="${relation[0]} "
	relation[1]="${relation[1]} "

	#echo ${relation[0]}

	c1=$( grep -i -c -w "${relation[0]}" filtered_names.txt )
	c2=$( grep -i -c -w "${relation[1]}" filtered_names.txt )
	#echo $c1

	if (( $c1 > 0 )) && (( $c2 > 0 )); then
		echo "${line}" >> filtered_relations_direct.txt
	elif (( $c1 > 0 )); then
		echo "${line}" >> filtered_relations_direct.txt
		echo "${relation[1]}" >> onehop_names.txt 
	elif (( $c2 > 0 )); then
		echo "${line}" >> filtered_relations_direct.txt
		echo "${relation[0]}" >> onehop_names.txt
	fi

	var=$var+1
	if ! (( $var % 1000 )) ; then
		echo $var
	fi
done
