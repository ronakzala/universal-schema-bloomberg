freq_mapping={}
with open("rel2count.txt", "r") as f:
	for line in f:
		splitted_line=line.split()
		if not len(splitted_line)==2:
			continue
		object_freq=int(splitted_line[0])
		object_name=splitted_line[1]

		freq_mapping[object_name]=object_freq

target=100

f=open('filtered_relations_merged_freq_uniq.txt','w')
with open("filtered_relations_merged_uniq.txt", "r") as f2:
	for line in f2:
		splitted_line=line.split('\t')
		if (freq_mapping[splitted_line[2]] >= target):
			f.write(line)

f.close()