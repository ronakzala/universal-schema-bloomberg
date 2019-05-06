politician_mapping={}
with open("filtered_names.txt", "r") as f:
	for line in f:
		splitted_line=line.split()
		if not len(splitted_line)==2:
			continue
		politician_mapping[splitted_line[0]]=splitted_line[1]

code_mapping={}
with open("freebase_names.txt", "r") as f:
	for line in f:
		splitted_line=line.split()
		if not len(splitted_line)==2:
			continue
		code_mapping[splitted_line[0]]=splitted_line[1]

i=0
for k,v in code_mapping.items():
	i+=1
	if i<10:
		print (k,v)

f=open('filtered_relations_merged_wordfreq_uniq.txt','w')
with open("filtered_relations_merged_freq_uniq.txt", "r") as f2:
	for line in f2:
		splitted_line=line.split()
		if not len(splitted_line)==4:
			continue
		id1=splitted_line[0]
		id2=splitted_line[1]
		rel=splitted_line[2]
		rel_strength=splitted_line[3]

		name1=id1
		if id1 in politician_mapping:
			name1=politician_mapping[id1]
		elif id1 in code_mapping:
			name1=code_mapping[id1]

		name2=id2
		if id2 in politician_mapping:
			name2=politician_mapping[id2]
		elif id2 in code_mapping:
			name2=code_mapping[id2]

		f.write(name1+"\t"+name2+"\t"+rel+"\t"+rel_strength+"\n")

f.close()
