import json

def mergeStringUnderscore(s):
	splitted_string=s.split()
	for ss in splitted_string:
		ss.capitalize()
	return '_'.join(splitted_string)

alias_mapping_orig={}
with open('bill_cp_to_wiki_cp.json') as f:
	alias_mapping_orig = json.load(f)

alias_mapping={}
for k,v in alias_mapping_orig.items():
	v_new=[]
	for viter in v:
		v_new.append(viter.lower())
	alias_mapping[k.lower()]=v_new

politician_mapping={}
with open("106.txt", "r") as f:
	for line in f:
		politician_name_full=line[:-1]
		politician_name=" ".join(politician_name_full.split()[:-1])
		politician_mapping[politician_name.lower()]=politician_name_full

print (len(politician_mapping))

i=0
f=open('politicians_filtered_relations_only2_freq_uniq_106.txt','w')
with open("filtered_relations_only2_freq_uniq_106.txt", "r") as f2:
	for line in f2:
		splitted_line=line.split('\t')
		if not len(splitted_line)==3:
			continue
		id1=splitted_line[0]
		id2=splitted_line[1]
		rel=splitted_line[2]

		name1=" ".join(id1.split('_')).lower()
		b1=False
		name2=" ".join(id2.split('_')).lower()
		b2=False
		for politician_name,politician_name_full in politician_mapping.items():
			if b1==False:
				if politician_name in name1:
					b1=True
					name1=politician_name_full
				else:
					if politician_name in alias_mapping:
						alias_list=alias_mapping[politician_name]
						for alias_name in alias_list:
							if alias_name in name1:
								b1=True
								name1=politician_name_full
								break

			if b2==False:
				if politician_name in name2:
					b2=True
					name2=politician_name_full
				else:
					if politician_name in alias_mapping:
						alias_list=alias_mapping[politician_name]
						for alias_name in alias_list:
							if alias_name in name2:
								b2=True
								name2=politician_name_full
								break

			if b1 and b2:
				break

		if b1 and b2:
			f.write(mergeStringUnderscore(name1)+"\t"+ mergeStringUnderscore(name2)+"\t"+rel+"\n")
		elif b1:
			f.write(mergeStringUnderscore(name1)+"\t"+ id2+"\t"+rel+"\n")
		elif b2:
			f.write(id1+"\t"+ mergeStringUnderscore(name2)+"\t"+rel+"\n")

		if i%10000==0:
			print (i)
		i+=1


f.close()