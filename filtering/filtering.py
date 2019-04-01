import numpy as np
from fuzzywuzzy import fuzz 
from fuzzywuzzy import process 

# parse the list of politicians
politician_list=[]
politician_set=set()
with open("unique_congress.txt", "r") as f:
	for line in f:
		if (line[-1]=='\n' or line[-1]=='\t'):
			politician_list.append(line[:-1].lower())
			politician_set.add(line[:-1].lower())
		else:
			politician_list.append(line.lower())
			politician_set.add(line.lower())

#print (politician_list)

# get objects (that is, their codes) that resemble the names mentioned in the list of politicians)
politician_codes=set()
target_similarity=80
code_mapping={}
with open("/mnt/nfs/scratch1/smysore/freebase/freebase_names", "r") as f:
	for line in f:
		splitted_line=line.split()
		if not len(splitted_line)==2:
			continue
		object_code=splitted_line[0]
		object_name=splitted_line[1].lower()
		object_name=' '.join(object_name.split('_'))

		code_mapping[object_code]=object_name
		if object_name=="":
			continue
		closest=process.extractOne(object_name,politician_list)

		if (closest[1]>=target_similarity):
			politician_codes.add(object_code)

#print (politician_codes)

# get relations from the other file in freebase
relation_lines_set=set()
one_hop_codes=set()
i=0
with open("/mnt/nfs/scratch1/smysore/freebase/freebase-two-entities.mtx", "r") as f:
	for line in f:
		i+=1
		if (line[-1]=='\n' or line[-1]=='\t'):
			line=line[:-1]

		splitted_line=line.split()
		object1_code=splitted_line[0]
		object2_code=splitted_line[1]
		object_relation=splitted_line[2]
		
		find1=False
		if object1_code in politician_codes:
			find1=True
		find2=False
		if object2_code in politician_codes:
			find2=True

		#print (object1_code," ",object2_code)
		#print (find1," ",find2)
		if (find1==True and find2==True):
			relation_lines_set.add(line)
		elif (find1==True and find2==False):
			relation_lines_set.add(line)
			one_hop_codes.add(object2_code)
		elif (find1==False and find2==True):
			relation_lines_set.add(line)
			one_hop_codes.add(object1_code)

#print (relation_lines_set)

with open("/mnt/nfs/scratch1/smysore/freebase/freebase-two-entities.mtx", "r") as f:
	for line in f:
		if (line[-1]=='\n' or line[-1]=='\t'):
			line=line[:-1]

		splitted_line=line.split()
		object1_code=splitted_line[0]
		object2_code=splitted_line[1]
		object_relation=splitted_line[2]
		
		find1=False
		if object1_code in one_hop_codes:
			find1=True
		find2=False
		if object2_code in one_hop_codes:
			find2=True				

		if (find1==True or find2==True):
			relation_lines_set.add(line)

print (relation_lines_set)

with open('filtered_relations.txt', 'w') as f:
	relation_lines_list=list(relation_lines_set)
	for relation_line in relation_lines_list:
		splitted_line=relation_line.split('\t')
		object1_code=splitted_line[0]
		object2_code=splitted_line[1]
		object_relation=splitted_line[2]
		object_relation_strength=splitted_line[3]

		if (object1_code not in code_mapping) or (object2_code not in code_mapping):
			continue 

		line2=str(code_mapping[object1_code]+" , "+code_mapping[object2_code]+" , "+object_relation+" , "+object_relation_strength+"\n")
		print (line2)
		f.write(line2)
