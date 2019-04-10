import json
import ast

def map_names():
	'''
	Map names in bills to names from wikipedia.
	Initial step - used for further manual analysis and corrections.
	'''
	cp_set_from_bills = set()
	for congress in ["106", "107", "108", "109"]:
		with open("../data/preprocessing_metadata/cp_info_%s.txt" % congress, 'r') as cp_file:
			cp_info = cp_file.readlines()
		cp_info = [' '.join(x.strip().split()[:-1]) for x in cp_info]
		cp_set_from_bills.update(cp_info)

	with open("../data/congressperson_data/unique_congress.txt", 'r') as cp_file:
		cp_names_from_wiki = cp_file.readlines()
	cp_set_from_wiki = [x.strip() for x in cp_names_from_wiki]

	mapped_tuples = {}
	for cp_name in list(cp_set_from_bills):
		full_name = cp_name.split()
		mapped_tuples[cp_name] = []
		flag = False
		for cp_name_wiki in cp_set_from_wiki:
			if cp_name_wiki.find(full_name[-1]) != -1:
				mapped_tuples[cp_name].append(cp_name_wiki)
				flag = True
		if not flag:
			print("Not found for: %s" % cp_name)

	with open('../data/preprocessing_metadata/bill_cp_to_wiki_cp.json', 'w') as outfile:
		json.dump(mapped_tuples, outfile, indent=4)


def create_dicts():
	'''
	Using data after manual corrections, create mapping dict in reverse direction
	'''
	with open('../data/preprocessing_metadata/bill_cp_to_wiki_cp.json', 'r') as infile:
		cp_dict = json.load(infile)
	
	wiki_to_bill_mapping = {}

	with open("../data/congressperson_data/unique_congress.txt", 'r') as cp_file:
		cp_names_from_wiki = cp_file.readlines()
	cp_set_from_wiki = set([x.strip() for x in cp_names_from_wiki])
	
	for bill_cp, wiki_cp_list in cp_dict.items():
		for cp_name in wiki_cp_list:
			if cp_name in wiki_to_bill_mapping.keys():
				print("Found again %s" % cp_name)
			wiki_to_bill_mapping[cp_name] = bill_cp
			cp_set_from_wiki.remove(cp_name)

	print(cp_set_from_wiki)
	with open('../data/preprocessing_metadata/wiki_cp_to_bill_cp.json', 'w') as outfile:
		json.dump(wiki_to_bill_mapping, outfile, indent=4)

create_dicts()
