from concrete.util import read_communication_from_file as rcff
import os
import concrete.util
import shutil
import sys
import multiprocessing as mp
import json
from concrete.util.tokenization import get_ner, get_tagged_tokens, get_token_taggings, get_tokens

dir_path = "/mnt/nfs/work1/mccallum/smysore/data/concretely_annotated_nyt/data/comms"
politicians_file_path = '../data/congressperson_data/unique_congress.txt'
#path of file which has the names of politicians


def get_politicians_set():
	'''
	Read in Unique Politician List and create set object
	:return set_politicians: Python Set
	'''
	set_politicians = set()

	with open(politicians_file_path, 'r') as filehandle:
		filecontents = filehandle.readlines()
		for line in filecontents:
			set_politicians.add(line.rstrip().lower())
			#for name in line.split():
				#print(name)
				#set_politicians.add(name)

	return set_politicians


def match_article_to_politician(fname):
	'''
	Per article function to return a mapping of politician to article
	- Uses NER tags
	- Pulls out a tagged person using all consecutive 'PERSON' tags
	:param fname: Current file being filtered
	:param set_politicians: Set of relevant politicians
	:return filtered_dictionary: politician name-list of filenames
	'''
	fname = fname.strip()
	full_path = os.path.join(dir_path, fname)
	filtered_dictionary = {}
	try:
		comm = rcff(full_path)
	except:
		return filtered_dictionary

	count = 0

	for (uuid, tokenization_object) in comm.tokenizationForUUID.items():
		#iterating through tokenization objects for one comm file
		ner_list = get_ner(tokenization_object)
		token_list = get_tokens(tokenization_object)

		idx = 0
		while idx < len(token_list):
			token_object = token_list[idx]
			tagged_token_object = ner_list[token_object.tokenIndex]

			if tagged_token_object.tag == 'PERSON':
				person = []
				while tagged_token_object.tag == 'PERSON':
					#if the cuurent token was a 'PERSON' add to person for all continuous 'PERSON' tags
					person.append(token_object.text)
					if idx >= len(token_list) - 1:
						break
					idx += 1
					token_object = token_list[idx]
					tagged_token_object = ner_list[token_object.tokenIndex]

				person = ' '.join(person).lower()
				if(person in set_politicians):
					print("Matched %s" % person)
					#check if this person is in the politician set, add to final dictionary if true
					if(person in filtered_dictionary):
						filtered_dictionary[person].append(fname) 
					else:
						filtered_dictionary[person] = [fname] 
			idx += 1

		return filtered_dictionary



def get_filtered_files():
	'''
	Use multiprocessing to filter files containing relevant politicians.
	- Currently uses NER tags, no desk filtering
	- Print out counts of filtered politician articles
	:return None, dump data to file 
	'''
	names = open("../data/anyt/anyt_filenames.txt", 'r')
	process_pool = mp.Pool(processes=mp.cpu_count(), maxtasksperchild=10000)

	filtered_articles = {politician: [] for politician in set_politicians}
	for politician_article_dicts in process_pool.imap_unordered(match_article_to_politician, names, chunksize=mp.cpu_count()):
		for k, v in politician_article_dicts.items():
			filtered_articles[k].extend(v)
	process_pool.close()
	process_pool.join()
	names.close()
	print(filtered_articles)
	with open("../data/anyt/politicians_filtered_articles.json", 'w') as f:
		json.dump(filtered_articles, f, indent=4)


set_politicians = get_politicians_set()
get_filtered_files()
