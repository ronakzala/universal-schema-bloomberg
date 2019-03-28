from concrete.util import read_communication_from_file as rcff
import os
import concrete.util
import shutil
import sys
import multiprocessing

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
			set_politicians.add(line.rstrip())
			print(line.rstrip())
			for name in line.split():
				print(name)
				set_politicians.add(name)

	return set_politicians


def match_article_to_politician(fname, set_politicians):
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
	comm =  rcff(dir_path)
	filtered_dictionary = {}
	count = 0

	for (uuid, tokenization_object) in comm.tokenizationForUUID.items():
		#iterating through tokenization objects for one comm file
		print("Count : ", count + 1)
	
		#picking up the NER tagged token list
		tagged_token_list_object = None
		for tto in (tokenization_object.tokenTaggingList):
			if(tto.taggingType == 'NER'):
				tagged_token_list_object = tto.taggedTokenList
				break

		for idx in range(len(tokenization_object.tokenList)):
			#iterating throught the token list to find all the 'PERSON' tags
			token_object = tokenization_object.tokenList[idx]	
			#token object
			tagged_token_object = tagged_token_list_object[token_object.tokenIndex]
			#tagged token Object from the NER tagged token List

			if(tagged_token_object.tag == 'PERSON'):
				person = ''
				while(tagged_token_object.tag == 'PERSON'):
					#if the cuurent token was a 'PERSON' add to person for all continuous 'PERSON' tags
					person += token_object.text
					idx += 1
					token_object = tokenization_object.tokenList[idx]
					tagged_token_object = tagged_token_list_object[token_object.tokenIndex]

				if(person in set_politicians):
					#check if this person is in the politician set, add to final dictionary if true
					if(person in filtered_dictionary):
						filtered_dictionary[person].append(filename) 
					else:
						#tempList = []
						#tempList[0] = filename
						filtered_dictionary[person] = [filename] 
		count += 1

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
	poltician_set = get_politician_set()
	filtered_articles = {politician: [] for politician in politician_set}

	for politician_article_dicts in process_pool.imap_unordered(match_article_to_politician, names, chunksize=mp.cpu_count()):
		for k, v in politician_article_dicts.items():
			filtered_articles[k].extend(v)
	process_pool.close()
	process_pool.join()
	names.close()
	print(filtered_articles)


get_filtered_files()
#TODO : store the dictionary
#Combine this with filter by desk ???
