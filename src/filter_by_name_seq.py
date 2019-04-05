from concrete.util import read_communication_from_file as rcff
import os
import concrete.util
import shutil
import sys
import multiprocessing as mp
import json
import argparse
from concrete.util.tokenization import get_ner, get_tagged_tokens, get_token_taggings, get_tokens
import logging

dir_path = "/mnt/nfs/work1/mccallum/smysore/data/concretely_annotated_nyt/data/comms"
politicians_file_path = '../data/unique_congress.txt' #path of file which has the names of politicians
filenames_path = "../data/anyt_filenames.txt"
output_dir = "../output/"

# dir_path = "../../../data/anyt_sample"
# politicians_file_path = "../../../universal-schema-bloomberg/data/congressperson_data/unique_congress.txt"
# filenames_path = "filenames.txt"

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
	logging.info("Created set of politicians")

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
	#logging.info('Processing file %s' %fname)
	full_path = os.path.join(dir_path, fname)
	filtered_dictionary = {}
	try:
		comm = rcff(full_path)
	except:
		logging.info("Failed to open %s " % fname)
		return -1

	count = 0

	# import ip:qdb; ipdb.set_trace()
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
				logging.info("Found person : %s in file %s " % (person, fname) )
				if(person in set_politicians):
					logging.info("Matched %s in file %s " % (person, fname))
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
	
	set_politicians = get_politicians_set()

	filtered_articles = {politician: [] for politician in set_politicians}

	count = 0
	for filename in os.listdir(dir_path):
		politician_article_dicts = match_article_to_politician(filename, set_politicians)
		if(isinstance(politician_article_dicts, int)):
			count = count + 1
		else:
			for k,v in politician_article_dicts.items():
				filtered_articles[k].extend(v)

	print("Count of unprocessed articles : %d " % count)
	print(filtered_articles)

	with open("politicians_filtered_articles.json", 'w') as f:
		json.dump(filtered_articles, f, indent=4)

	logging.info('Complete.')


# parsing arguments
p = argparse.ArgumentParser(description="Filters")
p.add_argument('--output-dir', default='output/', type=str,
                   help='path to output dir')

options = p.parse_args()

#making output directory
if not os.path.exists(options.output_dir):
	os.makedirs(options.output_dir)

#set up logging 
logging.basicConfig(filename= options.output_dir + 'log', filemode='w', level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())

get_filtered_files()
