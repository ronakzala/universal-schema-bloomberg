#from __future__ import print_function
import numpy as np
import h5py
import argparse
import sys
import os
import json
import csv
from sklearn.model_selection import train_test_split, KFold
from time import time
import gzip
import re

parser = argparse.ArgumentParser()
parser.add_argument('--path', help="Data set", type=str)
parser.add_argument('--congress', help="Congress Number")

NUM_WORDS = 1000
AYE = 3  # Also Yea
NAY = 2  # Also No
PRES = 1 # Also Not Voting
NA = 0   # Not listed at all (not in relevant chamber/Congress)

stop_words = ["ourselves", "hers", "between", "yourself", "but", "again", "there", "about", "once", "during", "out", "very", "having", "with", "they", "own", "an", "be", "some", "for", "do", "its", "yours", "such", "into", "of", "most", "itself", "other", "off", "is", "s", "am", "or", "who", "as", "from", "him", "each", "the", "themselves", "until", "below", "are", "we", "these", "your", "his", "through", "don", "nor", "me", "were", "her", "more", "himself", "this", "down", "should", "our", "their", "while", "above", "both", "up", "to", "ours", "had", "she", "all", "no", "when", "at", "any", "before", "them", "same", "and", "been", "have", "in", "will", "on", "does", "yourselves", "then", "that", "because", "what", "over", "why", "so", "can", "did", "not", "now", "under", "he", "you", "herself", "has", "just", "where", "too", "only", "myself", "which", "those", "i", "after", "few", "whom", "t", "being", "if", "theirs", "my", "against", "a", "by", "doing", "it", "how", "further", "was", "here", "than"]

def gen_congressperson_bill_dict(data, congress_num):
    ''' Create mapping of bill-ID and congressperson-ID
    :param data: Root path for all the data files
    :param congress_num: COngress number
    :return congressperson_dict
    :return bill_dict
    '''
    print("Generating congressperson_dict and bill_dict")
    i = 0
    j = 0
    congressperson_dict = {}
    bill_dict = {}

    for root, _, files in os.walk(data):
        if "votes" in root and "data.json" in files:
            json_data = json.loads(open(root + "/data.json").read())
            if "passage" in json_data["category"] and "bill" in json_data:
                b_id = str(congress_num) + json_data["bill"]["type"] + str(json_data["bill"]["number"])
                if b_id not in bill_dict:
                    bill_dict[b_id] = j
                    j += 1

                for category in json_data["votes"]:
                    for entry in json_data["votes"][category]:
                        try:
                            c_id = entry["id"]
                        except:
                            print "probably the vice president: {0}".format(entry)
                        if c_id not in congressperson_dict:
                            congressperson_dict[c_id] = i
                            i += 1

    return congressperson_dict, bill_dict


def gen_vote_matrix(data, cp_dict, bill_dict, congress_num):
    ''' Create vote matrix from roll call votes.
    :param data: Root path for all data files
    :param cp_dict: congressperson to ID mapping
    :param bill_dict: bill number to ID mapping
    :return vote_matrix: bill vs per congressperson votes
    '''
    print("Generating vote_matrix")
    vote_matrix = np.zeros((len(bill_dict), len(cp_dict)))
    for root, _, files in os.walk(data):
        if "votes" in root and "data.json" in files:
            json_data = json.loads(open(root + "/data.json").read())

            if "passage" in json_data["category"] and "bill" in json_data:
                billno = bill_dict[str(congress_num) + json_data["bill"]["type"] + str(json_data["bill"]["number"])]

                for category in json_data["votes"]:
                    if category == "Aye" or category == "Yea":
                        for entry in json_data["votes"][category]:
                            try:
                                vote_matrix[billno,cp_dict[entry["id"]]] = AYE
                            except:
                                print entry
                    elif category == "Nay" or category == "No":
                        for entry in json_data["votes"][category]:
                            vote_matrix[billno,cp_dict[entry["id"]]] = NAY
                    elif category == "Present" or category == "Not Voting":
                        for entry in json_data["votes"][category]:
                            vote_matrix[billno,cp_dict[entry["id"]]] = PRES

    return vote_matrix


def word_valid(word):
    ''' Checks validity of word (ignore stopwords)
    :param word: Word to be checked
    '''
    return (word.isalpha() and (word not in stop_words) and len(word) > 2)


def word_preprocess(word):
    ''' Cleanup word before using it to create bill matrix.
    :param word: Word to be cleaned
    :return word: Cleaned up word, "NUMBER" if word is numeric
    '''
    word = filter(lambda x: x.isalnum(), word).lower()
    if word.isdigit():
        return "NUMBER"
    else:
        return word


def gen_word_dict(data, bill_dict, congress_num):
    ''' Create dictionary of top 1000 words across all bills.
    :param data: Root path containing all data files
    :param bill_dict: Bill-ID Mapping
    :param congress_num: Congress Number
    :return ret_word_dict
    ''' 
    print("Generating word_dict")
    word_dict = {}
    done_bill_dict = {}

    for root, _, files in os.walk(data):
        if "document.txt" in files and any(os.sep + x[:3] + os.sep in root and (os.sep + x[3:] + os.sep in root) and x not in done_bill_dict for x in bill_dict.keys()):
            for bill in bill_dict.keys():
                if os.sep + bill[:3] + os.sep in root and os.sep + bill[3:] + os.sep in root:
                    done_bill_dict[bill] = 1
            bill_file = open(root + "/document.txt").read()
            for word in re.split('\W+', bill_file):
                word = word_preprocess(word)
                if word_valid(word):
                    if word in word_dict:
                        word_dict[word] += 1
                    else:
                        word_dict[word] = 1

    sorted_word_list = sorted(word_dict.keys(), key=lambda x: word_dict[x], reverse=True)
    sorted_word_list = sorted_word_list[:NUM_WORDS]
    ret_word_dict = {}

    with open("../data/preprocessing_metadata/words_%d.txt" % congress_num, 'w') as outfile:
        for i, entry in enumerate(sorted_word_list):
            ret_word_dict[entry] = i
            outfile.write(entry + '\n')

    return ret_word_dict


def parse_embeddings(pretrained_embed, word_dict):
    ''' Create array of embeddings of top 1000 words.
    Assign random embedding if word not present in glove embeddings.
    :param pretrained_embed: Glove word embeddings
    :param word_dict: Dictionary of top 1000 words
    :return array: of word embeddings
    '''
    print("Parsing Embedding Matrix")
    embeddings = np.zeros((len(word_dict), 50))
    included_words = {}

    with gzip.open(pretrained_embed, 'r') as f:
        content = f.read().split('\n')

        for line in content:
            data = line.split(' ')
            if data[0] in word_dict:
                embeddings[word_dict[data[0]]-1] = map(float, data[1:])
                included_words[data[0]] = 1

    for word in word_dict:
        if word not in included_words:
            embeddings[word_dict[word]-1] = np.random.random(50)

    return np.array(embeddings, dtype=np.float)


def gen_doc_term_matrix(data, bill_dict, word_dict, congress_num):
    ''' Generate doc term matrix using bill text/title.
    TODO: Add code to work with either all words or just title
    :param data: Root path containing all data files
    :param bill_dict: Bill-ID Mapping
    :param word_dict: top 1000 words
    :param congress_num: Congress Number
    :return doc_term_matrix
    '''
    print("Generating document-term matrix")
    doc_term_matrix = np.zeros((len(bill_dict), len(word_dict)))
    done_bill_dict = {}
    bill_details = {}

    for root, _, files in os.walk(data):
        if "document.txt" in files and any((os.sep + x[:3] + os.sep in root) and (os.sep + x[3:] + os.sep in root) and x not in done_bill_dict for x in bill_dict.keys()):
            bill_id = -1

            for bill in bill_dict.keys():
                if os.sep + bill[:3] + os.sep in root and os.sep + bill[3:] + os.sep in root:
                    done_bill_dict[bill] = 1
                    bill_id = bill_dict[bill]
                    bill_details[bill_id] = {"id": bill_id, "path": "", "name": bill, "words": []}

            bill_file = open(root + "/document.txt").read()
            bill_details[bill_id]["path"] = root + "/document.txt"
            curr_word_list = []

            for word in bill_file.split():
                word = word_preprocess(word)
                if word_valid(word) and word in word_dict:
                    doc_term_matrix[bill_id, word_dict[word]] = 1
                    curr_word_list.append(word)
            bill_details[bill_id]["words"] = list(set(curr_word_list))

    with open('../data/preprocessing_metadata/bills_%d.json' % congress_num, 'w') as outfile:
        json.dump(bill_details, outfile, indent=4)

    return doc_term_matrix


def make_party_name_map(cp_dict, congress_num):
    ''' Map Congressperson to Party.
    :param cp_dict: Congressperson-ID Mapping
    :param congress_num: Congress Number
    :return to file: party info
    '''
    print "Making name/party dict"
    id_map = {}

    with open('../data/congressperson_data/legislators-historic.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['bioguide_id'] in cp_dict:
                id_map[cp_dict[row['bioguide_id']]] = (row['first_name'] + " " + row['last_name'], row['party'])
            if row['lis_id'] in cp_dict:
                id_map[cp_dict[row['lis_id']]] = (row['first_name'] + " " + row['last_name'], row['party'])

    with open('../data/congressperson_data/legislators-current.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['bioguide_id'] in cp_dict:
                id_map[cp_dict[row['bioguide_id']]] = (row['first_name'] + " " + row['last_name'], row['party'])
            if row['lis_id'] in cp_dict:
                id_map[cp_dict[row['lis_id']]] = (row['first_name'] + " " + row['last_name'], row['party'])

    cp_info = [id_map[i] for i in range(1 + max(id_map.keys()))]
    f = open("../data/preprocessing_metadata/cp_info_%d.txt" % congress_num, "w")
    for (name, party) in cp_info:
        f.write(name + " " + party + "\n")

    return


def create_eval_data(vote_train, congress_number):
    ''' Zero out the data of a preselected set of congresspeople.
    These congresspeople will most likely be present in NYTimes Articles.
    :param vote_train: vote matrix of training data
    :param congress_number: Congress being preprocessed
    :return modified_votes: Matrix with some votes zeroed out.
    '''
    with open("../data/preprocessing_metadata/eval_info.json", 'r') as infile:
        eval_set = json.load(infile)
    with open("../data/preprocessing_metadata/cp_info_%s.txt" % congress_number, 'r') as cp_file:
        cp_info = cp_file.readlines()
    cp_info = [x.strip() for x in cp_info]
    for cp_name in eval_set[str(congress_number)]["no_data"]:
        idx = cp_info.index(cp_name.encode('ascii', 'ignore'))
        vote_train[:, idx] = 0

    return vote_train


def main(arguments):
    args = parser.parse_args(arguments)
    dataset = args.path
    data = dataset
    congress = int(args.congress)

    congressperson_dict, bill_dict = gen_congressperson_bill_dict(data, congress)
    num_bills = len(bill_dict)
    num_cp = len(congressperson_dict)

    word_dict = gen_word_dict(data, bill_dict, congress)
    doc_term_matrix = gen_doc_term_matrix(data, bill_dict, word_dict, congress)
    print(doc_term_matrix.shape)
    embedding_matrix = parse_embeddings("../data/glove.6B.50d.txt.gz", word_dict)

    vote_matrix = gen_vote_matrix(data, congressperson_dict, bill_dict, congress)

    make_party_name_map(congressperson_dict, congress)

    # May need to change this to hardcoded indices if we want to do eval experiments based on bill info
    # After this step, bill_info will no longer correspond correctly with indices?
    bill_train_val, bill_test, vote_train_val, vote_test = train_test_split(doc_term_matrix, vote_matrix, test_size=0.2)
    vote_train_val = create_eval_data(vote_train_val, congress)
    bill_train, bill_val, vote_train, vote_val = train_test_split(bill_train_val, vote_train_val, test_size=0.25)

    filename = dataset + '.hdf5'
    with h5py.File(filename, "w") as f:
        f['embedding_matrix'] = embedding_matrix
        f['bill_matrix_train'] = bill_train
        f['bill_matrix_test'] = bill_test
        f['bill_matrix_val'] = bill_val
        f['vote_matrix_train'] = vote_train
        f['vote_matrix_test'] = vote_test
        f['vote_matrix_val'] = vote_val
        f['num_bills'] = np.array([len(bill_dict)], dtype=np.int32)
        f['num_cp'] = np.array([len(congressperson_dict)], dtype=np.int32)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
