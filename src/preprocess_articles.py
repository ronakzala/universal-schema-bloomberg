import numpy as np
import h5py
import logging
import argparse
import sys
import os
import json
import csv
from time import time
import gzip
import re
from concrete.util import read_communication_from_file as rcff
import concrete.util
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--path', help="Data set", type=str)

NUM_WORDS = 2000
stop_words = ["ourselves", "hers", "between", "yourself", "but", "again", "there", "about", "once", "during", "out", "very", "having", "with", "they", "own", "an", "be", "some", "for", "do", "its", "yours", "such", "into", "of", "most", "itself", "other", "off", "is", "s", "am", "or", "who", "as", "from", "him", "each", "the", "themselves", "until", "below", "are", "we", "these", "your", "his", "through", "don", "nor", "me", "were", "her", "more", "himself", "this", "down", "should", "our", "their", "while", "above", "both", "up", "to", "ours", "had", "she", "all", "no", "when", "at", "any", "before", "them", "same", "and", "been", "have", "in", "will", "on", "does", "yourselves", "then", "that", "because", "what", "over", "why", "so", "can", "did", "not", "now", "under", "he", "you", "herself", "has", "just", "where", "too", "only", "myself", "which", "those", "i", "after", "few", "whom", "t", "being", "if", "theirs", "my", "against", "a", "by", "doing", "it", "how", "further", "was", "here", "than"]

def word_valid(word):
    ''' Checks validity of word (ignore stopwords)
    :param word: Word to be checked
    '''
    if not word:
        return False
    return (word.isalpha() and (word not in stop_words) and len(word) > 2)


def word_preprocess(word):
    ''' Cleanup word before using it to create bill matrix.
    :param word: Word to be cleaned
    :return word: Cleaned up word, "NUMBER" if word is numeric
    '''
    if not word.isalnum():
        return None
    word = word.lower() 
    if word.isdigit():
        return "NUMBER"
    else:
        return word


def gen_word_dict(data_path, file_list):
    ''' Create dictionary of top 1000 words across all bills.
    :param data_path: Root path containing all articles
    :param file_list: List of all articles of all politicians
    :return ret_word_dict
    ''' 
    logging.info("Generating word_dict")
    print("Generating word dict")
    word_dict = {}

    for idx, file_name in enumerate(file_list):
        full_path = os.path.join(data_path, file_name)
        comm = rcff(full_path)
        if idx % 1000 == 0:
            print("Processing %dth file" % idx)
        for word in re.split('\W+', comm.text):
            word = word_preprocess(word)
            if word_valid(word):
                if word in word_dict:
                    word_dict[word] += 1
                else:
                    word_dict[word] = 1

    # Sort the words from high frequency to low, and take the top NUM_WORDS
    sorted_word_list = sorted(word_dict.keys(), key=lambda x: word_dict[x], reverse=True)
    sorted_word_list = sorted_word_list[:NUM_WORDS]
    print(sorted_word_list[:100])
    ret_word_dict = {}

    with open("../data/preprocessing_metadata/article_words.txt", 'w') as outfile:
        for i, entry in enumerate(sorted_word_list):
            ret_word_dict[entry] = i
            outfile.write(entry + '\n')
    print("Wrote word list to file")
    return ret_word_dict


def per_politician_doc_term_vector(data_path, file_list, word_dict):
    ''' Generate doc term vector for a politician
    :param data_path: Root path containing all articles
    :param file_list: List of files for current politician
    :param word_dict: top 1000 words
    :return doc_term_vector (Column mean of doc_term_matrix)
    '''
    doc_term_matrix = np.zeros((len(file_list), len(word_dict)))
    done_bill_dict = {}
    bill_details = {}

    for idx, file_name in enumerate(file_list):
        full_path = os.path.join(data_path, file_name)
        comm = rcff(full_path)
        for word in re.split('\W+', comm.text):
            word = word_preprocess(word)
            if word_valid(word) and word in word_dict:
                doc_term_matrix[idx, word_dict[word]] = 1

    return np.mean(doc_term_matrix, axis=0)


def gen_politician_doc_term_matrix(politicians_bill_to_wiki, filtered_article_dict, data_path, word_dict):
    ''' Generate doc term matrix for all politicians.
    :param politicians_bill_to_wiki: Mapping of bill cp name to wiki cp name
    :param data_path: Root path for all articles
    :return doc_term_matrix
    '''
    politician_to_vector_dict = {}
    for cp_name, wiki_names in politicians_bill_to_wiki.items():
        print("Working on: %s" % cp_name)
        vectors = []
        logging.info("Working on: %s" % cp_name)
        # It is possible that a politician is referred to by many names in articles
        # We check all names that map to a specific name in roll call votes
        for cp_wiki_name in wiki_names:
            doc_term_vector = per_politician_doc_term_vector(
                data_path,
                filtered_article_dict[cp_wiki_name],
                word_dict
            )
            vectors.append(doc_term_vector)
        politician_to_vector_dict[cp_name] = np.mean(vectors, axis=0)

    return politician_to_vector_dict


def gen_per_congress_matrix(politician_to_vector_dict, congress_num):
    ''' Generate per congress matrices to be used directly by model.
    :param politician_to_vector_dict: Mapping cp to doc_term_vector
    :param congress_num: Current congress
    :return pol_term_matrix: For current congress
    '''
    with open("../data/preprocessing_metadata/cp_info_%s.txt" % congress_number, 'r') as cp_file:
        cp_info = cp_file.readlines()
    cp_info = [x.strip() for x in cp_info]
    pol_term_matrix = np.array((len(cp_info), NUM_WORDS))
    # This matrix can be indexed using the same index used in the model to refer to a specific cp
    for idx, cp_name in enumerate(cp_info):
        pol_term_matrix[idx, :] = poltician_to_vector_dict[cp_name]
    
    logging.info("Congress: %s, Size of matrix: %d" % (congress_num, pol_term_matrix.shape[0]))
    print("Congress: %s, Size of matrix: %d" % (congress_num, pol_term_matrix.shape[0]))
    return pol_term_matrix


def main(arguments):
    args = parser.parse_args(arguments)
    data_path = args.path
    logging.basicConfig(filename='preprocess_articles.log', filemode='w', level=logging.DEBUG)
    print("Hello")
    logging.info("Hello")
    # Get the JSON of filtered articles corresponding to each politician
    with open("../data/preprocessing_metadata/politicians_filtered_articles.json", 'r') as infile:
        filtered_article_dict = json.load(infile)

    # Get the JSON of CP name in bill -> CP name in wiki/articles
    with open("../data/preprocessing_metadata/bill_cp_to_wiki_cp.json", 'r') as infile:
        politicians_bill_to_wiki = json.load(infile)
    
    # Create file list from all articles for all politicians
    file_list = []
    for k, v in filtered_article_dict.items():
        file_list.extend(v)
    file_list = list(set(file_list))

    word_dict = gen_word_dict(data_path, file_list)
    politician_to_vector_dict = gen_politician_doc_term_matrix(
        politicians_bill_to_wiki,
        filtered_article_dict,
        data_path,
        word_dict
    )
    
    # Create an hdf5 file for each congress
    for congress in range(106, 110):
        pol_term_matrix = gen_per_congress_matrix(politician_to_vector_dict, str(congress))
        filename = congress + '_text_features.hdf5'
        file_path = os.path.join("../data", filename)
        with h5py.File(file_path, "w") as f:
            f['politician_article_matrix'] = pol_term_matrix

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
