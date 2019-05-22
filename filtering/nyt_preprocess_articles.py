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

def gen_word_dict(data_path, file_list):
    f=open('filtered_articles_text.txt','w')
    for idx, file_name in enumerate(file_list):
        print (idx)
        full_path = os.path.join(data_path, file_name)
        comm = rcff(full_path)
        if idx % 1000 == 0:
            print("Processing %dth file" % idx)
        #print (comm.text)
        f.write(comm.id+'\n')

def main(arguments):
    args = parser.parse_args(arguments)
    data_path = args.path
    with open("../data/preprocessing_metadata/politicians_filtered_articles.json", 'r') as infile:
        filtered_article_dict = json.load(infile)
    file_list = []
    for k, v in filtered_article_dict.items():
        file_list.extend(v)
    file_list = list(set(file_list))

    gen_word_dict(data_path, file_list)        

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
