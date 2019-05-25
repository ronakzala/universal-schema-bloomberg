import h5py
import argparse
import os
import numpy as np
import json
import logging
import pandas as pd

import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def performance_vs_record_length(model, vote_matrix, bill_matrix, vote_train, predictions, parties, congress_num):
	yes_and_no_votes = vote_matrix[:, :]
	yes_and_no_votes[yes_and_no_votes == 1] = 0
	length_of_records = np.count_nonzero(yes_and_no_votes, axis=0)

	yes_and_no_votes_train = vote_train[:, :]
	yes_and_no_votes_train[yes_and_no_votes_train == 1] = 0
	length_of_records_train = np.count_nonzero(yes_and_no_votes_train, axis=0)

	per_politician_accuracies = [0] * vote_matrix.shape[1]
	
	for i in range(vote_matrix.shape[0]):
		for j in range(vote_matrix.shape[1]):
			if vote_matrix[i][j] > 1:
				if predictions[i, j] == vote_matrix[i][j]:
					per_politician_accuracies[j] += 1

	normalized_accuracies = np.divide(np.array(per_politician_accuracies), length_of_records)
	plt.clf()
	plt.figure()
	plt.margins(0.05) # Optional, just adds 5% padding to the autoscaling
	    
	plt.plot(normalized_accuracies, length_of_records_train, marker='o', linestyle='', ms=6)

	figure_path = "figures/%s_performance_vs_record_length.png" % congress_num 
	plt.savefig(figure_path)
	logging.info("Saved to: %s" % figure_path)

	return normalized_accuracies


def get_voting_breakup(vote_matrix, congress_num, cp_party_info):
	party_votes_dict = {idx: {'Republican': 0, 'Democrat': 0, 'Independent': 0} for idx in range(vote_matrix.shape[0])}
	
	for i in range(vote_matrix.shape[0]):
		rep = [0, 0]
		dem = [0, 0]
		ind = [0, 0]
		for j in range(vote_matrix.shape[1]):
			if vote_matrix[i][j] == 3:
				if cp_party_info[j] == 'Republican':
					rep = [rep[0] + 1, rep[1] + 1]
				if cp_party_info[j] == 'Democrat':
					dem = [dem[0] + 1, dem[1] + 1]
				else:
					ind = [ind[0] + 1, ind[1] + 1]
			elif vote_matrix[i][j] == 2:
				if cp_party_info[j] == 'Republican':
					rep = [rep[0], rep[1] + 1]
				if cp_party_info[j] == 'Democrat':
					dem = [dem[0], dem[1] + 1]
				else:
					ind = [ind[0], ind[1] + 1]
		if rep[0] > (rep[1] // 2):
			party_votes_dict[i]['Republican'] = 3
		else:
			party_votes_dict[i]['Republican'] = 2
		if dem[0] > (dem[1] // 2):
			party_votes_dict[i]['Democrat'] = 3
		else:
			party_votes_dict[i]['Democrat'] = 2
		if ind[0] > (ind[1] // 2):
			party_votes_dict[i]['Independent'] = 3
		else:
			party_votes_dict[i]['Independent'] = 2

	return party_votes_dict


def party_majority_baseline(vote_matrix, congress_num, accuracies):
	with open("../data/preprocessing_metadata/cp_info_%s.txt" % congress_num, 'r') as cp_file:
		cp_info = cp_file.readlines()
	cp_party_info = [x.strip().split()[-1] for x in cp_info]

	if os.path.exists("../data/preprocessing_metadata/majority_vote_info_%s.json" % congress_num):
		with open("../data/preprocessing_metadata/majority_vote_info_%s.json" % congress_num, 'r') as json_file:
			party_votes_dict = json.load(json_file)
	else:
		party_votes_dict = get_voting_breakup(vote_matrix, congress_num, cp_party_info)
		
		with open("../data/preprocessing_metadata/majority_vote_info_%s.json" % congress_num, 'w') as json_file:
			json.dump(party_votes_dict, json_file, indent=4)

	logging.info("%s" % party_votes_dict)

	yes_and_no_votes = vote_matrix[:, :]
	yes_and_no_votes[yes_and_no_votes == 1] = 0
	length_of_records = np.count_nonzero(yes_and_no_votes, axis=0)

	per_politician_accuracies = [0] * vote_matrix.shape[1]
	
	for i in range(vote_matrix.shape[0]):
		for j in range(vote_matrix.shape[1]):
			if vote_matrix[i][j] == party_votes_dict[i][cp_party_info[i]]:
				per_politician_accuracies[j] += 1
				
	normalized_accuracies = np.divide(np.array(per_politician_accuracies), length_of_records)

	for cp, acc, acc_real in zip(cp_info, normalized_accuracies, accuracies):
		logging.info("%s: %f, %f" % (cp, acc, acc_real))


def cluster_congress(model, congress_num):
	with open("../data/preprocessing_metadata/cp_info_%s.txt" % congress_num, 'r') as cp_file:
		cp_info = cp_file.readlines()
	parties = [x.strip().split()[-1] for x in cp_info]
	cp_names = [' '.join(x.strip().split()[:-1]) for x in cp_info]

	cp_embeddings = model.embedding2.weight.detach().numpy()

	cp_embeddings = StandardScaler().fit_transform(cp_embeddings)
	pca = PCA(n_components=2)
	principal_components = pca.fit_transform(cp_embeddings)

	color_dict = {"Republican": 'r', "Democrat": 'b', "Independent": 'y'}
	df = pd.DataFrame(dict(x=principal_components[:, 0], y=principal_components[:, 1], label=parties))

	groups = df.groupby('label')

	plt.clf()
	plt.figure()
	plt.margins(0.05) # Optional, just adds 5% padding to the autoscaling
	for name, group in groups:
	    plt.plot(group.x, group.y, marker='o', linestyle='', ms=6, label=name, color=color_dict[name])
	plt.legend()

	figure_path = "figures/%s_spread.png" % congress_num 
	plt.savefig(figure_path)
	logging.info("Saved to: %s" % figure_path)

	return parties


def top_words_per_party(model, vote_matrix, bill_matrix, predictions, parties, congress_num):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	party_to_words = {}
	logging.info("Here")
	party_keys = ['dem_yes', 'dem_no', 'rep_yes', 'rep_no', 'ind_yes', 'ind_no']
	for party_key in party_keys + ['total']:
		party_to_words[party_key] = np.zeros((1000,))

	for i in range(vote_matrix.shape[0]):
		for j in range(vote_matrix.shape[1]):
			if vote_matrix[i][j] > 1:
				words = bill_matrix[i]
				prediction = predictions[i, j]

				if prediction == 3:
					if parties[j] == 'Democrat':
						party_to_words['dem_yes'] = party_to_words['dem_yes'] + words
					if parties[j] == 'Republican':
						party_to_words['rep_yes'] = party_to_words['rep_yes'] + words
					else:
						party_to_words['ind_yes'] = party_to_words['ind_yes'] + words
				else:
					if parties[j] == 'Democrat':
						party_to_words['dem_no'] = party_to_words['dem_no'] + words
					if parties[j] == 'Republican':
						party_to_words['rep_no'] = party_to_words['rep_no'] + words
					else:
						party_to_words['ind_no'] = party_to_words['ind_no'] + words
				party_to_words['total'] = party_to_words['total'] + words

	with open("../data/preprocessing_metadata/words_%s.txt" % congress_num, 'r') as words_file:
		words_info = words_file.readlines()
	words_info = [x.strip() for x in words_info]

	logging.info("---------------Top Words by Party and Vote:")
	for party_key in party_keys:
		normalized_list = np.divide(party_to_words[party_key], party_to_words['total'])
		top_words_idx = np.argsort(normalized_list)[-20:]
		logging.info(top_words_idx)
		top_words = [words_info[idx] for idx in top_words_idx][::-1]
		logging.info(party_key)
		logging.info(top_words)

	return


def compare_majority_with_trained_model(model, vote_matrix, bill_matrix, vote_train, text_features, predictions, congress_num='106'):
	with open("../data/preprocessing_metadata/cp_info_%s.txt" % congress_num, 'r') as cp_file:
		cp_info = cp_file.readlines()
	parties = [x.strip().split()[-1] for x in cp_info]
	accuracies = performance_vs_record_length(model, vote_matrix, bill_matrix, vote_train, predictions, parties, congress_num)
	party_majority_baseline(vote_matrix, congress_num, accuracies)


def make_plots(model, vote_matrix, bill_matrix, vote_train, text_features, predictions, congress_num='106'):
	logging.basicConfig(level=logging.DEBUG)
	parties = cluster_congress(model, congress_num)
	#top_words_per_party(model, vote_matrix, bill_matrix, predictions, parties, congress_num)
	accuracies = performance_vs_record_length(model, vote_matrix, bill_matrix, vote_train, predictions, parties, congress_num)

	return