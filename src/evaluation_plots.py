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


def cluster_congress(model, congress_num):
	with open("../data/preprocessing_metadata/cp_info_%s.txt" % congress_num, 'r') as cp_file:
		cp_info = cp_file.readlines()
	parties = [x.strip().split()[-1] for x in cp_info]
	cp_names = [' '.join(x.strip().split()[:-1]) for x in cp_info]

	logging.basicConfig(level=logging.DEBUG)
	cp_embeddings = model.embedding2.weight.detach().numpy()
	logging.info(cp_embeddings.shape)

	cp_embeddings = StandardScaler().fit_transform(cp_embeddings)
	pca = PCA(n_components=2)
	principal_components = pca.fit_transform(cp_embeddings)
	logging.info(principal_components.shape)

	color_dict = {"Republican": 'r', "Democrat": 'b', "Independent": 'y'}
	df = pd.DataFrame(dict(x=principal_components[:, 0], y=principal_components[:, 1], label=parties))

	groups = df.groupby('label')

	# Plot
	'''
	fig, ax = plt.subplots()
	ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
	for name, group in groups:
	    ax.plot(group.x, group.y, marker='o', linestyle='', ms=6, label=name, color=color_dict[name])
	ax.legend()
	'''
	plt.figure()
	plt.margins(0.05) # Optional, just adds 5% padding to the autoscaling
	for name, group in groups:
	    plt.plot(group.x, group.y, marker='o', linestyle='', ms=6, label=name, color=color_dict[name])
	plt.legend()

	plt.savefig("figures/%s_spread.png" % congress_num)
	plt.show()

'''
def top_words_per_party(model, vote_matrix, bill_matrix, text_features):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	count_preds = 0
	predictions = np.zeros_like(vote_matrix)
	for i in range(vote_matrix.shape[0]):
		for j in range(vote_matrix.shape[1]):
			if vote_matrix[i][j] > 1:
				count_preds += 1
				X = bill_matrix[i]
				c = torch.ones(1) * j
				t = torch.from_numpy(text_features[j])
				X = X.to(device)
				c = c.to(device)
				t = t.to(device)
				pred = model([X, c, t])
				predictions[i, j] = 3 if pred.item() >= 0.5 else 2

	logging.info("Made predictions for : %d out of %d" % (count_preds, vote_matrix.shape[0] * vote_matrix.shape[1]))
	return predictions
'''

def make_plots(model, vote_matrix, bill_matrix, text_features, congress_num='106'):
	cluster_congress(model, congress_num)
	return