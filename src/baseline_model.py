import h5py
import argparse
import os
import numpy as np
import json
import logging

import torch
import torch.nn as nn

import time

parser = argparse.ArgumentParser()
parser.add_argument('--datafile', default='', help='data file')
parser.add_argument('--classifier', default='nb', help='classifier to use')
parser.add_argument('--eta', default='0.0001', help='learning rate')
parser.add_argument('--nepochs', default='20', help='number of epochs')
parser.add_argument('--dp', default='10', help='dp size')
parser.add_argument('--modelpath', default='', help='model path')
parser.add_argument('--congress', default='106', help='congress session')
parser.add_argument('--runeval', default=True, help='Run full eval', type=bool)
parser.add_argument('--lognum', default='', help='Log Num to avoid duplication')

class BillModel(nn.Module):
	def __init__(self, embedding_matrix, model_params):
		super(BillModel,self).__init__()
		# Initialize bill word embeddings with the corresponding glove embeddings
		self.embedding1 = nn.Embedding(
			model_params["num_words"], 
			model_params["word_embed_len"],
			_weight=embedding_matrix
		)
		# Linear layer to transform from bill space to politician space
		self.linear1 = nn.Linear(
			model_params["word_embed_len"],
			model_params["dp_size"]
		)
		# Embedding layer for congresspersons
		self.embedding2 = nn.Embedding(
			model_params["num_cp"],
			model_params["dp_size"]
		)
		self.sigmoid = nn.Sigmoid()
		nn.init.uniform_(self.linear1.weight, -0.01, 0.01)
		nn.init.uniform_(self.embedding2.weight, -0.01, 0.01)
        
	def forward(self, x):
		x0 = x[0].long()
		y1 = self.embedding1(x0)
		y1 = y1.mean(0)
		y1 = self.linear1(y1)
		x1 = x[1].long()
		y2 = self.embedding2(x1)
		y2 = y2.view(y2.size(1))
		y = torch.dot(y1, y2)
		y = self.sigmoid(y)
		return y


def main():
	'''
	Parse input parameters
	dp: Size of interior dot product and congressperson embedding
	'''
	opt = parser.parse_args()
	data_file = h5py.File(opt.datafile, 'r')

	bill_matrix_train = data_file['bill_matrix_train']
	bill_matrix_val = data_file['bill_matrix_val']
	bill_matrix_test = data_file['bill_matrix_test']
	vote_matrix_train = data_file['vote_matrix_train']
	vote_matrix_val = data_file['vote_matrix_val']
	vote_matrix_test = data_file['vote_matrix_test']

	embedding_matrix = data_file['embedding_matrix']
	embedding_matrix = torch.tensor(embedding_matrix)
	num_bills = data_file['num_bills'][0]
	log_name = opt.datafile.split('/')[-1].split('.')[0]

	model_params = {
		"nepochs": int(opt.nepochs),
		"eta": float(opt.eta),
		"dp_size": int(opt.dp),
		"num_words": bill_matrix_train.shape[1],
		"word_embed_len": 50,
		"num_cp": data_file['num_cp'][0],
		"congress": opt.congress,
		"full_eval": opt.runeval,
		"lognum": opt.lognum
	}

	logging.basicConfig(filename='log_files/baseline_%s_%s.log' % (log_name, model_params["lognum"]), filemode='w', level=logging.DEBUG)
	logging.info("Number of bills: %d" % num_bills)
	logging.info("Baseline accuracy: %f" % get_baseline(np.array(vote_matrix_train), np.array(vote_matrix_val), np.array(vote_matrix_test)))

	# Use existing model, located at given path
	if opt.modelpath != '':
		if os.path.isfile(opt.modelpath):
			model = torch.load(opt.modelpath)
			model.eval()
			evaluate_predictions(model, bill_matrix_test, vote_matrix_test, False, opt.congress, full_eval=model_params["full_eval"])
		else:
			logging.error("Error loading model from %s" % opt.modelpath)
		return

	nn_model = train_nn_embed_m(
		bill_matrix_train,
		vote_matrix_train,
		bill_matrix_val,
		vote_matrix_val,
		embedding_matrix,
		model_params
	)
	evaluate_predictions(nn_model, bill_matrix_test, vote_matrix_test, False, opt.congress, full_eval=model_params["full_eval"])


def make_sparse_list_input(inp):
	'''
	Process input for model
	:param inp
	'''
	retset = {}
	for i in range(inp.shape[0]):
		vec_len = inp[i].sum(axis=0)
		vec = torch.zeros(int(vec_len))
		k = 0
		for j in range(inp.shape[1]):
			if inp[i][j] > 0:
				vec[k] = j
				k += 1
		retset[i] = vec

	return retset


def evaluate_predictions(model, bill_matrix, vote_matrix, val=True, congress='106', epoch=1, full_eval=False):
	bill_matrix = make_sparse_list_input(bill_matrix)
	predictions = get_predictions(model, vote_matrix, bill_matrix)
	accuracy, precision, recall, f1 = get_accuracy_stats(np.array(vote_matrix), predictions)
	logging.info("%s Accuracy: %.6f" % ("Val" if val else "Test", accuracy))
	if epoch != 0:
		logging.info("Precision %.6f, Recall %.6f, F1 %.6f" % (precision, recall, f1))
	if val or not full_eval:
		return accuracy

	with open("../data/preprocessing_metadata/eval_info.json", 'r') as infile:
		eval_set = json.load(infile)
	with open("../data/preprocessing_metadata/cp_info_%s.txt" % congress, 'r') as cp_file:
		cp_info = cp_file.readlines()

	cp_info = [x.strip() for x in cp_info]
	set_no_data = set()
	for cp_name in eval_set[congress]["no_data"]:
		idx = cp_info.index(cp_name)#.encode('ascii', 'ignore'))
		set_no_data.add(idx)
	set_rest = set(range(vote_matrix.shape[1])) - set_no_data

	logging.info("Stats for No Data:")
	accuracy, precision, recall, f1 = get_accuracy_stats(np.array(vote_matrix)[:, list(set_no_data)], predictions[:, list(set_no_data)])
	logging.info("%s Accuracy: %.6f" % ("Val" if val else "Test", accuracy))
	logging.info("Precision %.6f, Recall %.6f, F1 %.6f" % (precision, recall, f1))

	logging.info("Stats for Full Data:")
	accuracy, precision, recall, f1 = get_accuracy_stats(np.array(vote_matrix)[:, list(set_rest)], predictions[:, list(set_rest)])
	logging.info("%s Accuracy: %.6f" % ("Val" if val else "Test", accuracy))
	logging.info("Precision %.6f, Recall %.6f, F1 %.6f" % (precision, recall, f1))
	
	return None


def train_nn_embed_m(bill_matrix_train, vote_matrix_train, bill_matrix_test, vote_matrix_test, embedding_matrix, model_params):
	'''
	Train Neural Network embedding
	proc_bill:
		-> Get embeddings of all words in bill
		-> Calculate mean of the word embeddings
		-> Apply y = xA.T + b transformation to it to learn weights of size dp_size * word_embed_len
	proc_cp:
		-> Get embeddings of len dp_size for all politicians (learn weights)
	par_model:
		-> ParallelTable of proc_bill and proc_cp
		-> Pass in the train/test_in matrices to proc_bill, and the train_test_out matrices to proc_cp
	model:
		-> Pass input data through the par_model to learn embedding weights for the bill embeddings, and politician embeddings
		-> Dot product of these two embeddings yields the politicians vote on specific bill
		-> Sigmoid brings it between [0, 1]

	'''
	model = BillModel(embedding_matrix, model_params)
	model = model.double()
	logging.info("Using: %s" % "cuda" if torch.cuda.is_available() else "cpu")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)

	nll = nn.BCELoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=model_params["eta"])

	bill_matrix_train = make_sparse_list_input(bill_matrix_train)
	val_accuracy = 0.0
	for ep in range(model_params["nepochs"]):
		logging.info("Epoch: %d -------------------------" % ep)
		new_accuracy = evaluate_predictions(
			model, bill_matrix_test, vote_matrix_test, True, model_params["congress"], epoch=ep, full_eval=model_params["full_eval"])
		if new_accuracy < val_accuracy:
			break
		val_accuracy = new_accuracy
		for i in range(vote_matrix_train.shape[0]):
			for j in range(vote_matrix_train.shape[1]):
				if vote_matrix_train[i][j] > 1:
					X = bill_matrix_train[i]
					c = torch.ones(1) * j
					y = torch.ones(1) * (vote_matrix_train[i][j] - 2)
					y = y.double()
					X = X.to(device)
					y = y.to(device)
					c = c.to(device) 
					optimizer.zero_grad()
					pred = model([X, c])
					loss = nll(pred, y)
					loss.backward()
					optimizer.step()

	model_path = "saved_models/baseline" + time.strftime("%Y%m%d-%H-%M-%S") + "_" + model_params["congress"] + "_" + model_params["lognum"] + ".pt"
	torch.save(model, model_path)
	logging.info("Saved model to: %s" % model_path)

	return model


def get_predictions(model, vote_matrix, bill_matrix):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	predictions = np.zeros_like(vote_matrix)
	for i in range(vote_matrix.shape[0]):
		for j in range(vote_matrix.shape[1]):
			if vote_matrix[i][j] > 1:
				X = bill_matrix[i]
				c = torch.ones(1) * j
				X = X.to(device)
				c = c.to(device)
				pred = model([X, c])
				predictions[i, j] = 3 if pred.item() >= 0.5 else 2

	return predictions


def get_accuracy_stats(vote_matrix, predictions):
	'''
	Return accuracy, precision, recall, F1
	'''
	predictions[predictions == 1] = 0
	vote_matrix[vote_matrix == 1] = 0

	true_positives_count = ((vote_matrix + predictions) == 6).sum()
	false_positives_count = ((vote_matrix - predictions) == -1).sum()
	false_negatives_count = ((vote_matrix - predictions) == 1).sum()
	true_negatives_count = ((vote_matrix + predictions) == 4).sum()
	
	accuracy = (true_positives_count + true_negatives_count) / (true_positives_count + true_negatives_count + false_positives_count + false_negatives_count)
	precision = true_positives_count / (true_positives_count + false_positives_count)
	recall = true_positives_count / (true_positives_count + false_negatives_count)
	f1 = 2.0 * precision * recall / (precision + recall)
	
	return accuracy, precision, recall, f1


def get_baseline(vote_matrix_train, vote_matrix_val, vote_matrix_test):
	'''
	-- Calculate baseline accuracy for a congress given test and training sets
	'''
	denom_count = (vote_matrix_train > 1).sum() + (vote_matrix_val > 1).sum() + (vote_matrix_test > 1).sum()
	num_count = (vote_matrix_train == 3).sum() + (vote_matrix_val == 3).sum() + (vote_matrix_test == 3).sum()

	return num_count / denom_count


main()
