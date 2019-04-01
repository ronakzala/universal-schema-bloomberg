import h5py
import argparse
import os
import numpy as np

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

	# IN matrices refer to bills, OUT matrices refer to politicians
	big_matrix_train_bills = data_file['big_matrix_train_in']
	big_matrix_train_cp = data_file['big_matrix_train_out']
	big_matrix_test_bills = data_file['big_matrix_test_in']
	big_matrix_test_cp = data_file['big_matrix_test_out']

	embedding_matrix = data_file['embedding_matrix']
	embedding_matrix = torch.tensor(embedding_matrix)
	num_bills = data_file['num_bills'][0]

	model_params = {
		"nepochs": int(opt.nepochs),
		"eta": float(opt.eta),
		"dp_size": int(opt.dp),
		"num_words": big_matrix_train_bills[0].shape[1],
		"word_embed_len": 50,
		"num_cp": data_file['num_cp'][0]
	}

	print("Number of bills: ", num_bills)
	print("Baseline accuracy: ", get_baseline(big_matrix_train_cp[0], big_matrix_test_cp[0]))

	# Use existing model, located at given path
	if opt.modelpath != '':
		if os.path.isfile(opt.modelpath):
			model = torch.load(opt.modelpath)
			model.eval()
			evaluate_predictions(model, big_matrix_test_bills[0], big_matrix_test_cp[0], False)
		else:
			print("Error loading model from %s" % opt.modelpath)
		return


	''' Run model using cross-validation for replication purposes '''
	if opt.classifier == 'nn_embed_m':
		for i in range(big_matrix_train_bills.shape[0]):
			doc_term_matrix_train = big_matrix_train_bills[i]
			doc_term_matrix_test = big_matrix_test_bills[i]
			vote_matrix_train = big_matrix_train_cp[i]
			vote_matrix_test = big_matrix_test_cp[i]

			nn_model = train_nn_embed_m(
				doc_term_matrix_train,
				vote_matrix_train,
				doc_term_matrix_test,
				vote_matrix_test,
				False,
				embedding_matrix,
				model_params
			)
			evaluate_predictions(nn_model, doc_term_matrix_test, vote_matrix_test, False)

	''' Run model without cross-validation for data generation purposes '''
	if opt.classifier == 'nn_embed_m_nocv':
		doc_term_matrix_train = big_matrix_train_bills[0]
		doc_term_matrix_test = big_matrix_test_bills[0]
		vote_matrix_train = big_matrix_train_cp[0]
		vote_matrix_test = big_matrix_test_cp[0]
		
		nn_model = train_nn_embed_m(
			doc_term_matrix_train,
			vote_matrix_train,
			doc_term_matrix_test,
			vote_matrix_test,
			True,
			embedding_matrix,
			model_params
		)
		evaluate_predictions(nn_model, doc_term_matrix_test, vote_matrix_test, False)


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


def evaluate_predictions(model, doc_term_matrix, vote_matrix, val=True):
	mod_test = make_sparse_list_input(doc_term_matrix)
	if val:
		print("Start-of-epoch Stats:")

	mod_test = make_sparse_list_input(doc_term_matrix)
	acc_test = nn_get_acc_m(model, mod_test, vote_matrix)
	print("Test accuracy: ", acc_test)
	prec, recall = nn_get_prec_rec_m(model, mod_test, vote_matrix)
	print ("Precision, Recall, F1:", prec, recall, 2 * prec * recall / (prec + recall))



def train_nn_embed_m(doc_term_matrix_train, vote_matrix_train, doc_term_matrix_test, vote_matrix_test, nocv, embedding_matrix, model_params):
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

	nll = nn.BCELoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=model_params["eta"])

	doc_term_matrix_train = make_sparse_list_input(doc_term_matrix_train)

	for ep in range(model_params["nepochs"]):
		print("Epoch: %d -------------------------" % ep)
		acc_train = nn_get_acc_m(model, doc_term_matrix_train, vote_matrix_train)
		evaluate_predictions(model, doc_term_matrix_test, vote_matrix_test)

		for i in range(vote_matrix_train.shape[0]):
			for j in range(vote_matrix_train.shape[1]):
				if out[i][j] > 1:
					X = doc_term_matrix_train[i]
					c = torch.ones(1) * j
					y = torch.ones(1) * (vote_matrix_train[i][j] - 2)
					y = y.double()
					optimizer.zero_grad()
					pred = model([X, c])
					loss = nll(pred, y)
					loss.backward()
					optimizer.step()

	model_path = time.strftime("%Y%m%d-%H-%M-%S") + ".pt"
	torch.save(model, model_path)

	return model


def nn_get_acc_m(model, inp, out):
	'''
	-- Get accuracy of model given model and test set
	'''
	num_count = 0.0
	denom_count = 0.0
	for i in range(out.shape[0]):
		for j in range(out.shape[1]):

			if out[i][j] > 1:
				X = inp[i]
				c = torch.ones(1) * j
				y = torch.ones(1) * (out[i][j] - 2)
				pred = model([X, c])
				pred = 1 if pred.item() >= 0.5 else 0
				if pred == y[0]:
					num_count += 1
				denom_count += 1

	return num_count / denom_count


def nn_get_prec_rec_m(model, inp, out):
	'''
	-- Get precision and recall of model given model and test set
	'''
	true_positives = 0.0
	positives = 0.0
	trues = 0.0
	for i in range(out.shape[0]):
		for j in range(out.shape[1]):
			if out[i][j] > 1:
				X = inp[i]
				c = torch.ones(1) * j
				y = torch.ones(1) * (out[i][j] - 2)
				pred = model([X, c])
				pred = 1 if pred.item() >= 0.5 else 0
				if (y[0] == 0 and pred == 0):
					true_positives += 1
				if y[0] == 0:
					trues += 1 
				if pred == 0:
					positives += 1 

	return true_positives / positives, true_positives / trues


def get_baseline(out1, out2):
	'''
	-- Calculate baseline accuracy for a congress given test and training sets
	'''
	denom_count = (out1 > 1).sum() + (out2 > 1).sum()
	num_count = (out1 == 3).sum() + (out2 == 3).sum()

	return num_count / denom_count


main()
