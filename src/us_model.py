import h5py
import argparse
import os
import numpy as np
import json
import logging
import shutil
import torch
import torch.nn as nn

import time

import evaluation_plots

def boolean_string(s):
	if s not in {'False', 'True'}:
		raise ValueError('Not a valid boolean string')
	return s == 'True'


parser = argparse.ArgumentParser()
parser.add_argument('--datafile', default='', help='data file')
parser.add_argument('--classifier', default='nb', help='classifier to use')
parser.add_argument('--eta', default='0.0001', help='learning rate')
parser.add_argument('--nepochs', default='20', help='number of epochs')
parser.add_argument('--dp', default='10', help='dp size')
parser.add_argument('--modelpath', default='', help='model path')
parser.add_argument('--congress', default='106', help='congress session')
parser.add_argument('--runeval', default=False, type=boolean_string, help='Run full eval')
parser.add_argument('--lognum', default='', help='Log num to avoid duplication')


class BillModel(nn.Module):
	def __init__(self, embedding_matrix, rel_embeddings, ent_embeddings, model_params):
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
		self.linear2 = nn.Linear(
			model_params["ent_size"] + model_params["rel_size"],
			model_params["dp_size"]
		)
		# Embedding layer for congresspersons
		self.embedding2 = nn.Embedding(
			model_params["num_cp"],
			model_params["dp_size"]
		)	
		# Embedding layer for US entities
		self.embedding_entities = nn.Embedding(
			model_params["num_ent"],
			model_params["ent_size"],
			_weight=ent_embeddings
		)
		# Embedding layer for US relations
		self.embedding_relations = nn.Embedding(
			model_params["num_rel"],
			model_params["rel_size"],
			_weight=rel_embeddings
		)
		self.sigmoid = nn.Sigmoid()
		self.emb_size = model_params["dp_size"]
		nn.init.uniform_(self.linear1.weight, -0.01, 0.01)
		nn.init.uniform_(self.linear2.weight, -0.01, 0.01)
		nn.init.uniform_(self.embedding2.weight, -0.01, 0.01)
        
	def forward(self, x):
		# Transform bill text into CP space
		x0 = x[0].long()
		y1 = self.embedding1(x0)
		y1 = y1.mean(0)
		y1 = self.linear1(y1)
		
		# Transform CP index to CP embedding
		x1 = x[1].long()
		y2 = self.embedding2(x1)
		y2 = y2.view(y2.size(1))
		
		# US Specific - Feedforward Network
		rel_ents = x[2].long()
		scores = x[3].double()
		rels = rel_ents[:, 0]
		ents = rel_ents[:, 1]
		rel_embs = self.embedding_relations(rels)
		ent_embs = self.embedding_entities(ents)
		# Concatenate rel+ent and pass through a linear layer with tanh non-linearity
		transformed = torch.tanh(self.linear2(torch.cat((rel_embs, ent_embs), 1)))
		# Transpose the output and multiple with scores (Gives a weird column wise multiplication afaik)
		# Finally calculate mean across rows to get a single column
		vnus = torch.mean((transformed.transpose(0, 1) * scores), 1)
		
		# Add dot(v_b, v_c) and dot(v_b, v_text)
		y4 = torch.dot(y1, y2)
		y5 = torch.dot(y1, vnus)
		y6 = torch.add(y4, y5)
		y = self.sigmoid(y6)
		
		return y


def main():
	'''
	Parse input parameters
	dp: Size of interior dot product and congressperson embedding
	'''
	opt = parser.parse_args()
	data_file = h5py.File(opt.datafile, 'r')

	# Read in the learned embeddings from the latest universal schema model
	# These embeddings have been placed in this location locally, not pushed to the repo
	rel_array = np.load('../data/learnt_row_embeddings.npy')
	ent_array = np.load('../data/learnt_col_embeddings.npy')

	# Append a final row to both arrays which serves as the dummy rel, ent
	rel_array = np.append(rel_array, np.zeros((1, rel_array.shape[1])), axis=0)
	ent_array = np.append(ent_array, np.zeros((1, ent_array.shape[1])), axis=0)
	rel_array = torch.tensor(rel_array)
	ent_array = torch.tensor(ent_array)
	
	# Convert all scores from str to float, and change lists to np arrays
	# pol_to_pairs contains a mapping of cp to list of rel, ent, scores
	with open("../data/congressperson_data/pol_to_pairs.json") as f:
		pol_to_pairs = json.load(f)
	for k in pol_to_pairs.keys():
		pol_to_pairs[k]['pairs'] = np.array(pol_to_pairs[k]['pairs'])
		print(pol_to_pairs[k]['pairs'].shape)
		float_scores = [float(score) for score in pol_to_pairs[k]['scores']]
		pol_to_pairs[k]['scores'] = np.array(float_scores)
		
	bill_matrix_train = data_file['bill_matrix_train']
	bill_matrix_val = data_file['bill_matrix_val']
	bill_matrix_test = data_file['bill_matrix_test']
	vote_matrix_train = data_file['vote_matrix_train']
	vote_matrix_val = data_file['vote_matrix_val']
	vote_matrix_test = data_file['vote_matrix_test']

	embedding_matrix = data_file['embedding_matrix']
	embedding_matrix = torch.tensor(embedding_matrix)
	num_bills = data_file['num_bills'][0]

	# Create dict of all usable param values
	model_params = {
		"nepochs": int(opt.nepochs),
		"eta": float(opt.eta),
		"dp_size": int(opt.dp),
		"num_words": bill_matrix_train.shape[1],
		"word_embed_len": 50,
		"num_cp": data_file['num_cp'][0],
		"num_article_words": 2000,
		"num_ent": ent_array.shape[0],
		"ent_size": ent_array.shape[1],
		"num_rel": rel_array.shape[0],
		"rel_size": rel_array.shape[1],
		"congress": opt.congress,
		"full_eval": opt.runeval,
		"make_plots": False,
		"debug": False,
		"lognum": opt.lognum,
		"identifier": 'us_model_%s_%s_%s' % (opt.congress, 'eval' if opt.runeval else 'no_eval', opt.lognum)
	}

	if not os.path.exists("./saved_models"):
		os.mkdir("./saved_models")
	if not os.path.exists("./log_files"):
		os.mkdir("./log_files")

	if model_params["debug"]:
		logging.basicConfig(level=logging.DEBUG)
	else:
		logging.basicConfig(filename='log_files/%s.log' % model_params["identifier"], filemode='w', level=logging.DEBUG)

	logging.info("Number of bills: %d" % num_bills)
	logging.info("Baseline accuracy: %f" % get_baseline(np.array(vote_matrix_train), np.array(vote_matrix_val), np.array(vote_matrix_test)))
	logging.info("Running eval: %s" % opt.runeval)

	# Use existing model, located at given path
	if opt.modelpath != '':
		if os.path.isfile(opt.modelpath):
			if torch.cuda.is_available():
				nn_model = torch.load(opt.modelpath)
			else:
				nn_model = torch.load(opt.modelpath, map_location='cpu')
			nn_model.eval()
		else:
			logging.error("Error loading model from %s" % opt.modelpath)

	else:
		nn_model = train_nn_embed_m(
			bill_matrix_train,
			vote_matrix_train,
			bill_matrix_val,
			vote_matrix_val, 
			embedding_matrix,
			rel_array,
			ent_array,
			pol_to_pairs,
			model_params
		)

	_, predictions = evaluate_predictions(
		nn_model, bill_matrix_test, vote_matrix_test, pol_to_pairs, False, opt.congress, full_eval=model_params["full_eval"])

	'''
	if model_params["make_plots"]:
		evaluation_plots.make_plots(nn_model, vote_matrix_test, bill_matrix_test, text_features, predictions, opt.congress)
	'''


def make_sparse_list_input(inp):
	'''
	Makes list of indices to index into embedding matrix
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


def evaluate_predictions(model, bill_matrix, vote_matrix, pol_to_pairs, val=True, congress='106', epoch=1, full_eval=False):
	bill_matrix = make_sparse_list_input(bill_matrix)
	# Get predictions for the test set
	predictions = get_predictions(model, vote_matrix, bill_matrix, pol_to_pairs, congress)
	
	# Use predictions to calculate accuracy stats for the whole test set
	accuracy, precision, recall, f1 = get_accuracy_stats(np.array(vote_matrix), predictions)
	logging.info("%s Accuracy: %.6f" % ("Val" if val else "Test", accuracy))	
	logging.info("Precision %.6f, Recall %.6f, F1 %.6f" % (precision, recall, f1))
	if val or not full_eval:		
		return accuracy, predictions

	# Do the rest only if eval is switched on
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
	logging.info("Test Accuracy: %.6f" % accuracy)
	logging.info("Precision %.6f, Recall %.6f, F1 %.6f" % (precision, recall, f1))

	logging.info("Stats for Full Data:")
	accuracy, precision, recall, f1 = get_accuracy_stats(np.array(vote_matrix)[:, list(set_rest)], predictions[:, list(set_rest)])
	logging.info("Test Accuracy: %.6f" % accuracy)
	logging.info("Precision %.6f, Recall %.6f, F1 %.6f" % (precision, recall, f1))

	return None, predictions


def train_nn_embed_m(bill_matrix_train, vote_matrix_train, bill_matrix_test, vote_matrix_test,
		embedding_matrix, rel_embeddings, ent_embeddings, pol_to_pairs, model_params):
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
	if not os.path.exists(model_params["identifier"]):
		os.mkdir("temp_models_%s" % model_params["identifier"])
	model = BillModel(embedding_matrix, rel_embeddings, ent_embeddings, model_params)
	model = model.double()
	logging.info("Using: %s" % "cuda" if torch.cuda.is_available() else "cpu")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	
	# Need cp_info to map from indices in the training loop to cp names in pol_to_pairs
	with open("../data/preprocessing_metadata/cp_info_%s.txt" % model_params["congress"], 'r') as cp_file:
		cp_info = cp_file.readlines()
	cp_info = [' '.join(x.strip().split()[:-1]) for x in cp_info]

	nll = nn.BCELoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=model_params["eta"])

	bill_matrix_train = make_sparse_list_input(bill_matrix_train)
	val_accuracy = 0.0
	flag = False
	for ep in range(model_params["nepochs"]):
		logging.info("Epoch: %d -------------------------" % ep)
		new_accuracy, _ = evaluate_predictions(
			model,
			bill_matrix_test,
			vote_matrix_test,
			pol_to_pairs,
			True,
			model_params["congress"],
			epoch=ep,
			full_eval=model_params["full_eval"]
		)
		if new_accuracy < val_accuracy:
			flag = True
			break
		val_accuracy = new_accuracy

		for i in range(vote_matrix_train.shape[0]):
			for j in range(vote_matrix_train.shape[1]):
				if vote_matrix_train[i][j] > 1:
					X = bill_matrix_train[i]
					c = torch.ones(1) * j
					y = torch.ones(1) * (vote_matrix_train[i][j] - 2)
					y = y.double()
					t = torch.from_numpy(pol_to_pairs[cp_info[j]]['pairs'])
					s = torch.from_numpy(pol_to_pairs[cp_info[j]]['scores'])
					#logging.info(text_features[j])
					X = X.to(device)
					y = y.to(device)
					c = c.to(device) 
					t = t.to(device)
					s = s.to(device)
					optimizer.zero_grad()
					pred = model([X, c, t, s])
					#logging.info(pred)
					loss = nll(pred, y)
					loss.backward()
					optimizer.step()

		temp_model_path = "temp_models_%s/epoch_%d.pt" % (model_params["identifier"], ep)
		torch.save(model, temp_model_path)

	if flag:
		logging.info("Loading model from epoch %d" % (ep - 1))
		model = torch.load("temp_models_%s/epoch_%d.pt" % (model_params["identifier"], ep - 1))
		model.eval()
	if os.path.exists("temp_models_%s" % model_params["identifier"]):
		shutil.rmtree("temp_models_%s" % model_params["identifier"])

	model_path = "saved_models/us_" + time.strftime("%Y%m%d-%H-%M-%S") + "_" + model_params["congress"] + "_" + model_params["lognum"] + ".pt"
	torch.save(model, model_path)
	logging.info("Saved model to: %s" % model_path)

	return model


def get_predictions(model, vote_matrix, bill_matrix, pol_to_pairs, congress):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	
	with open("../data/preprocessing_metadata/cp_info_%s.txt" % congress, 'r') as cp_file:
		cp_info = cp_file.readlines()
	cp_info = [' '.join(x.strip().split()[:-1]) for x in cp_info]
	
	count_preds = 0
	predictions = np.zeros_like(vote_matrix)
	for i in range(vote_matrix.shape[0]):
		for j in range(vote_matrix.shape[1]):
			if vote_matrix[i][j] > 1:
				count_preds += 1
				X = bill_matrix[i]
				c = torch.ones(1) * j
				t = torch.from_numpy(pol_to_pairs[cp_info[j]]['pairs'])
				s = torch.from_numpy(pol_to_pairs[cp_info[j]]['scores'])
				X = X.to(device)
				c = c.to(device)
				t = t.to(device)
				s = s.to(device)
				pred = model([X, c, t, s])
				predictions[i, j] = 3 if pred.item() >= 0.5 else 2

	logging.info("Made predictions for : %d out of %d" % (count_preds, vote_matrix.shape[0] * vote_matrix.shape[1]))
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
