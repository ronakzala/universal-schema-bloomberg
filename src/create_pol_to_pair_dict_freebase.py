import json
import os
import numpy as np
from scipy.special import expit

us_embedding_path = "us_trial_2"
# in data/congressperson_data 
with open("mid_to_cp.json", 'r') as f:
	mid_to_cp = json.load(f)

# From latest US Model
with open(os.path.join(us_embedding_path, "op2idx-full.json"), 'r') as f:
	op_to_idx = json.load(f)

# From latest US Model
with open(os.path.join(us_embedding_path, "ent2idx-full.json"), 'r') as f:
	ent_to_idx = json.load(f)

# Get all politicians (including those without mids)
with open("../universal-schema-bloomberg/data/preprocessing_metadata/bill_cp_to_wiki_cp.json", 'r') as f:
	bill_cp_to_wiki_cp = json.load(f)

# From latest relations filtering
with open(os.path.join(us_embedding_path, "filtered_relations_merged_freq_uniq.txt"), 'r') as f:
	filtered_rels = f.readlines()

# From latest US Model
rel_array = np.load(os.path.join(us_embedding_path, 'learnt_row_embeddings.npy'))
ent_array = np.load(os.path.join(us_embedding_path, 'learnt_col_embeddings.npy'))
dummy_rel = len(rel_array)
dummy_ent = len(ent_array)
print(dummy_rel, dummy_ent)

triples = [x.strip().split()[:-1] for x in filtered_rels]
print(len(triples))

pol_to_pair = {pol: {'pairs': [], 'scores': []} for pol in bill_cp_to_wiki_cp.keys()}
#print(pol_to_pair)

for triple in triples:
	if triple[0] in mid_to_cp.keys() and mid_to_cp[triple[0]] in pol_to_pair.keys() and triple[2] in op_to_idx.keys() and triple[1] in ent_to_idx.keys():
		emb_ent = np.concatenate((ent_array[ent_to_idx[triple[0]]], ent_array[ent_to_idx[triple[1]]]))
		emb_rel = rel_array[op_to_idx[triple[2]]]
		score = expit(np.dot(emb_ent, emb_rel))
		pol_to_pair[mid_to_cp[triple[0]]]['pairs'].append((op_to_idx[triple[2]], ent_to_idx[triple[1]]))
		pol_to_pair[mid_to_cp[triple[0]]]['scores'].append(str(score))
	if triple[1] in mid_to_cp.keys() and mid_to_cp[triple[1]] in pol_to_pair.keys() and triple[2] in op_to_idx.keys() and triple[0] in ent_to_idx.keys():
		emb_ent = np.concatenate((ent_array[ent_to_idx[triple[0]]], ent_array[ent_to_idx[triple[1]]]))
		emb_rel = rel_array[op_to_idx[triple[2]]]
		score = expit(np.dot(emb_ent, emb_rel))
		pol_to_pair[mid_to_cp[triple[1]]]['pairs'].append((op_to_idx[triple[2]], ent_to_idx[triple[0]]))
		pol_to_pair[mid_to_cp[triple[1]]]['scores'].append(str(score))

for pol in pol_to_pair.keys():
	if len(pol_to_pair[pol]['scores']) == 0:
		print("Not found for %s" % pol)
		pol_to_pair[pol]['pairs'].append((dummy_rel, dummy_ent))
		pol_to_pair[pol]['scores'].append((str(0.5)))

with open(os.path.join(us_embedding_path, "pol_to_pairs.json"), 'w') as f:
	json.dump(pol_to_pair, f, indent=4)
