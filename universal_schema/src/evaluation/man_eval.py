"""
Helper functions to help manual evaluation.
- Print nearest neighbours or rows, cols and entities.
- Print highest scoring row-col pairs.
"""
from __future__ import unicode_literals
from __future__ import print_function

import os, sys, argparse
import codecs, json, time
import random
from collections import defaultdict
import numpy as np
import tables
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Write unicode to stdout.
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
label_set = {'amt_unit_netag', 'amt_misc_netag', 'cnd_unit_netag',
             'cnd_misc_netag', 'material_netag',
             'target_netag', 'descriptor_netag', 'prop_unit_netag',
             'prop_type_netag', 'synth_aprt_netag', 'char_aprt_netag',
             'brand_netag', 'intrmed_netag', 'prop_misc_netag',
             'cnd_type_netag', 'aprt_unit_netag', 'aprt_des_netag'}


def nearest_roles_si(int_mapped_path, run_path, side_info, dataset):
    """
    - Form a data structure with the predicate-argument pairs. Should be same
        length as role reps array.
    - Find the unique predicate argument pairs and store their indices so you
        can only read those role reps from the array for nearest neighbours
        search.
    - Find nearest neighbours of predicate-argument pairs in this unique subset
        of predicate-argument pair reps.
    :param int_mapped_path: string; directory with raw and im example files.
    :param run_path: string; directory with all run and eval items.
    :param side_info: string; says what side_info to look at when getting role neighbours.
    :return: None.
    """
    if dataset == 'anyt':
        splits = [('ri-test-gold', 'ri-test-gold', 89160),
                  ('ri-test-gold-fcw', 'ri-test-gold-fcw', 89160)]
    elif dataset == 'ms500k':
        splits = [('ri-dev-gold', 'ri-dev-gold', 377),
                  ('ri-test-gold', 'ri-test-gold', 2828)]
    else:
        raise ValueError('Unknown dataset {:s}'.format(dataset))

    for split_fname, split_str, split_size in splits:
        try:
            raw_file = codecs.open(os.path.join(int_mapped_path, '{:s}.json'.format(split_fname)), 'r', 'utf-8')
        except IOError as e:
            sys.stderr.write('{}\n'.format(str(e)))
            continue
        sys.stdout.write('Reading: {:s}\n'.format(raw_file.name))
        # Read in all pairs of predicates and arguments.
        pred_arg_si_pairs = []
        read_examples = 0
        for json_line in raw_file:
            rdict = json.loads(json_line.replace('\r\n', '\\r\\n'), encoding='utf-8')
            for arg, si in zip(rdict['col'], rdict['col_{:s}'.format(side_info)]):
                pred_arg_si_pairs.append((rdict['row'][0], arg, si))
            read_examples += 1
            if read_examples > split_size:
                break
        # Find the unique pairs and their indices.
        uniq_pred_arg_si_pairs = []
        uniq_pred_arg_si_pairs_idxs = []
        uniq_pred_arg_si_pairs_set = set()
        for idx, pair in enumerate(pred_arg_si_pairs):
            if pair not in uniq_pred_arg_si_pairs_set:
                uniq_pred_arg_si_pairs.append(pair)
                uniq_pred_arg_si_pairs_set.add(pair)
                uniq_pred_arg_si_pairs_idxs.append(idx)
        role2idx = dict([(predargsi_tup, idx) for (predargsi_tup, idx) in zip(uniq_pred_arg_si_pairs, range(len(uniq_pred_arg_si_pairs)))])
        idx2role = dict([(val, key) for key, val in role2idx.items()])
        # Read in the numpy array.
        rep_fname = os.path.join(run_path, '{:s}-role_rep.npy'.format(split_str))
        pred_arg_reps = np.load(rep_fname)
        rep_lens_fname = os.path.join(run_path, '{:s}-arg_lens.npy'.format(split_str))
        ex_pred_arg_lens = np.load(rep_lens_fname)
        assert(read_examples == ex_pred_arg_lens.shape[0])
        assert (len(pred_arg_si_pairs) == pred_arg_reps.shape[0])
        sys.stdout.write('Read examples: {:d}\n'.format(read_examples))
        sys.stdout.write('Total predicate-argument pairs: {:d}\n'.format(len(pred_arg_si_pairs)))
        sys.stdout.write('Unique predicate-argument pairs: {:d}\n'.format(len(uniq_pred_arg_si_pairs)))
        # Index into the reps array and read only the reps for the unique pred_arg pairs.
        uniq_pred_arg_reps = pred_arg_reps[np.array(uniq_pred_arg_si_pairs_idxs), :]

        # Build a nearest neighbour search data structure.
        start = time.time()
        nearest_predarg_reps = NearestNeighbors(n_neighbors=11, metric='cosine',
                                                algorithm='brute')
        nearest_predarg_reps.fit(uniq_pred_arg_reps)
        end = time.time()
        sys.stdout.write('Neighbour data structure formed in: {:.4f}s\n'.format(end - start))

        # For the entities print out the nearest entities.
        start = time.time()
        with codecs.open(os.path.join(run_path, '{:s}-role_neighbors.txt'.format(split_str)), 'w', 'utf-8') as resfile:
            count = 0
            for pred, arg, qsi in role2idx.keys():
                intid = role2idx[(pred, arg, qsi)]
                role_vec = uniq_pred_arg_reps[intid, :]
                role_vec = role_vec.reshape(1, role_vec.shape[0])
                neigh_ids = nearest_predarg_reps.kneighbors(role_vec, return_distance=False)
                # Not using the 0 index throws some unhashable type: 'numpy.ndarray' error.
                neighbours = [idx2role[id] for id in list(neigh_ids[0])]
                neighbours_str = ['({:s}, {:s}, {:s})'.format(op, ent, si) for (op, ent, si) in neighbours]
                resfile.write('({:s}, {:s}, {:s})'.format(pred, arg, qsi) + '\n')
                resfile.write('\t'.join(neighbours_str))
                resfile.write('\n\n')
                if count % 100 == 0:
                    sys.stdout.write('Example {:d}: ({:s}, {:s})\n'
                                     .format(count, pred, arg))
                count += 1
                if count > 2000:
                    break
            sys.stdout.write('Wrote results in: {:s}\n'.format(resfile.name))
        end = time.time()
        sys.stdout.write('Nearest neighbours found in: {:.4f}s\n'.format(end - start))


def read_role_reps(int_mapped_path, split_fname, split_size, split_str, run_path):
    """
    Shared util function for the role rep nn printing functions:
    - Form a data structure with the predicate-argument pairs. Should be same
        length as role reps array.
    - Find the unique predicate argument pairs and store their indices so you
        can only read those role reps from the array for nearest neighbours
        search.
    :param int_mapped_path: string; directory with raw and im example files.
    :param split_fname: string; filename for the split to read from the int_mapped_path
    :param split_size: int; number of examples in the split.
    :param split_str: string; split corresponding to the split_fname.
    :param run_path: string; directory with all run and eval items.
    :return:
        uniq_pred_arg_reps: numpy.array [uniq_reps, rep_dim]
        role2idx: dict((pred-str, arg-str): int)
        idx2role: dict(int: (pred-str, arg-str))
    """
    raw_file = codecs.open(os.path.join(int_mapped_path, '{:s}.json'.format(split_fname)), 'r', 'utf-8')
    sys.stdout.write('Reading: {:s}\n'.format(raw_file.name))
    # Read in all pairs of predicates and arguments.
    pred_arg_pairs = []
    read_examples = 0
    for json_line in raw_file:
        rdict = json.loads(json_line.replace('\r\n', '\\r\\n'), encoding='utf-8')
        for arg in rdict['col']:
            pred_arg_pairs.append((rdict['row'][0], arg))
        read_examples += 1
        if read_examples > split_size:
            break
    # Find the unique pairs and their indices.
    uniq_pred_arg_pairs = []
    uniq_pred_arg_pairs_idxs = []
    uniq_pred_arg_pairs_set = set()
    for idx, pair in enumerate(pred_arg_pairs):
        if pair not in uniq_pred_arg_pairs_set:
            uniq_pred_arg_pairs.append(pair)
            uniq_pred_arg_pairs_set.add(pair)
            uniq_pred_arg_pairs_idxs.append(idx)
    role2idx = dict([(predarg_tup, idx) for (predarg_tup, idx) in zip(uniq_pred_arg_pairs, range(len(uniq_pred_arg_pairs)))])
    idx2role = dict([(val, key) for key, val in role2idx.items()])
    # Read in the numpy array.
    rep_fname = os.path.join(run_path, '{:s}-role_rep.npy'.format(split_str))
    pred_arg_reps = np.load(rep_fname)
    rep_lens_fname = os.path.join(run_path, '{:s}-arg_lens.npy'.format(split_str))
    ex_pred_arg_lens = np.load(rep_lens_fname)
    assert(read_examples == ex_pred_arg_lens.shape[0])
    assert (len(pred_arg_pairs) == pred_arg_reps.shape[0])
    sys.stdout.write('Read examples: {:d}\n'.format(read_examples))
    sys.stdout.write('Total predicate-argument pairs: {:d}\n'.format(len(pred_arg_pairs)))
    sys.stdout.write('Unique predicate-argument pairs: {:d}\n'.format(len(uniq_pred_arg_pairs)))
    # Index into the reps array and read only the reps for the unique pred_arg pairs.
    uniq_pred_arg_reps = pred_arg_reps[np.array(uniq_pred_arg_pairs_idxs), :]

    return uniq_pred_arg_reps, role2idx, idx2role


def nearest_roles_within_predicates(int_mapped_path, run_path, dataset):
    """
    - Form a data structure with the predicate-argument pairs. Should be same
        length as role reps array.
    - Find the unique predicate argument pairs and store their indices so you
        can only read those role reps from the array for nearest neighbours
        search.
    - Find nearest neighbours of predicate-argument pairs with the predicate held
        constant. Subject to the criteria that there are a non-trivial number of
        unique pred-arg pairs for the given predicate.
    :param int_mapped_path: string; directory with raw and im example files.
    :param run_path: string; directory with all run and eval items.
    :param dataset: string; read different eval files based on whether its ms500k
        or anyt.
    :return: None.
    """
    if dataset == 'anyt':
        splits = [('ri-test-gold', 'ri-test-gold', 89160),
                  ('ri-test-gold-fcw', 'ri-test-gold-fcw', 89160)]
    elif dataset == 'ms500k':
        splits = [('ri-dev-gold', 'ri-dev-gold', 377),
                  ('ri-test-gold', 'ri-test-gold', 2828)]
    else:
        raise ValueError('Unknown dataset {:s}'.format(dataset))

    for split_fname, split_str, split_size in splits:
        start = time.time()
        uniq_pred_arg_reps, role2idx, idx2role = read_role_reps(int_mapped_path=int_mapped_path,
                                                                split_fname=split_fname, split_size=split_size,
                                                                split_str=split_str, run_path=run_path)
        # Group role2idx by predicate.
        pred2roleidxs = defaultdict(list)
        for role, roleidx in role2idx.iteritems():
            pred, arg = role
            pred2roleidxs[pred].append((role, roleidx))
        resfile = codecs.open(os.path.join(run_path, '{:s}-predicate-role_neighbors.txt'.format(split_str)),
                              'w', 'utf-8')
        sys.stdout.write('Unique predicates: {:d}\n'.format(len(pred2roleidxs)))

        nontrivial_preds = 0
        for pred in pred2roleidxs.iterkeys():
            # If there are too few roles in which the predicate participates then
            # move on.
            if len(pred2roleidxs[pred]) < 20:
                continue
            sys.stdout.write('Predicate: {:s}; Occurences in a role: {:d}\n'.format(pred, len(pred2roleidxs[pred])))
            nontrivial_preds += 1
            predrole_idxs = [idx for role, idx in pred2roleidxs[pred]]
            predrole_reps = uniq_pred_arg_reps[np.array(predrole_idxs), :]
            # Build a nearest neighbour search data structure.
            nearest_predarg_reps = NearestNeighbors(n_neighbors=11, metric='cosine',
                                                    algorithm='brute')
            nearest_predarg_reps.fit(predrole_reps)

            # For the entities print out the nearest entities.
            start = time.time()
            predrole2idx = dict([(role, idx) for idx, (role, oldidx) in enumerate(pred2roleidxs[pred])])
            idx2predrole = dict([(val, key) for key, val in predrole2idx.items()])
            for curpred, arg in predrole2idx.keys():
                intid = predrole2idx[(curpred, arg)]
                role_vec = predrole_reps[intid, :]
                role_vec = role_vec.reshape(1, role_vec.shape[0])
                neigh_ids = nearest_predarg_reps.kneighbors(role_vec, return_distance=False)
                nearest_vecs = predrole_reps[neigh_ids[0], :]
                cosine_distances = metrics.pairwise.cosine_similarity(role_vec, nearest_vecs)
                # Not using the 0 index throws some unhashable type: 'numpy.ndarray' error.
                neighbours = [idx2predrole[id] for id in list(neigh_ids[0])]
                distances = [dist for dist in list(cosine_distances[0])]
                neighbours_str = ['({:s}, {:s}, {:.4f})'.format(op, ent, dist) for ((op, ent), dist) in zip(neighbours,
                                                                                                            distances)]
                resfile.write('({:s}, {:s})'.format(pred, arg) + '\n')
                resfile.write('\t'.join(neighbours_str))
                resfile.write('\n\n')
        sys.stdout.write('Non-trivial predicates: {:d}\n'.format(nontrivial_preds))
        sys.stdout.write('Wrote results in: {:s}\n'.format(resfile.name))
        end = time.time()
        sys.stdout.write('Nearest neighbours found in: {:.4f}s\n'.format(end - start))


def nearest_roles(int_mapped_path, run_path, dataset):
    """
    - Form a data structure with the predicate-argument pairs. Should be same
        length as role reps array.
    - Find the unique predicate argument pairs and store their indices so you
        can only read those role reps from the array for nearest neighbours
        search.
    - Find nearest neighbours of predicate-argument pairs in this unique subset
        of predicate-argument pair reps.
    :param int_mapped_path: string; directory with raw and im example files.
    :param run_path: string; directory with all run and eval items.
    :param dataset: string; read different eval files based on whether its ms500k
        or anyt.
    :return: None.
    """
    if dataset == 'anyt':
        splits = [('ri-test-gold', 'ri-test-gold', 89160),
                  ('ri-test-gold-fcw', 'ri-test-gold-fcw', 89160)]
    elif dataset == 'ms500k':
        splits = [('ri-dev-gold', 'ri-dev-gold', 377),
                  ('ri-test-gold', 'ri-test-gold', 2828)]
    else:
        raise ValueError('Unknown dataset {:s}'.format(dataset))

    for split_fname, split_str, split_size in splits:
        uniq_pred_arg_reps, role2idx, idx2role = read_role_reps(int_mapped_path=int_mapped_path,
                                                                split_fname=split_fname, split_size=split_size,
                                                                split_str=split_str, run_path=run_path)
        # Build a nearest neighbour search data structure.
        start = time.time()
        nearest_predarg_reps = NearestNeighbors(n_neighbors=11, metric='cosine',
                                                algorithm='brute')
        nearest_predarg_reps.fit(uniq_pred_arg_reps)
        end = time.time()
        sys.stdout.write('Neighbour data structure formed in: {:.4f}s\n'.format(end - start))

        # For the entities print out the nearest entities.
        start = time.time()
        with codecs.open(os.path.join(run_path, '{:s}-role_neighbors.txt'.format(split_str)), 'w', 'utf-8') as resfile:
            count = 0
            for pred, arg in role2idx.keys():
                intid = role2idx[(pred, arg)]
                role_vec = uniq_pred_arg_reps[intid, :]
                role_vec = role_vec.reshape(1, role_vec.shape[0])
                neigh_ids = nearest_predarg_reps.kneighbors(role_vec, return_distance=False)
                nearest_vecs = uniq_pred_arg_reps[neigh_ids[0], :]
                cosine_distances = metrics.pairwise.cosine_similarity(role_vec, nearest_vecs)
                # Not using the 0 index throws some unhashable type: 'numpy.ndarray' error.
                neighbours = [idx2role[id] for id in list(neigh_ids[0])]
                distances = [dist for dist in list(cosine_distances[0])]
                neighbours_str = ['({:s}, {:s}, {:.4f})'.format(op, ent, dist) for ((op, ent), dist) in zip(neighbours,
                                                                                                            distances)]
                resfile.write('({:s}, {:s})'.format(pred, arg) + '\n')
                resfile.write('\t'.join(neighbours_str))
                resfile.write('\n\n')
                if count % 100 == 0:
                    sys.stdout.write('Example {:d}: ({:s}, {:s})\n'
                                     .format(count, pred, arg))
                count += 1
                if count > 2000:
                    break
            sys.stdout.write('Wrote results in: {:s}\n'.format(resfile.name))
        end = time.time()
        sys.stdout.write('Nearest neighbours found in: {:.4f}s\n'.format(end - start))


def make_ent2types(int_mapped_path, ent2type_path):
    """
    Create a map going from entity to a dict of the number of times the string is tagged
    as a particular type. This is useful in finding nearest neighbours in the
    ltentvs case where the coarse grained type is necessary to find the finer grained type.
    :param int_mapped_path: string; directory with raw and im example files.
        This should be the im path of the ltentvs experiment.
    :param ent2type_path: string; path where the ent2type dict gets saved.
    :return: None.
    """
    # A dict of dicts {entity: {type: tagged_count}}
    ent2type = {}
    example_file = codecs.open(os.path.join(int_mapped_path, 'train.json'), 'r', 'utf-8')
    count = 0
    for line in example_file:
        rdict = json.loads(line)
        # Expect this to be the ltentvs examples.
        per_ent_type_count = defaultdict(int)
        ents = rdict['col']
        ent_types = rdict['col_types']
        for ent, type in zip(ents, ent_types):
            try:
                ent2type[ent][type] += 1
            except KeyError:
                ent2type[ent] = per_ent_type_count
                ent2type[ent][type] += 1
        count += 1
        if count % 100000 == 0:
            sys.stdout.write('Processing: {:d}; found ents: {:d}\n'.format(count, len(ent2type)))

    # Print the counts for manual examination.
    sortedfile = os.path.join(ent2type_path, 'ent2type-sorted.txt')
    ent_count, type_count = 0, 0
    with codecs.open(sortedfile, 'w', 'utf-8') as f:
        for ent, types in sorted(ent2type.items(), key=lambda (k, v): len(v), reverse=True):
            ent_count += 1
            type_count += len(types)
            # Get a string of type:count in sorted order of count.
            nice = ' '.join(['{}:{}'.format(k,v) for k,v in sorted(types.items(), key=lambda (k,v): v, reverse=True)])
            f.write('{}: {}\n\n'.format(ent, nice))
        sys.stdout.write(
            'entity count: {:d}; average types per entity: {:.4f}\n'.format(ent_count, float(type_count) / ent_count))
        sys.stdout.write('Wrote: {:s}\n'.format(f.name))

    # Write the dict of dicts out.
    ent2typs_fname = os.path.join(ent2type_path, 'ent2type.json')
    jasonified = dict(map(lambda (k, v): tuple((k, dict(v))), ent2type.items()))
    with codecs.open(ent2typs_fname, 'w', 'utf-8') as fp:
        json.dump(jasonified, fp)
        sys.stdout.write('Wrote: {:s}\n'.format(ent2typs_fname))


def build_ltentvs_reps(ent2type, word2idx, type2idx, coarse_embeddings,
                       latent_type_mats):
    # Examine the top 3 types of an entity.
    MAX_TYPES = 3
    type2ents = defaultdict(list)
    type2ent_embeds = defaultdict(list)
    type2ent2idx = defaultdict(dict)
    for ent, type_count in ent2type.iteritems():
        if len(type_count) >= MAX_TYPES:
            most_freq_types = sorted(type_count, key=type_count.get, reverse=True)
            most_freq_types = most_freq_types[:MAX_TYPES]
        else:
            most_freq_types = type_count.keys()
        for etype in most_freq_types:
            type2ents[etype].append(ent)
            ent_vec = coarse_embeddings[word2idx[ent]]
            type_mat = latent_type_mats[type2idx[etype], :, :]
            type2ent_embeds[etype].append(type_mat.dot(ent_vec))
            type2ent2idx[etype][ent] = len(type2ent2idx[etype])
    return type2ents, type2ent_embeds, type2ent2idx


def build_mementvs_reps(ent2type, word2idx, type2idx, coarse_embeddings,
                        latent_type_mats, latent_basis_mats):
    # Examine the top 3 types of an entity.
    MAX_TYPES = 3
    type2ents = defaultdict(list)
    type2ent_embeds = defaultdict(list)
    type2ent2idx = defaultdict(dict)
    for ent, type_count in ent2type.iteritems():
        if len(type_count) >= MAX_TYPES:
            most_freq_types = sorted(type_count, key=type_count.get, reverse=True)
            most_freq_types = most_freq_types[:MAX_TYPES]
        else:
            most_freq_types = type_count.keys()
        for etype in most_freq_types:
            type2ents[etype].append(ent)
            # Compute the entity representation in the new space as done in the
            # model forward pass.
            ent_vec = coarse_embeddings[word2idx[ent]]
            type_mat = latent_type_mats[type2idx[etype], :, :]
            basis_mat = latent_basis_mats[type2idx[etype], :, :]
            basis_weights = type_mat.dot(ent_vec)
            weighted_comb = np.sum(basis_mat * basis_weights[:, None], axis=0)
            type2ent_embeds[etype].append(weighted_comb)
            type2ent2idx[etype][ent] = len(type2ent2idx[etype])
    return type2ents, type2ent_embeds, type2ent2idx


def build_shmementvs_reps(ent2type, word2idx, type2idx, coarse_embeddings,
                          latent_type_mats, shared_basis_mat):
    # Examine the top 3 types of an entity.
    MAX_TYPES = 3
    type2ents = defaultdict(list)
    type2ent_embeds = defaultdict(list)
    type2ent2idx = defaultdict(dict)
    for ent, type_count in ent2type.iteritems():
        if len(type_count) >= MAX_TYPES:
            most_freq_types = sorted(type_count, key=type_count.get, reverse=True)
            most_freq_types = most_freq_types[:MAX_TYPES]
        else:
            most_freq_types = type_count.keys()
        for etype in most_freq_types:
            type2ents[etype].append(ent)
            ent_vec = coarse_embeddings[word2idx[ent]]
            type_mat = latent_type_mats[type2idx[etype], :, :]
            basis_weights = type_mat.dot(ent_vec)
            weighted_comb = np.sum(shared_basis_mat * basis_weights[:, None], axis=0)
            type2ent_embeds[etype].append(weighted_comb)
            type2ent2idx[etype][ent] = len(type2ent2idx[etype])
    return type2ents, type2ent_embeds, type2ent2idx


def nearest_fine_entities(op2count_path, int_mapped_path, run_path, experiment):
    """
    - Read entity embeddings; Read the type-
    - Find nearest neighbours the entities.
    - Print the entities.
    :param op2count_path: string; directory with lemmaop2count json.
    :param int_mapped_path: string; directory with raw and im example files.
    :param run_path: string; directory with all run and eval items.
    :param experiment: string; experiment determines the way finer ents reps are
        calculated.
    :return: None
    """
    # Read idx2word map.
    idx2word_fname = os.path.join(int_mapped_path, 'ent2idx-full.json')
    with codecs.open(idx2word_fname, 'r', 'utf-8') as fp:
        word2idx = json.load(fp)
        sys.stdout.write('Read: {}\n'.format(fp.name))
    sys.stdout.write('word2idx: {}\n'.format(len(word2idx)))
    type2idx_fname = os.path.join(int_mapped_path, 'type2idx-full.json')
    with codecs.open(type2idx_fname, 'r', 'utf-8') as fp:
        type2idx = json.load(fp)
        sys.stdout.write('Read: {}\n'.format(fp.name))
    sys.stdout.write('type2idx: {}\n'.format(len(type2idx)))

    # Read the ent2type dict.
    ent2type = os.path.join(op2count_path, 'ent2type.json')
    with codecs.open(ent2type, 'r', 'utf-8') as fp:
        ent2type = json.load(fp)
        sys.stdout.write('Read: {}\n'.format(fp.name))
    sys.stdout.write('ent2type: {}\n'.format(len(ent2type)))

    # Read embeddings.
    with open(os.path.join(run_path, 'learnt_col_embeddings.npy'), 'r') as fp:
        coarse_embeddings = np.load(fp)
        sys.stdout.write('Read: {}\n'.format(fp.name))
    sys.stdout.write('Embeddings: {}\n'.format(coarse_embeddings.shape))

    with open(os.path.join(run_path, 'learnt_lt_matrices.npy'), 'r') as fp:
        latent_type_mats = np.load(fp)
        sys.stdout.write('Read: {}\n'.format(fp.name))
    sys.stdout.write('Latent type matrices: {}\n'.format(latent_type_mats.shape))

    # For each type build a list of entities, the entity embeddings and a
    # ent2idx.
    if experiment == 'ltentvs':
        type2ents, type2ent_embeds, type2ent2idx = build_ltentvs_reps(
            ent2type, word2idx, type2idx, coarse_embeddings, latent_type_mats)
    elif experiment == 'mementvs':
        with open(os.path.join(run_path, 'learnt_basis_matrices.npy'), 'r') as fp:
            latent_basis_mats = np.load(fp)
            sys.stdout.write('Read: {}\n'.format(fp.name))
        sys.stdout.write('Latent basis matrices: {}\n'.format(latent_basis_mats.shape))

        type2ents, type2ent_embeds, type2ent2idx = build_mementvs_reps(
            ent2type, word2idx, type2idx, coarse_embeddings, latent_type_mats,
            latent_basis_mats)
    elif experiment == 'shmementvs':
        with open(os.path.join(run_path, 'learnt_shbasis_matrix.npy'), 'r') as fp:
            shared_latent_basis_mat = np.load(fp)
            sys.stdout.write('Read: {}\n'.format(fp.name))
        sys.stdout.write('Shared latent basis matrices: {}\n'.format(shared_latent_basis_mat.shape))

        type2ents, type2ent_embeds, type2ent2idx = build_shmementvs_reps(
            ent2type, word2idx, type2idx, coarse_embeddings, latent_type_mats,
            shared_latent_basis_mat)
    else:
        sys.stderr.write('No fine entities for experiment: {:s}\n'.format(experiment))
        return

    # For every entity type do a nearest neighbour search within entities of its
    # coarse type.
    resfile = codecs.open(os.path.join(run_path, 'type_subset_neighbors-cosine.txt'),
                          'w', 'utf-8')
    start = time.time()
    for etype in type2ent_embeds.keys():
        # Convert the list of embeddings to numpy array of embeddings for each type.
        embeds = np.vstack(type2ent_embeds[etype])
        sys.stdout.write('Type: {:s}; Finer entities: {}\n'.format(etype,
                                                                   embeds.shape))
        ent2idx = type2ent2idx[etype]
        idx2ent = dict([(v, k) for k, v in ent2idx.items()])

        # Build a nearest neighbour search data structure.
        nearest_ents = NearestNeighbors(n_neighbors=11, metric='cosine',
                                        algorithm='brute', n_jobs=-1)
        nearest_ents.fit(embeds)
        # Get nearest neighbours of some candidate entities and write them to
        # a file.
        ents = type2ents[etype]
        count = 0
        resfile.write('TYPE: {}\n'.format(etype))
        for ent in ents:
            intid = ent2idx[ent]
            ent_vec = embeds[intid, :]
            ent_vec = ent_vec.reshape(1, ent_vec.shape[0])
            neigh_ids = nearest_ents.kneighbors(ent_vec, return_distance=False)
            # Not using the 0 index throws some unhashable type: 'numpy.ndarray' error.
            neighbours = [idx2ent[id] for id in list(neigh_ids[0])]
            resfile.write(ent.split('_')[0] + '\n')
            resfile.write('\t'.join(neighbours))
            resfile.write('\n\n')
            count += 1
            if count % 50 == 0 and count != 0:
                try:
                    sys.stdout.write('Processing {:d} entity: {:s}\n'.format(count, ent))
                except UnicodeError:
                    pass
            if count > 100:
                break
    resfile.close()
    sys.stdout.write('Wrote results in: {:s}\n'.format(resfile.name))
    end = time.time()
    sys.stdout.write('Nearest neighbours found in: {:.4f}s\n'.format(end - start))


def nearest_operations(op2count_path, int_mapped_path, run_path):
    """
    - Read entity embeddings.
    - Filter out everything except the operation embeddings.
    - Find nearest neighbours the entities.
    - Print the entities.
    :param op2count_path: string; directory with lemmaop2count json.
    :param int_mapped_path: string; directory with raw and im example files.
    :param run_path: string; directory with all run and eval items.
    :return: None
    """
    # Read idx2word map.
    idx2word_file = os.path.join(int_mapped_path, 'op2idx-full.json')
    with codecs.open(idx2word_file, 'r', 'utf-8') as fp:
        op2idx_serial = json.load(fp)
        sys.stdout.write('Read: {}\n'.format(fp.name))
    sys.stdout.write('word2idx: {}\n'.format(len(op2idx_serial)))
    idx2op = dict([(v, k) for k, v in op2idx_serial.items()])

    # Read all embeddings.
    with open(os.path.join(run_path, 'learnt_row_embeddings.npy'), 'r') as fp:
        op_embeddings = np.load(fp)
        sys.stdout.write('Read: {}\n'.format(fp.name))
    sys.stdout.write('Embeddings: {}\n'.format(op_embeddings.shape))

    # Build a nearest neighbour search data structure.
    start = time.time()
    nearest_ents = NearestNeighbors(n_neighbors=11, metric='cosine',
                                    algorithm='brute')
    nearest_ents.fit(op_embeddings)
    end = time.time()
    sys.stdout.write('Neighbour data structure formed in: {:.4f}s\n'.format(end - start))

    # For the entities print out the nearest entities.
    start = time.time()
    with codecs.open(os.path.join(run_path, 'op_neighbors.txt'), 'w', 'utf-8') as resfile:
        count = 0
        for ent in op2idx_serial.keys():
            intid = op2idx_serial[ent]
            ent_vec = op_embeddings[intid, :]
            ent_vec = ent_vec.reshape(1, ent_vec.shape[0])
            neigh_ids = nearest_ents.kneighbors(ent_vec, return_distance=False)
            # Not using the 0 index throws some unhashable type: 'numpy.ndarray' error.
            neighbours = [idx2op[id] for id in list(neigh_ids[0])]
            resfile.write(ent + '\n')
            resfile.write('\t'.join(neighbours))
            resfile.write('\n\n')
            count += 1
            if count > 10000:
                break
        sys.stdout.write('Wrote results in: {:s}\n'.format(resfile.name))
    end = time.time()
    sys.stdout.write('Nearest neighbours found in: {:.4f}s\n'.format(end - start))


def nearest_entities(op2count_path, int_mapped_path, run_path):
    """
    - Read entity embeddings.
    - Find nearest neighbours the entities.
    - Print the entities.
    :param op2count_path: string; directory with lemmaop2count json.
    :param int_mapped_path: string; directory with raw and im example files.
    :param run_path: string; directory with all run and eval items.
    :return: None
    """
    # Read idx2word map.
    idx2word_file = os.path.join(int_mapped_path, 'ent2idx-full.json')
    with codecs.open(idx2word_file, 'r', 'utf-8') as fp:
        word2idx = json.load(fp)
        sys.stdout.write('Read: {}\n'.format(fp.name))
    idx2word = dict([(v, k) for k, v in word2idx.items()])
    sys.stdout.write('word2idx: {}\n'.format(len(word2idx)))

    # Read embeddings.
    with open(os.path.join(run_path, 'learnt_col_embeddings.npy'), 'r') as fp:
        embeddings = np.load(fp)
        sys.stdout.write('Read: {}\n'.format(fp.name))
    sys.stdout.write('Embeddings: {}\n'.format(embeddings.shape))

    # Build a nearest neighbour search data structure.
    start = time.time()
    nearest_ents = NearestNeighbors(n_neighbors=11, metric='cosine',
                                    algorithm='brute', n_jobs=-1)
    nearest_ents.fit(embeddings)
    end = time.time()
    sys.stdout.write('Neighbour data structure formed in: {:.4f}s\n'.format(end - start))

    # For the entities print out the nearest entities.
    start = time.time()
    with codecs.open(os.path.join(run_path, 'entity_neighbors-cosine.txt'), 'w', 'utf-8') as resfile:
        count = 0
        for ent in word2idx.keys():
            intid = word2idx[ent]
            ent_vec = embeddings[intid, :]
            ent_vec = ent_vec.reshape(1, ent_vec.shape[0])
            neigh_ids = nearest_ents.kneighbors(ent_vec, return_distance=False)
            # Not using the 0 index throws some unhashable type: 'numpy.ndarray' error.
            neighbours = [idx2word[id] for id in list(neigh_ids[0])]
            resfile.write(ent + '\n')
            # Omit itself from the list.
            resfile.write('\t'.join(neighbours[1:]))
            resfile.write('\n\n')
            if count % 100 == 0 and count != 0:
                sys.stdout.write('Processing {:d} entity: {:s}\n'.format(count, ent))
            count += 1
            if count > 1000:
                break
        sys.stdout.write('Wrote results in: {:s}\n'.format(resfile.name))
    end = time.time()
    sys.stdout.write('Nearest neighbours found in: {:.4f}s\n'.format(end - start))


def print_scores(run_path, thresh_score, thresh_str):
    """
    Pick the row-cols corresponding to high predicted scores.
    :param run_path: string; directory with all run and eval items.
    :param thresh_score: float [0-1.0]; says which score the probs should
        cut off at.
    :param thresh_str: string; says if the printed values should be higher or
        lower than the threshold passed.
    :return: None.
    """
    splits = ['dev-probs', 'test-probs']
    for split in splits:
        split_res_fname = os.path.join(run_path, split + '.json')
        sys.stdout.write('Processing: {:s}\n'.format(split_res_fname))
        high_score_fname = split_res_fname.split('.')[0] + '-{:s}.txt'.format(thresh_str)
        res_file = codecs.open(split_res_fname, 'r', 'utf-8')
        high_score_file = codecs.open(high_score_fname, 'w', 'utf-8')
        verb_set = set()
        for json_line in res_file:
            try:
                res_dict = json.loads(json_line.replace('\r\n', '\\r\\n'), encoding='utf-8')
            except ValueError:
                continue
            # Based on the suffix go either way.
            if thresh_str == 'high':
                if res_dict['prob'] > thresh_score:
                    verb_set.update(res_dict['row'])
                    high_score_file.write('prob: {:0.4f}\n'.format(res_dict['prob']))
                    high_score_file.write('row: {:s}\n'.format(' '.join(res_dict['row'])))
                    high_score_file.write('col: {:s}\n'.format(' '.join(res_dict['col'])))
                    high_score_file.write('seg: {:s}\n'.format(res_dict['segment']))
                    high_score_file.write('\n\n')
            if thresh_str == 'low':
                if res_dict['prob'] < thresh_score:
                    verb_set.update(res_dict['row'])
                    high_score_file.write('prob: {:0.4f}\n'.format(res_dict['prob']))
                    high_score_file.write('row: {:s}\n'.format(' '.join(res_dict['row'])))
                    high_score_file.write('col: {:s}\n'.format(' '.join(res_dict['col'])))
                    high_score_file.write('seg: {:s}\n'.format(res_dict['segment']))
                    high_score_file.write('\n\n')

        sys.stdout.write('Verb set count: {:d}\n'.format(len(verb_set)))
        high_score_file.write('Verbs:\n')
        high_score_file.write('{:s}\n'.format(' '.join(list(verb_set))))
        sys.stdout.write('Wrote: {:s}\n'.format(high_score_file.name))
        high_score_file.close()


def print_ranked_args(run_path, experiment, num_replicas=499):
    """
    For every possibly positive example print the ranked list of entities that
    could have filled a randomly chosen argument slot instead.
    :param run_path: string; directory with all run and eval items.
    :param num_replicas: int; number of entities sampled to fill the argument slot.
    :param experiment: string; model variant.
    :return: None.
    """
    score_fname = os.path.join(run_path, 'test-500-ranking-probs.json')
    sys.stdout.write('Processing: {:s}\n'.format(score_fname))
    rankedents_fname = os.path.join(run_path, 'test-500-ranked-ents.txt')
    score_file = codecs.open(score_fname, 'r', 'utf-8')
    rankedents_file = codecs.open(rankedents_fname, 'w', 'utf-8')

    wrote_count = 0
    # 2000 random events for which arguments were ranked.
    event_count = 2000
    while event_count > 0:
        original_event = json.loads(score_file.readline())
        rep_count = 16 if experiment == 'typevs' else num_replicas
        ent2score = {}
        # Read the entities and their scores in.
        while rep_count > 0:
            repdict = json.loads(score_file.readline())
            # Get the argument entity that was replaced and its type.
            if experiment in ['typevs']:
                etype = repdict['col'][repdict['replaced_idx']]
                ent2score[(etype, )] = repdict['prob']
            elif experiment in ['typentvs']:
                etype = repdict['col'][repdict['replaced_idx']]
                ent = repdict['col'][repdict['replaced_idx']+1]
                ent2score[(etype, ent)] = repdict['prob']
            elif experiment in ['ltentvs', 'mementvs', 'shmementvs']:
                etype = repdict['col_types'][repdict['replaced_idx']]
                ent = repdict['col'][repdict['replaced_idx']]
                ent2score[(etype, ent)] = repdict['prob']
            rep_count -= 1
            replaced_idx = repdict['replaced_idx']
        # Sort the entities by score.
        sorted_args = sorted(ent2score, key=ent2score.get, reverse=True)
        segment = ' '.join(original_event['segment'].split('_'))
        trigger = original_event['row'][0]
        li = original_event['col']
        li[replaced_idx] = li[replaced_idx] + '-REPLACED'
        args = ', '.join(li)
        prob = original_event['prob']
        # Only write it out if its a well scored original event.
        if prob < 0.6:
            event_count -= 1
            continue
        rankedents_file.write('segment: {:s}\n'.format(segment))
        rankedents_file.write('trigger: {:s}; prob {:.4f}; replaced arg: {:d}\n'.
                              format(trigger, prob, replaced_idx))
        rankedents_file.write('args: {:s}\n'.format(args))
        rankedents_file.write('ranked alternatives: type: entity: prob\n'.format(args))
        # Get the top few args out into a nice output string.
        args = sorted_args[:20]
        out_lines = []
        for arg in args:
            if experiment == 'typevs':
                out_s = '{:s}: {:.4f}'.format(arg[0], ent2score[arg])
            else:
                out_s = '{:s}: {:s}: {:.4f}'.format(arg[0], arg[1], ent2score[arg])
            out_lines.append(out_s)
        # Write the args out, a few args per line.
        out_lines = out_lines[::-1]
        while out_lines:
            try:
                one_line = [out_lines.pop() for i in range(1)]
            except IndexError:
                pass
            rankedents_file.write('; '.join(one_line)+'\n')
        rankedents_file.write('\n')
        event_count -= 1
        wrote_count += 1
    sys.stdout.write('Wrote: {:d} examples\n'.format(wrote_count))
    sys.stdout.write('Wrote: {:s}\n'.format(rankedents_file.name))
    rankedents_file.close()
    score_file.close()


def split_nearest_rowcol_rep(doc_ids, docids2doc, run_path, split_str, row_col_str,
                             sample_idxs):
    """
    For the given split find the nearest rows/columns.
    :param doc_ids: list(string); ids of the documents to query.
    :param docids2doc: dict(docid:dict(row:,col)); id to content maping.
    :param run_path: string; directory with all run and eval items.
    :param split_str: string; train/dev/test.
    :param row_col_str: string; row/col.
    :param sample_idxs: list(int); indices of the sampled representations to
        read for nearest neighbour search.
    :return: None.
    """
    # Read embeddings corresponding to the samples indices.
    rep_fname = os.path.join(run_path, '{:s}-{:s}_rep.h5'.format(split_str, row_col_str))
    rep_file = tables.open_file(rep_fname, mode='r')
    sys.stdout.write('{:s} {:s} learnt reps full size: {}\n'.
                     format(split_str, row_col_str, rep_file.root.data.shape))
    # This appears to be the slowest step (as expected). But maybe I'm just doing
    # this incorrectly. Maybe I can manually use the iterrows function to get this
    # sample of rows. For values of len(sample_idxs)>~30k this is just un-usable.
    rep_vecs = rep_file.root.data[sample_idxs, :]
    sys.stdout.write('{:s} {:s} learnt reps sample size: {}\n'.
                     format(split_str, row_col_str, rep_vecs.shape))
    rep_file.close()

    # Build a nearest neighbour search data structure.
    start = time.time()
    nearest_rc = NearestNeighbors(n_neighbors=6, metric='cosine',
                                  algorithm='brute')
    nearest_rc.fit(rep_vecs)
    end = time.time()
    sys.stdout.write('Neighbour data structure formed in: {:.4f}s\n'.format(end - start))

    # For the entities print out the nearest entities.
    key1 = 'col' if row_col_str == 'col' else 'row'
    key2 = 'col' if row_col_str == 'row' else 'row'
    start = time.time()
    res_fname = os.path.join(run_path, '{:s}-{:s}_neighbors.txt'.
                             format(split_str, row_col_str))
    with codecs.open(res_fname, 'w', 'utf-8') as resfile:
        count = 0
        for doc_id_index, doc_id in enumerate(doc_ids):
            query_vec = rep_vecs[doc_id_index, :]
            query_vec = query_vec.reshape(1, query_vec.shape[0])
            neigh_ids = nearest_rc.kneighbors(query_vec, return_distance=False)
            val1 = ' '.join(docids2doc[doc_id][key1])
            val2 = ' '.join(docids2doc[doc_id][key2])
            resfile.write('Query {:s}: {:s}\n'.format(key1, val1))
            resfile.write('Corresponding {:s}: {:s}\n'.format(key2, val2))
            # Omit itself from the list.
            neighbourdocids = [doc_ids[id] for id in list(neigh_ids[0])][1:]
            for i, neighbourdocid in enumerate(neighbourdocids):
                val1 = ', '.join(docids2doc[neighbourdocid][key1])
                val2 = ', '.join(docids2doc[neighbourdocid][key2])
                resfile.write('Neigh{:d} {:s}: {:s}\n'.format(i+1, key1, val1))
                resfile.write('Neigh{:d} {:s}: {:s}\n'.format(i+1, key2, val2))
            resfile.write('\n\n')
            count += 1
            if count > 3000:
                break
        sys.stdout.write('Wrote results in: {:s}\n'.format(resfile.name))
    end = time.time()
    sys.stdout.write('Nearest neighbours found in: {:.4f}s\n\n'.format(end - start))


def nearest_reps(int_mapped_path, run_path, row_col_str, num_examine=30000):
    """
    For each split of data find the nearest rows and columns of the examples.
    :param int_mapped_path: string; directory with raw and im example files.
    :param run_path: string; directory with all run and eval items.
    :param row_col_str: string; row/col.
    :param num_examine: number of examples to sample from the full giant splits.
    :return: None.
    """
    # Load the hyperparams from disk.
    with codecs.open(os.path.join(run_path, 'run_info.json'), 'r', 'utf-8') as fp:
        run_info = json.load(fp)
        eval_hparams = run_info['eval_hparams']
    ev_train_size, ev_dev_size, ev_test_size = eval_hparams['ev_train_size'], \
                                               eval_hparams['ev_dev_size'], \
                                               eval_hparams['ev_test_size']
    splits = [('test', ev_test_size), ('dev', ev_dev_size)]
    for split_str, split_size in splits:
        num_examine = min(num_examine, split_size)
        # Sample the indices corresponding to a subset of the full set of
        # examples.
        sample_idxs = random.sample(xrange(0, split_size), num_examine)
        # Sort to ensure that the doc_ids and these indices are aligned.
        sample_idxs.sort()
        sample_idxs_set = set(sample_idxs)
        max_to_read = max(sample_idxs_set)
        sys.stdout.write('Max to read: {}\n'.format(max_to_read))
        raw_file = codecs.open(os.path.join(int_mapped_path, '{:s}.json'.format(split_str)), 'r', 'utf-8')
        # Create a docid2doc dict. Always read in 300k examples from the splits
        # And pick a subset on how many embeddings were saved in the run_path.
        doc_ids = []
        docids2doc = {}
        read_examples = 0
        for json_line in raw_file:
            if read_examples in sample_idxs_set:
                rdict = json.loads(json_line.replace('\r\n', '\\r\\n'), encoding='utf-8')
                doc_ids.append(rdict['doc_id'])
                docids2doc[rdict['doc_id']] = {'col': rdict['col'],
                                               'row': rdict['row']}
            read_examples += 1
            if read_examples > max_to_read:
                break
        sys.stdout.write('Created docid2doc for: {}; length: {}\n'.format(raw_file.name, len(docids2doc)))
        split_nearest_rowcol_rep(doc_ids, docids2doc, run_path, split_str, row_col_str, sample_idxs)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest=u'subcommand',
                                       help=u'The action to perform.')
    # Nearest ents. Just nearest neighbours in all entities.
    near_ents = subparsers.add_parser(u'nearest_ents')
    near_ents.add_argument(u'--int_mapped_path', required=True,
                            help=u'Path to the int mapped dataset.')
    near_ents.add_argument(u'--op2count_path', required=True,
                          help=u'Path to the op2count json file.')
    near_ents.add_argument(u'--run_path', required=True,
                           help=u'Path to directory with all run items.')

    # Nearest ents. Just nearest neighbours in finer entities.
    near_fents = subparsers.add_parser(u'nearest_fine_ents')
    near_fents.add_argument(u'--int_mapped_path', required=True,
                            help=u'Path to the int mapped dataset.')
    near_fents.add_argument(u'--op2count_path', required=True,
                            help=u'Path to the op2count json file.')
    near_fents.add_argument(u'--run_path', required=True,
                            help=u'Path to directory with all run items.')
    near_fents.add_argument(u'--experiment', required=True,
                            choices=['ltentvs', 'mementvs', 'shmementvs'],
                            help=u'The experiment for which fine ents are needed.')

    # Nearest ents. Nearest neighbours in all operations.
    near_ops = subparsers.add_parser(u'nearest_ops')
    near_ops.add_argument(u'--int_mapped_path', required=True,
                          help=u'Path to the int mapped dataset.')
    near_ops.add_argument(u'--op2count_path', required=True,
                          help=u'Path to the op2count json file.')
    near_ops.add_argument(u'--run_path', required=True,
                          help=u'Path to directory with all run items.')

    # Nearest neighbours in the predicate-argument pairs.
    near_roles = subparsers.add_parser(u'nearest_roles')
    near_roles.add_argument(u'--int_mapped_path', required=True,
                            help=u'Path to the int mapped dataset.')
    near_roles.add_argument(u'--run_path', required=True,
                            help=u'Path to directory with all run items.')
    near_roles.add_argument(u'--dataset', required=True,
                            choices=['anyt', 'ms500k'],
                            help=u'The dataset being used.')
    near_roles.add_argument(u'--experiment', required=True,
                            choices=['rnentdepvs', 'rnentvs', 'gnentdepvs', 'gnentvs',
                                     'dsentrivs', 'dsentrivsmt', 'dsentetrivs'],
                            help=u'Roles from the experiment being looked at.')

    # Nearest rows. (always the ops except in naryus)
    near_rows = subparsers.add_parser(u'nearest_rows')
    near_rows.add_argument(u'--int_mapped_path', required=True,
                           help=u'Path to the int mapped dataset.')
    near_rows.add_argument(u'--run_path', required=True,
                           help=u'Path to directory with all run items.')

    # Nearest cols. (different based on experiment)
    near_cols = subparsers.add_parser(u'nearest_cols')
    near_cols.add_argument(u'--int_mapped_path', required=True,
                           help=u'Path to the int mapped dataset.')
    near_cols.add_argument(u'--run_path', required=True,
                           help=u'Path to directory with all run items.')

    # High/low scoring examples row-col pairs.
    thresh_scores = subparsers.add_parser('print_scores')
    thresh_scores.add_argument('--run_path', required=True,
                               help='Path to the directory where the run items'
                                    'are stored.')
    thresh_scores.add_argument('--thresh_str', default='high',
                               choices=['high', 'low'],
                               help='Whether to threshold higher or lower than'
                                    'the score..')
    thresh_scores.add_argument('--thresh_score', type=float, default=0.7,
                               help='Score to threshold above or below at.')

    # Rank which other entity in the vocabulary is a valid slot filler for a given
    # argument slot.
    ranked_ents = subparsers.add_parser('print_rankedents')
    ranked_ents.add_argument('--run_path', required=True,
                             help='Path to the directory where the run items'
                                  'are stored.')
    ranked_ents.add_argument(u'--experiment', required=True,
                             choices=['typevs', 'typentvs', 'ltentvs',
                                      'mementvs', 'shmementvs'],
                             help=u'The experiment for which results should be printed.')

    # Make a type2ent map for the ltentvs experiment.
    ent2type = subparsers.add_parser(u'ent2type')
    ent2type.add_argument(u'--int_mapped_path', required=True,
                          help=u'Path to the int mapped ltentvs dataset.')
    ent2type.add_argument(u'--ent2type_path', required=True,
                          help=u'Path to write the ent2type file to.')

    cl_args = parser.parse_args()

    if cl_args.subcommand == 'nearest_ents':
        nearest_entities(int_mapped_path=cl_args.int_mapped_path,
                         op2count_path=cl_args.op2count_path,
                         run_path=cl_args.run_path)
    elif cl_args.subcommand == 'nearest_fine_ents':
        nearest_fine_entities(int_mapped_path=cl_args.int_mapped_path,
                              op2count_path=cl_args.op2count_path,
                              run_path=cl_args.run_path,
                              experiment=cl_args.experiment)
    elif cl_args.subcommand == 'nearest_ops':
        nearest_operations(int_mapped_path=cl_args.int_mapped_path,
                           op2count_path=cl_args.op2count_path,
                           run_path=cl_args.run_path)
    elif cl_args.subcommand == 'nearest_roles':
        if cl_args.dataset == 'ms500k':
            side_info = 'types'
        elif cl_args.dataset == 'anyt':
            side_info = 'deps'
        else:
            side_info = None
        if cl_args.experiment in ['rnentdepvs', 'dsentetrivs']:
            nearest_roles_si(int_mapped_path=cl_args.int_mapped_path, run_path=cl_args.run_path,
                             side_info=side_info, dataset=cl_args.dataset)
        else:
            nearest_roles(int_mapped_path=cl_args.int_mapped_path, run_path=cl_args.run_path,
                          dataset=cl_args.dataset)
            nearest_roles_within_predicates(int_mapped_path=cl_args.int_mapped_path,
                                            run_path=cl_args.run_path,
                                            dataset=cl_args.dataset)
    elif cl_args.subcommand == 'nearest_rows':
        nearest_reps(int_mapped_path=cl_args.int_mapped_path,
                     run_path=cl_args.run_path, row_col_str='row')
    elif cl_args.subcommand == 'nearest_cols':
        nearest_reps(int_mapped_path=cl_args.int_mapped_path,
                     run_path=cl_args.run_path, row_col_str='col')
    elif cl_args.subcommand == 'print_scores':
        print_scores(run_path=cl_args.run_path, thresh_score=cl_args.thresh_score,
                     thresh_str=cl_args.thresh_str)
    elif cl_args.subcommand == 'print_rankedents':
        print_ranked_args(run_path=cl_args.run_path, experiment=cl_args.experiment)
    elif cl_args.subcommand == 'ent2type':
        make_ent2types(int_mapped_path=cl_args.int_mapped_path,
                       ent2type_path=cl_args.ent2type_path)


if __name__ == '__main__':
    main()
