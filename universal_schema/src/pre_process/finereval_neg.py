"""
Create a set of examples to check how well ranking by the model works.
"""
from __future__ import unicode_literals
from __future__ import print_function
import sys
import os
import codecs
import json
import random
import time

import data_utils
import pp_settings as pps

sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
random.seed('somefixedrandomstate')


def get_random_enttype(ent2type_li, num_samples=1):
    """
    Given the ent2type map as a list return a random entity and its type.
    :param ent2type_li: list((ent, dict(type:int))); Says which entity was tagged with
        which type how many times across the corpus.
    :param num_samples: list
    :return:res: list(tuple(ent, type)):
                ent: string; the entity.
                type: string; the most likely type of the entity.
    """
    # This returns a tuple of the entity and the count of the types.
    randents = random.sample(ent2type_li, num_samples)
    res = []
    for ent, type_count in randents:
        most_freq_types = sorted(type_count, key=type_count.get, reverse=True)
        # Assume the most frequent to be the correct tag.
        etype = most_freq_types[0]
        res.append((ent, etype))
    return res


def make_finer_neg(in_path, map_path, experiment_str, neg_arg_percent):
    """
    For the positive example create a negative example which replaces a fraction
    of the positive arguments with a randomly sampled negative example.
    :param in_path: dir with example-per-line json files: 'train', 'dev', 'test'.
    :param map_path: dir with the maps going from the entity to the types its
        tagged with.
    :param experiment_str: string; says which experiment you're making the intmaps
        for.
    :param neg_arg_percent: int; [0-100]; What fraction of the arguments should get
        negated.
    :return: None.
    """
    ent2type = os.path.join(map_path, 'ent2type.json')
    with codecs.open(ent2type, 'r', 'utf-8') as fp:
        ent2type = json.load(fp)
        sys.stdout.write('Read: {}\n'.format(fp.name))
    # Make it into a list so its easier to get random things from it.
    ent2type_li = ent2type.items()
    # Fix the order this comes in somehow so everything is deterministic
    # across the data generated for all the models.
    ent2type_li.sort()
    sys.stdout.write('ent2type: {}\n'.format(len(ent2type_li)))

    # Add other splits here if needed later.
    labels_li = list(pps.label_set)
    splits = ['test']
    for split in splits:
        # File containing per-line jsons for each positive example.
        pos_split_fname = os.path.join(in_path, split) + '.json'
        # File to write out.
        neg_split_fname = os.path.join(in_path, split) + '-{}-finerneg.json'.format(neg_arg_percent)

        # Open everything up.
        pos_split_file = codecs.open(pos_split_fname, 'r', 'utf-8')
        neg_split_file = codecs.open(neg_split_fname, 'w', 'utf-8')
        start_time = time.time()
        sys.stdout.write('Processing: {:s}\n'.format(pos_split_file.name))
        example_count = 0
        arg_count = 0
        replaced_count = 0
        for pos_json in data_utils.read_perline_json(pos_split_file):
            # Same as the pos row.
            neg_doc_id = pos_json['doc_id']
            neg_row = pos_json['row']
            neg_segment = pos_json['segment']
            # Make the finer negative col.
            if experiment_str in ['typevs']:
                neg_col = pos_json['col']
                # Replace a fixed number of elements from the column with a random type.
                to_replace = int(len(neg_col)*(float(neg_arg_percent)/100))
                # If it turns out to be 0 then replace a single argument.
                to_replace = 1 if to_replace == 0 else to_replace
                replace_idxs = random.sample(range(len(neg_col)), to_replace)
                for idx in replace_idxs:
                    neg_col[idx] = random.choice(labels_li) + '_netag'
                # Write the example out to a file.
                im_data = {
                    'row': neg_row,
                    'col': neg_col,
                    'segment': neg_segment,
                    'doc_id': neg_doc_id
                }
                arg_count += len(neg_col)
                replaced_count += to_replace
            if experiment_str in ['typentvs']:
                neg_col = pos_json['col']
                # Replace a fixed number of elements from the column with random elements.
                to_replace = int(len(neg_col)/2 * (float(neg_arg_percent) / 100))
                # If it turns out to be 0 then replace a single argument.
                to_replace = 1 if to_replace == 0 else to_replace
                # Get indices of the types in the list.
                replace_idxs = random.sample(range(0, len(neg_col), 2), to_replace)
                for idx in replace_idxs:
                    ent, etype = get_random_enttype(ent2type_li)[0]
                    neg_col[idx] = etype
                    neg_col[idx+1] = ent
                # Write the example out to a file.
                im_data = {
                    'row': neg_row,
                    'col': neg_col,
                    'segment': neg_segment,
                    'doc_id': neg_doc_id
                }
                arg_count += len(neg_col)/2
                replaced_count += to_replace
            elif experiment_str in ['ltentvs']:
                neg_col = pos_json['col']
                neg_col_types = pos_json['col_types']
                # Replace a fixed number of elements from the column with random elements.
                to_replace = int(len(neg_col) * (float(neg_arg_percent)/100))
                # If it turns out to be 0 then replace a single argument.
                to_replace = 1 if to_replace == 0 else to_replace
                replace_idxs = random.sample(range(len(neg_col)), to_replace)
                for idx in replace_idxs:
                    ent, etype = get_random_enttype(ent2type_li)[0]
                    neg_col[idx] = ent
                    neg_col_types[idx] = etype
                # Write the example out to a file.
                im_data = {
                    'row': neg_row,
                    'col': neg_col,
                    'col_types': neg_col_types,
                    'segment': neg_segment,
                    'doc_id': neg_doc_id
                }
                arg_count += len(neg_col)
                replaced_count += to_replace
            jsons = json.dumps(im_data)
            neg_split_file.write(jsons + '\n')
            example_count += 1
            if example_count % 100000 == 0:
                sys.stdout.write('Processing example: {:d}\n'.format(example_count))
        neg_split_file.close()
        average_args = float(arg_count)/example_count
        average_replaced = float(replaced_count)/example_count
        sys.stdout.write('Average args: {:.4f}; Average replaced: {:.4f}\n'.
                         format(average_args, average_replaced))
        sys.stdout.write('Wrote: {:s}\n'.format(neg_split_file.name))
        sys.stdout.write('Took: {:4.4f}s\n\n'.format(time.time() - start_time))


def make_ranking_ent_examples(in_path, map_path, experiment_str, num_replicas=499):
    """
    For every example make a set of other examples all of which consist of one
    randomly replaced entity.
    :param in_path: dir with example-per-line json files: 'train', 'dev', 'test'.
    :param map_path: dir with the maps going from the entity to the types its
        tagged with.
    :param experiment_str: string; says which experiment you're making the intmaps
        for.
    :param num_replicas: int; says how how many examples with an element replaced
        should be created for every given example.
    :return: None.
    """
    MIN_ARGS = 3
    ent2type = os.path.join(map_path, 'ent2type.json')
    with codecs.open(ent2type, 'r', 'utf-8') as fp:
        ent2type = json.load(fp)
        sys.stdout.write('Read: {}\n'.format(fp.name))
    # Make it into a list so its easier to get random things from it.
    ent2type_li = ent2type.items()

    sys.stdout.write('ent2type: {}\n'.format(len(ent2type_li)))

    labels_li = list(pps.label_set)
    splits = ['test']
    for split in splits:
        # File containing per-line jsons for each positive example. This is shuffled
        # so I get a random set of examples instead of the first n split examples.
        pos_split_fname = os.path.join(in_path, split) + '.json'
        # File to write out.
        ranking_split_fname = os.path.join(in_path, split) + '-{}-ranking.json'.format(num_replicas+1)

        # Open everything up.
        pos_split_file = codecs.open(pos_split_fname, 'r', 'utf-8')
        ranking_split_file = codecs.open(ranking_split_fname, 'w', 'utf-8')
        start_time = time.time()
        sys.stdout.write('Processing: {:s}\n'.format(pos_split_file.name))
        example_count = 0
        arg_count = 0
        for pos_json in data_utils.read_perline_json(pos_split_file):
            # Check to make sure that there are atleast a minimum number of arguments
            # in the example.
            num_arguments = len(pos_json['col'])/2 if experiment_str == 'typentvs' else len(pos_json['col'])
            if num_arguments < MIN_ARGS:
                continue
            arg_count += num_arguments
            if example_count % 50 == 0:
                sys.stdout.write('Processing example: {:d}\n'.format(example_count))
            # Make one argument replaced replicas for 2000 events.
            if example_count == 2000:
                break
            # Write the actual positive example out first.
            jsons = json.dumps(pos_json)
            ranking_split_file.write(jsons + '\n')
            # Next sample a set of random entities some of which are hopefully meaningful for the
            # given event.
            random_ent_type_pairs = get_random_enttype(ent2type_li, num_samples=num_replicas)
            # Decide on which entity to replace.
            if experiment_str in ['typentvs']:
                idx_to_replace = random.choice(range(0, len(pos_json['col']), 2))
            else:
                idx_to_replace = random.choice(range(len(pos_json['col'])))
            if experiment_str == 'typevs':
                for etype in labels_li:
                    # Same as the pos row.
                    rep_doc_id = pos_json['doc_id']
                    rep_row = pos_json['row']
                    rep_col = pos_json['col']
                    # Make the replaced replicas.
                    rep_col[idx_to_replace] = etype + '_netag'
                    # Write the example out to a file.
                    im_data = {
                        'row': rep_row,
                        'col': rep_col,
                        'replaced_idx': idx_to_replace,
                        'doc_id': rep_doc_id,
                    }
                    jsons = json.dumps(im_data)
                    ranking_split_file.write(jsons + '\n')
            else:
                for ent, etype in random_ent_type_pairs:
                    # Same as the pos row.
                    rep_doc_id = pos_json['doc_id']
                    rep_row = pos_json['row']
                    rep_col = pos_json['col']
                    # Make the replaced replicas.
                    if experiment_str in ['typentvs']:
                        rep_col[idx_to_replace] = etype
                        rep_col[idx_to_replace+1] = ent
                        # Write the example out to a file.
                        im_data = {
                            'row': rep_row,
                            'col': rep_col,
                            'replaced_idx': idx_to_replace,
                            'doc_id': rep_doc_id,
                        }
                        jsons = json.dumps(im_data)
                        ranking_split_file.write(jsons + '\n')
                    elif experiment_str in ['ltentvs']:
                        rep_col_types = pos_json['col_types']
                        rep_col[idx_to_replace] = ent
                        rep_col_types[idx_to_replace] = etype
                        # Write the example out to a file.
                        im_data = {
                            'row': rep_row,
                            'col': rep_col,
                            'col_types': rep_col_types,
                            'replaced_idx': idx_to_replace,
                            'doc_id': rep_doc_id,
                        }
                        jsons = json.dumps(im_data)
                        ranking_split_file.write(jsons + '\n')
            example_count += 1
        ranking_split_file.close()
        average_args = float(arg_count) / example_count
        sys.stdout.write('Average args: {:.4f}\n'.format(average_args))
        sys.stdout.write('Wrote: {:s}\n'.format(ranking_split_file.name))
        sys.stdout.write('Took: {:4.4f}s\n\n'.format(time.time() - start_time))


if __name__ == '__main__':
    if sys.argv[1] == 'make_finerneg':
        for percent in [20, 50, 80, 100]:
            for experiment in ['typevs', 'typentvs', 'ltentvs']:
                make_finer_neg(in_path='/iesl/canvas/smysore/material_science_framex/datasets_proc/{:s}'.format(experiment),
                               map_path='/iesl/canvas/smysore/material_science_framex/datasets_proc',
                               experiment_str=experiment, neg_arg_percent=percent)
    elif sys.argv[1] == 'make_rankingent':
        for experiment in ['typevs', 'typentvs', 'ltentvs']:
            make_ranking_ent_examples(in_path='/iesl/canvas/smysore/material_science_framex/datasets_proc/{:s}'.format(experiment),
                                      map_path='/iesl/canvas/smysore/material_science_framex/datasets_proc',
                                      experiment_str=experiment)
    else:
        sys.stderr.write('Unknown action {}.\n'.format(sys.argv[1]))