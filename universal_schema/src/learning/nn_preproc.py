"""
Code to pre-process text data and embeddings to be fed to the model.
- Mainly involves mapping tokens to integers.
- Building maps going from the word to the pre-trained embeddings to the
    embedding for faster access later on. (Only if you have pretrained
    embeddings, of course.)
"""
from __future__ import unicode_literals
import os, sys, argparse
import codecs, json
import time
import re
import itertools
from collections import defaultdict
import random

import data_utils
import le_settings as les

# Write unicode to stdout.
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)


def make_readable_neg(in_path, experiment_str, dataset):
    """
    For the positive and the shuffled positive file passed create a readable
    negative examples file by combining the positive row and the shuffled columns.
    :param in_path: dir with example-per-line json files: 'train', 'dev', 'test'.
    :param experiment_str: string; says which experiment you're making the intmaps
        for.
    :param dataset: string; says which experiment the neg examples are for. sempl18
        requires special treatment. Everything else is treated alike.
    :return: None.
    """
    si_str = 'deps' if dataset == 'anyt' else 'types'
    splits = ['dev', 'test', 'train']
    for split in splits:
        # File containing per-line jsons for each positive example.
        pos_split_fname = os.path.join(in_path, split) + '.json'
        # File containing per-line jsons which have been shuffled but
        # are still all positive.
        shuf_split_fname = os.path.join(in_path, 'neg', split) + '-shuf.json'
        # File to write out.
        neg_split_fname = os.path.join(in_path, split) + '-neg.json'

        # Open everything up.
        # Some datasets (sempl18) will lack either a dev/test file or a corresponding
        # shuffled file. Ignore those.
        try:
            pos_split_file = codecs.open(pos_split_fname, 'r', 'utf-8')
            shuf_split_file = codecs.open(shuf_split_fname, 'r', 'utf-8')
        except IOError as e:
            sys.stdout.write('{}\n'.format(str(e)))
            continue
        # In the case of train-neg files for the anyt data and rnentvs model
        # the train-neg file should be appended to instead of being written to
        # from scratch.
        neg_split_file = codecs.open(neg_split_fname, 'a+', 'utf-8')
        start_time = time.time()
        sys.stdout.write('Processing: {:s}\n'.format(pos_split_file.name))
        example_count = 0
        for pos_json, shuf_json in itertools.izip(data_utils.read_json(pos_split_file),
                                                  data_utils.read_json(shuf_split_file)):
            # Ignore if the pos example is from sempl18, since these negative examples
            # have already been created. Some of the negative examples used once in sempl18
            # will get reused.
            if dataset != 'sempl18' and pos_json['doc_id'][0:7] == 'sempl18':
                continue
            # Same as the pos row.
            pos_doc_id = pos_json['doc_id']
            pos_row = pos_json['row']
            neg_col = shuf_json['col']
            if experiment_str in ['naryus', 'typevs', 'typentvs', 'rnentvs', 'rientvs']:
                # Write the example out to a file.
                im_data = {
                    'row': pos_row,
                    'col': neg_col,
                    'doc_id': pos_doc_id
                }
            elif experiment_str in ['rnentdepvs']:
                neg_colsi = shuf_json['col_{:s}'.format(si_str)]
                # Write the example out to a file.
                im_data = {
                    'row': pos_row,
                    'col': neg_col,
                    'col_{:s}'.format(si_str): neg_colsi,
                    'doc_id': pos_doc_id
                }
            elif experiment_str in ['rnentsentvs']:
                neg_sentcon = shuf_json['col_sentcon']
                # Write the example out to a file.
                im_data = {
                    'row': pos_row,
                    'col': neg_col,
                    'col_sentcon': neg_sentcon,
                    'doc_id': pos_doc_id
                }
            elif experiment_str in ['rnentsisentvs']:
                neg_colsi = shuf_json['col_{:s}'.format(si_str)]
                neg_sentcon = shuf_json['col_sentcon']
                # Write the example out to a file.
                im_data = {
                    'row': pos_row,
                    'col': neg_col,
                    'col_{:s}'.format(si_str): neg_colsi,
                    'col_sentcon': neg_sentcon,
                    'doc_id': pos_doc_id
                }
            elif experiment_str in ['ltentvs', 'dsentvs', 'dsentvsgro']:
                neg_colsi = shuf_json['col_{:s}'.format(si_str)]
                neg_col_ids = shuf_json['col_ids']
                neg_row_id = shuf_json['row_id']
                # Write the example out to a file.
                im_data = {
                    'row': pos_row,
                    'row_id': neg_row_id,
                    'col': neg_col,
                    'col_ids': neg_col_ids,
                    'col_{:s}'.format(si_str): neg_colsi,
                    'doc_id': pos_doc_id
                }
            jsons = json.dumps(im_data)
            neg_split_file.write(jsons + '\n')
            example_count += 1
            if example_count % 10000 == 0:
                sys.stdout.write('Processing example: {:d}\n'.format(example_count))
        neg_split_file.close()
        sys.stdout.write('Wrote: {:s}\n'.format(neg_split_file.name))
        sys.stdout.write('Took: {:4.4f}s\n\n'.format(time.time() - start_time))


def map_split_to_int(split_file, intmapped_out_file, op2idx, ent2idx, size_str,
                     experiment_str, si_str=None, sentword2idx=None, sideinfo2idx=None,
                     update_map=True):
    """
    Convert text to set of int mapped tokens. Mapping words to integers at
    all times, train/dev/test. msall tokens and ents not lowercased.
    :param split_file: file; an open file to read from.
    :param intmapped_out_file: file; the file to write int-mapped per
        line jsons for each example to.
    :param sideinfo2idx: dict; mapping every col side info {deps/types} to an integer.
    :param ent2idx: dict; mapping ents/col elements to integers.
    :param op2idx: dict; mapping ops/rows to integers.
    :param size_str: string [small/full]; says if you want to make a
        small example set or full.
    :param sentword2idx: dict; mapping col sentcon tokens to ints
    :param experiment_str: string; says which experiment you're making the intmaps
        for.
    :param si_str: string; says which side info should get read. {deps/types}
    :param update_map: bool; if the word2idx map should be updated. To control
        sharing of embeddings between different splits.
    :return: word2idx: dict(str:int); maps every token to an integer.
    """
    if experiment_str in ['rnentdepvs', 'rnentsisentvs',
                          'ltentvs', 'dsentvs', 'dsentrivs'] and si_str == None:
        raise ValueError('Need to specify si_str.')
    row_oovs, col_oovs = set(), set()
    split_docs = defaultdict(int)
    num_examples = 0
    # reserve some indices for special tokens.
    ent2idx['<pad>'], op2idx['<pad>'], sideinfo2idx['<pad>'] = les.PAD, les.PAD, les.PAD
    ent2idx['<oov>'], op2idx['<oov>'], sideinfo2idx['<oov>'] = les.OOV, les.OOV, les.OOV
    ent2idx['<start>'], op2idx['<start>'], sideinfo2idx['<start>'] = les.START, les.START, les.START
    ent2idx['<stop>'], op2idx['<stop>'], sideinfo2idx['<stop>'] = les.STOP, les.STOP, les.STOP
    sentword2idx['<stop>'], sentword2idx['<start>'], sentword2idx['<pad>'], sentword2idx['<oov>'] = \
        les.STOP, les.START, les.PAD, les.OOV

    start_time = time.time()
    sys.stdout.write('Processing: {:s}\n'.format(split_file.name))
    for example_json in data_utils.read_json(split_file):
        # Read the list 'things' in the row, mostly a single operation;
        # But it could be a list of (possibly multi-word) entities.
        # Depending on the experiment.
        example_row = example_json['row']
        # Read the column which has the types/type+ents/skeletons.
        example_col = example_json['col']
        if experiment_str in ['rnentdepvs', 'rnentsisentvs', 'ltentvs', 'dsentvs',
                              'dsentrivs', 'dsentvsgro']:
            example_col_sideinfo = example_json['col_{:s}'.format(si_str)]
        if experiment_str in ['rnentsentvs', 'rnentsisentvs']:
            # List of strings now.
            example_col_sentcons = example_json['col_sentcon']
            # Go over list of strings and tokenize them.
            temp = []
            for pair_sentcon in example_col_sentcons:
                # Replace <*_ptag> with <*ptag> I made the terrible
                # call to use underscores inside these placeholders.
                pair_sentcon = re.sub('\<predicate\_ptag\>', '<predicateptag>', pair_sentcon)
                pair_sentcon = re.sub('\<argument\_ptag\>', '<argumentptag>', pair_sentcon)
                # Tokenize the sentence context. Not adding starts and stops to the
                # tokenized contexts.
                tokenized = pair_sentcon.split('_')
                temp.append(tokenized)
            example_col_sentcons = temp

        # Keep track of the doc id to help look at look at the original text.
        doc_id = example_json['doc_id']
        # Some of the dois have underscores in them even aside from the one I
        # inserted in the end. So rebuild the correct doi.
        doi_str = '_'.join(doc_id.split('_')[:-1])
        split_docs[doi_str] += 1  # Keep track of the number of documents.
        num_examples += 1
        if num_examples % 10000 == 0:
            sys.stdout.write('Processing the {:d}th example\n'.format(num_examples))
        # Make a small dataset if asked for.
        if size_str == 'small' and num_examples == 20000:
            break
        # Add start and stop states but the batcher ignores the symbols in the case of
        # the verb schema models. (Only need this when im doing the language modeling
        # kind of thing)
        if experiment_str == 'naruys':
            example_row = ['<start>'] + example_row + ['<stop>']
            example_col = ['<start>'] + example_col + ['<stop>']
        else:
            example_col = ['<start>'] + example_col + ['<stop>']
            if experiment_str in ['rnentdepvs', 'rnentsisentvs', 'ltentvs', 'dsentvs', 'dsentrivs',
                                  'dsentvsgro']:
                example_col_sideinfo = ['<start>'] + example_col_sideinfo + ['<stop>']
        # Update the token-to-int map.
        if update_map:
            for row_tok in example_row:
                if row_tok not in op2idx:
                    op2idx[row_tok] = len(op2idx)
            for col_tok in example_col:
                if col_tok not in ent2idx:
                    ent2idx[col_tok] = len(ent2idx)
            if experiment_str in ['rnentdepvs', 'rnentsisentvs', 'ltentvs', 'dsentvs',
                                  'dsentrivs', 'dsentvsgro']:
                for col_si_tok in example_col_sideinfo:
                    if col_si_tok not in sideinfo2idx:
                        sideinfo2idx[col_si_tok] = len(sideinfo2idx)
            if experiment_str in ['rnentsentvs', 'rnentsisentvs']:
                # This is one sentence.
                for col_sentcon in example_col_sentcons:
                    for con_token in col_sentcon:
                        if con_token not in sentword2idx:
                            sentword2idx[con_token] = len(sentword2idx)
        # Map row and columns to integers.
        intmapped_row = []
        for row_tok in example_row:
            # This case cant happen for me because im updating the map
            # for every split. But in case I set update_map to false
            # this handles it.
            intmapped_tok = op2idx.get(row_tok, op2idx['<oov>'])
            intmapped_row.append(intmapped_tok)
            if intmapped_tok == op2idx['<oov>']:
                row_oovs.add(row_tok)
        intmapped_col = []
        for col_tok in example_col:
            intmapped_tok = ent2idx.get(col_tok, ent2idx['<oov>'])
            intmapped_col.append(intmapped_tok)
            if intmapped_tok == ent2idx['<oov>']:
                col_oovs.add(col_tok)
        intmapped_col_si = []
        if experiment_str in ['rnentdepvs', 'rnentsisentvs', 'ltentvs', 'dsentvs',
                              'dsentrivs', 'dsentvsgro']:
            for col_si_tok in example_col_sideinfo:
                intmapped_tok = sideinfo2idx.get(col_si_tok, sideinfo2idx['<oov>'])
                intmapped_col_si.append(intmapped_tok)
        intmapped_col_sentcons = []
        if experiment_str in ['rnentsentvs', 'rnentsisentvs']:
            for pair_sentcon in example_col_sentcons:
                intmapped_pair_sentcon = []
                for token in pair_sentcon:
                    if update_map:
                        itok = sentword2idx[token]
                    # If update_map was set to false then place oov.x
                    else:
                        itok = sentword2idx.get(token, sentword2idx['<oov>'])
                    intmapped_pair_sentcon.append(itok)
                intmapped_col_sentcons.append(intmapped_pair_sentcon)

        # Write the example out to a file.
        if experiment_str in ['rnentdepvs', 'ltentvs', 'dsentvs', 'dsentrivs',
                              'dsentvsgro']:
            row_id = example_json['row_id']
            col_ids = row_id + example_json['col_ids'] + row_id
            im_data = {
                'row': intmapped_row,
                'row_id': row_id,
                'col': intmapped_col,
                'col_ids': col_ids,
                'col_{:s}'.format(si_str): intmapped_col_si,
                'doc_id': doc_id
            }
        elif experiment_str == 'rnentsentvs':
            im_data = {
                'row': intmapped_row,
                'col': intmapped_col,
                'col_sentcon': intmapped_col_sentcons,
                'doc_id': doc_id
            }
        elif experiment_str == 'rnentsisentvs':
            im_data = {
                'row': intmapped_row,
                'col': intmapped_col,
                'col_{:s}'.format(si_str): intmapped_col_si,
                'col_sentcon': intmapped_col_sentcons,
                'doc_id': doc_id
            }
        else:
            im_data = {
                'row': intmapped_row,
                'col': intmapped_col,
                'doc_id': doc_id
            }
        jsons = json.dumps(im_data)
        intmapped_out_file.write(jsons+'\n')

    intmapped_out_file.close()
    av_seq_len = float(sum(split_docs.values()))/len(split_docs)
    sys.stdout.write('num_event_chains: {:d}; num_events: {:d}; av_chain_len: {:0.4f}\n'.
                     format(len(split_docs), num_examples, av_seq_len))
    sys.stdout.write('total_rc-vocab_size: {:d}; row_vocab_size: {:d}; col_vocab_size: {:d} '
                     'num_row_oovs: {:d}; num_col_oovs: {:d}\n'.
                     format(len(op2idx)+len(ent2idx), len(op2idx), len(ent2idx),
                            len(row_oovs), len(col_oovs)))
    sys.stdout.write('Took: {:4.4f}s\n'.format(time.time()-start_time))
    return op2idx, ent2idx, sideinfo2idx, sentword2idx


def make_int_maps(in_path, out_path, experiment_str, size_str, dataset):
    """
    For each split map all the row and column entities/types/tokens to integers.
    :param in_path: dir with example-per-line json files: 'train', 'dev', 'test'.
    :param out_path: path to which int mapped files should get written.
    :param experiment_str: string; says which experiment you're making the intmaps
        for.
    :param dataset: string; says which dataset the int maps are for. {anyt/ms500k}
    :param size_str: says if you want to make a small example set or full. This
        is unnecessary but removing it requires changes to the predict utils
        and the batchers. And anyone else reading int mapped files.
    :return: None.
    TODO: Side info maps werent saved for the em models so this needs to be re-run
    for that data. Also eval data doesnt have si, look into that. --med-pri.
    """
    si_str = 'deps' if dataset == 'anyt' else 'types'
    # Try to read maps in if they already exist on disk. For when files not
    # converted to ints earlier have to be converted to ints after the whole
    # other set of files.
    try:
        intmap_path = os.path.join(out_path, 'op2idx-{:s}.json'.format(size_str))
        with codecs.open(intmap_path, 'r', 'utf-8') as fp:
            op2idx = json.load(fp)
            sys.stdout.write('Read: {:s}\n'.format(fp.name))
    except IOError:
        op2idx = {}
    try:
        intmap_path = os.path.join(out_path, 'ent2idx-{:s}.json'.format(size_str))
        with codecs.open(intmap_path, 'r', 'utf-8') as fp:
            ent2idx = json.load(fp)
            sys.stdout.write('Read: {:s}\n'.format(fp.name))
    except IOError:
        ent2idx = {}
    try:
        if si_str == 'deps':
            intmap_path = os.path.join(out_path, 'deps2idx-{:s}.json'.format(size_str))
        else:
            intmap_path = os.path.join(out_path, 'type2idx-{:s}.json'.format(size_str))
        with codecs.open(intmap_path, 'r', 'utf-8') as fp:
            sideinfo2idx = json.load(fp)
            sys.stdout.write('Read: {:s}\n'.format(fp.name))
    except IOError:
        sideinfo2idx = {}
    try:
        intmap_path = os.path.join(out_path, 'sentword2idx-{:s}.json'.format(size_str))
        with codecs.open(intmap_path, 'r', 'utf-8') as fp:
            sentword2idx = json.load(fp)
            sys.stdout.write('Read: {:s}\n'.format(fp.name))
    except IOError:
        sentword2idx = {}
    # Just list all files to map to ints and map what ever is present.
    # Do the dev and test sets before the train so when you measure oovs wrt
    # of those wrt to the train you know how many arent in the train set.
    if dataset == 'anyt':
        if experiment_str in ['emdsentvs', 'emdsentvssup', 'emdsentvsdum', 'emdsentposvsdum']:
            splits = ['ep-test-emex', 'ep-dev-emex',
                      'ep-test-pdepemex', 'ep-dev-pdepemex',
                      'ep-test-emex-fcw', 'ep-dev-emex-fcw',
                      'ep-test-pdepemex-fcw', 'ep-dev-pdepemex-fcw',
                      'dev', 'test', 'train',
                      # These will just be ignored for emdsentvs and emdsentvsdum.
                      'dev-neg', 'test-neg', 'train-neg']
        elif experiment_str in ['dsentvsgro']:
            splits = ['ep-test-grocandidates', 'ep-dev-grocandidates',
                      'ep-test-grocandidates-fcw', 'ep-dev-grocandidates-fcw',
                      'ep-test-candidates', 'ep-dev-candidates',
                      'ep-test-pdepcandidates', 'ep-dev-pdepcandidates',
                      'ep-test-candidates-fcw', 'ep-dev-candidates-fcw',
                      'ep-test-pdepcandidates-fcw', 'ep-dev-pdepcandidates-fcw',
                      'dev', 'test', 'train',
                      'dev-neg', 'test-neg', 'train-neg']
        else:
            splits = ['ep-test-llpdepcandidates', 'ep-dev-llpdepcandidates',
                      'ep-test-llpdepcandidates-fcw', 'ep-dev-llpdepcandidates-fcw',
                      'ep-test-candidates', 'ep-dev-candidates',
                      'ep-test-pdepcandidates', 'ep-dev-pdepcandidates',
                      'ep-test-candidates-fcw', 'ep-dev-candidates-fcw',
                      'ep-test-pdepcandidates-fcw', 'ep-dev-pdepcandidates-fcw',
                      'ri-dev-gold', 'ri-test-gold',
                      'ri-dev-gold-fcw', 'ri-test-gold-fcw',
                      'dev', 'test', 'train',
                      'dev-neg', 'test-neg', 'train-neg']
    elif dataset == 'ms500k':
        if experiment_str in ['emdsentvs', 'emdsentvsdum', 'emdsentposvsdum']:
            splits = ['ep-test-emex', 'ep-dev-emex',
                      'dev', 'test', 'train']
        elif experiment_str in ['emdsentvssup']:
            splits = ['ep-test-emex', 'ep-dev-emex',
                      'dev', 'test', 'train',
                      'test-neg', 'dev-neg', 'train-neg']
        else:
            splits = ['ep-test-candidates', 'ep-dev-candidates',
                      'ri-dev-gold', 'ri-test-gold',
                      'dev', 'test', 'train',
                      'dev-neg', 'test-neg', 'train-neg']
    else:
        raise ValueError('Unknown dataset: {:s}'.format(dataset))
    # Manually setting this to false when I want to update only specific splits.
    # TODO: Fix the above. --med-pri.
    update_map = True
    for split in splits:
        split_fname = os.path.join(in_path, split) + '.json'
        # Try to open the split file, if it doesnt exist move to the next split.
        # Different experiments and datasets will contain different kinds of
        # splits.
        try:
            split_file = codecs.open(split_fname, 'r', 'utf-8')
        except IOError as e:
            sys.stderr.write('{}\n'.format(str(e)))
            continue
        intmapped_out_fname = os.path.join(out_path, split) + '-im-{:s}.json'.format(size_str)
        intmapped_out_file = codecs.open(intmapped_out_fname, 'w', 'utf-8')
        # Choose to use a different int mapping function for the role negatives data which
        # have a different structure than all the other negative example files.
        if experiment_str == 'dsentrivs' and 'neg' in split:
            op2idx, ent2idx, sideinfo2idx, sentword2idx = map_split_to_int_ri_neg(
                split_file, intmapped_out_file, si_str=si_str, op2idx=op2idx, ent2idx=ent2idx,
                sideinfo2idx=sideinfo2idx, sentword2idx=sentword2idx,
                experiment_str=experiment_str, size_str=size_str, update_map=update_map)
        elif experiment_str in ['emdsentvs', 'emdsentvssup', 'emdsentvsdum', 'emdsentposvsdum']:
            op2idx, ent2idx, sideinfo2idx = map_em_split_to_int(
                split_file, intmapped_out_file, si_str=si_str, op2idx=op2idx, ent2idx=ent2idx,
                sideinfo2idx=sideinfo2idx, experiment_str=experiment_str, size_str=size_str,
                update_map=update_map)
        else:
            op2idx, ent2idx, sideinfo2idx, sentword2idx = map_split_to_int(
                split_file, intmapped_out_file, si_str=si_str, op2idx=op2idx, ent2idx=ent2idx,
                sideinfo2idx=sideinfo2idx, sentword2idx=sentword2idx,
                experiment_str=experiment_str, size_str=size_str, update_map=update_map)

        sys.stdout.write('Wrote: {:s}\n'.format(intmapped_out_fname))
        sys.stdout.write('\n')
    if update_map:
        # Write the maps.
        intmap_out_path = os.path.join(out_path, 'op2idx-{:s}.json'.format(size_str))
        with codecs.open(intmap_out_path, 'w', 'utf-8') as fp:
            json.dump(op2idx, fp, indent=2)
        sys.stdout.write('Wrote: {:s}\n'.format(intmap_out_path))

        intmap_out_path = os.path.join(out_path, 'ent2idx-{:s}.json'.format(size_str))
        with codecs.open(intmap_out_path, 'w', 'utf-8') as fp:
            json.dump(ent2idx, fp, indent=2)
        sys.stdout.write('Wrote: {:s}\n'.format(intmap_out_path))

        if experiment_str in ['rnentdepvs', 'rnentsisentvs', 'ltentvs', 'dsentvs', 'dsentrivs',
                              'emdsentvs', 'emdsentvssup', 'emdsentvsdum', 'dsentvsgro']:
            if si_str == 'deps':
                intmap_out_path = os.path.join(out_path, 'deps2idx-{:s}.json'.format(size_str))
            else:
                intmap_out_path = os.path.join(out_path, 'type2idx-{:s}.json'.format(size_str))
            with codecs.open(intmap_out_path, 'w', 'utf-8') as fp:
                json.dump(sideinfo2idx, fp, indent=2, sort_keys=True)
            sys.stdout.write('Wrote: {:s}\n'.format(intmap_out_path))
        if experiment_str in ['rnentsentvs', 'rnentsisentvs']:
            intmap_out_path = os.path.join(out_path, 'sentword2idx-{:s}.json'.format(size_str))
            with codecs.open(intmap_out_path, 'w', 'utf-8') as fp:
                json.dump(sentword2idx, fp, indent=2, sort_keys=True)
            sys.stdout.write('Wrote: {:s}\n'.format(intmap_out_path))


def main():
    """
    Parse command line arguments and call all the above routines.
    :return:
    """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='subcommand',
                                       help='The action to perform.')

    # Make readable negative examples.
    readable_neg = subparsers.add_parser('readable_neg')
    readable_neg.add_argument('-i', '--in_path', required=True,
                              help='Path to the processed train/dev/test '
                                   'splits.')
    readable_neg.add_argument('-e', '--experiment', required=True,
                              choices=['naryus', 'typevs', 'typentvs', 'ltentvs',
                                       'scriptlm', 'rnentvs', 'rnentdepvs', 'rnentsentvs',
                                       'rnentsisentvs', 'rientvs', 'dsentvs', 'emdsentvssup',
                                       'dsentvsgro'],
                              help='Name of the experiment running.')
    readable_neg.add_argument('-d', '--dataset', required=True,
                              choices=['sempl18', 'anyt', 'ms500k', 'conll2012wsj',
                                       'conll2009en'],
                              help='Name of the dataset for which neg examples are being created.')

    # Map sentences to list of int mapped tokens.
    make_int_map = subparsers.add_parser('int_map')
    make_int_map.add_argument('-i', '--in_path', required=True,
                              help='Path to the processed train/dev/test '
                                   'splits.')
    make_int_map.add_argument('-e', '--experiment', required=True,
                              choices=['naryus', 'typevs', 'typentvs', 'ltentvs',
                                       'scriptlm', 'rnentvs', 'rnentdepvs', 'rnentsentvs',
                                       'rnentsisentvs', 'dsentvs', 'dsentrivs', 'emdsentvs',
                                       'emdsentvssup', 'emdsentvsdum', 'emdsentposvsdum',
                                       'dsentvsgro'],
                              help='Name of the experiment running.')
    make_int_map.add_argument('-s', '--size', required=True,
                              choices=['small', 'full'],
                              help='Make a small version with 20000 examples or'
                                   'a large version with the whole dataset.')
    make_int_map.add_argument('-d', '--dataset', required=True,
                              choices=['sempl18', 'anyt', 'ms500k', 'conll2012wsj',
                                       'conll2009en'],
                              help='Name of the dataset for which neg examples are being created.')

    cl_args = parser.parse_args()

    if cl_args.subcommand == 'int_map':
        make_int_maps(in_path=cl_args.in_path, out_path=cl_args.in_path,
                      experiment_str=cl_args.experiment, size_str=cl_args.size,
                      dataset=cl_args.dataset)
    elif cl_args.subcommand == 'readable_neg':
        make_readable_neg(in_path=cl_args.in_path, experiment_str=cl_args.experiment,
                          dataset=cl_args.dataset)
    else:
        raise ValueError('Unknown action: {:s}'.format(cl_args.subcommand))


if __name__ == '__main__':
    main()
