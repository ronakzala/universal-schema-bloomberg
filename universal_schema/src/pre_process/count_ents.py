"""
Build a dictionary which counts the number of occurrences of operations and use
this to filter infrequent operations.
"""
from __future__ import unicode_literals
from __future__ import print_function
import os, sys
import codecs, json
from nltk.stem.wordnet import WordNetLemmatizer
import time
from collections import defaultdict

import data_utils as du
import pp_settings as pps

sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
lemmatizer = WordNetLemmatizer()


def count_ops(in_file, count_dict, lemma_dict):
    """
    Open file and add count of operations in the file.
    :param in_file:
    :param count_dict:
    :param lemma_dict:
    :return:
    """
    with codecs.open(in_file, 'r', 'utf-8') as fp:
        paper_dict = json.load(fp, encoding='utf-8')
    par_dicts = paper_dict['paragraphs']
    for par_dict in par_dicts:
        # Get tokens and labels of all sentences in par.
        sents_toks = par_dict['proc_tokens']
        sents_labs = par_dict['proc_labels_cnnner']
        assert (len(sents_toks) == len(sents_labs))
        # Go over each sentence and form counts.
        for sent_toks, sent_labs in zip(sents_toks, sents_labs):
            assert (len(sent_toks) == len(sent_labs))
            # Fix the predicted labels for obvious mistakes. Fix stray I-tags without B-tags.
            sent_labs = du.fix_bio_labels(sent_labs)
            # Get the operations in the sentence.
            sent_ops = []
            in_op = False
            for tok, lab in zip(sent_toks, sent_labs):
                # You saw a beginning, so add ent to the ents and place label into
                # sentence.
                if lab == 'B-Operation':
                    in_op = True
                    sent_ops.append(tok)
                # You see a "in", if its the same current label concat it with the B ent
                # token and skip any addition to the sentence.
                elif lab == 'I-Operation' and in_op:
                    sent_ops[-1] = (sent_ops[-1] + ' ' + tok).strip()
            # Update the count_dict.
            for op in sent_ops:
                # Cache the lemmatized operation so you dont have to call expensive
                # wordnet every time.
                try:
                    lemmatized_op = lemma_dict[op]
                except KeyError:
                    # Say every operation is a verb.
                    lemmatized_op = lemmatizer.lemmatize(op, pos='v')
                    lemma_dict[op] = lemmatized_op
                # Count the lemmatized operation.
                try:
                    count_dict[lemmatized_op] += 1
                except KeyError:
                    count_dict[lemmatized_op] = 1


def count_ents(in_fname, count_dict):
    """
    Open file and add count of entities in the file.
    :param in_fname: string; full path to the document to read.
    :param count_dict: dict(string:int); dict counting the entities.
    :return: None.
        Modify count_dict in place.
    """
    with codecs.open(in_fname, 'r', 'utf-8') as fp:
        paper_dict = json.load(fp, encoding='utf-8')
    par_dicts = paper_dict['paragraphs']
    for par_dict in par_dicts:
        # Get tokens and labels of all sentences in par.
        sents_toks = par_dict['proc_tokens']
        sents_labs = par_dict['proc_labels_cnnner']
        assert (len(sents_toks) == len(sents_labs))
        # Go over each sentence and form counts.
        for sent_toks, sent_labs in zip(sents_toks, sents_labs):
            assert (len(sent_toks) == len(sent_labs))
            # Fix the predicted labels for obvious mistakes. Fix stray I-tags without B-tags.
            sent_labs = du.fix_bio_labels(sent_labs)
            # Go over the labels and mark everything we dont care about to "O".
            filt_labs = []
            for lab in sent_labs:
                cur_lab = lab[2:]
                # If its in the argument label set we care save it.
                if cur_lab in pps.maanns_arg_label_set:
                    filt_labs.append(lab)
                else:
                    filt_labs.append("O")
            # Get the entities in the sentence.
            cur_lab = ''
            seg_ents = []
            for tok, lab in zip(sent_toks, filt_labs):
                if lab == 'O':
                    continue
                # You saw a beginning, so add ent to the sent and place label into
                # labels.
                elif lab[0] == 'B':
                    cur_lab = lab[2:]
                    seg_ents.append(tok)
                # You see a "in", if its the same current label concat it with the B ent
                # token in the sentence and skip any addition to labs.
                elif lab[0] == 'I' and lab[2:] == cur_lab:
                    seg_ents[-1] = (seg_ents[-1] + ' ' + tok).strip()

            # Update the count_dict.
            for ent in seg_ents:
                try:
                    count_dict[ent] += 1
                except KeyError:
                    count_dict[ent] = 1


def form_count_dict(in_dir, in_doi_file, etype='op', out_dir=None):
    """
    Go over the input directory and
    :param in_dir: input directory with json files.
    :param out_dir: output directory to write counts json to.
    :param etype: what to count. operations or all other entities.
    :param in_doi_file: text file with dois of papers in in_dir to process.
    :return: None.
    """
    count_dict = {}
    operation2lemma = {}
    # Input dois.
    doi_file = codecs.open(in_doi_file, 'r', 'utf-8')
    # Go over the files in a directory and count ops.
    di = du.DirIterator(in_path=os.path.join(in_dir, 'data'), doi_file=doi_file, safe_dois=True)
    paper_count = 0

    start = time.time()
    for in_paper in di:
        paper_count += 1
        if paper_count % 100 == 0:
            sys.stdout.write('Processing: {:d}; {:s} count: {:d}\n'.
                             format(paper_count, etype, len(count_dict)))
        # Either count operations or all other entities.
        if etype == 'op':
            count_ops(in_file=in_paper, count_dict=count_dict,
                      lemma_dict=operation2lemma)
        elif etype == 'ent':
            count_ents(in_fname=in_paper, count_dict=count_dict)
        else:
            pass

    # Print the counts.
    sortedopfname = os.path.join(out_dir, 'lemma{:s}2count-sorted.txt'.format(etype))
    with codecs.open(sortedopfname, 'w', 'utf-8') as fp:
        du.print_sorted_dict(count_dict, fp)
        sys.stdout.write('Wrote: {:s}\n'.format(fp.name))

    # Write the counts and the lemma dict to disk.
    op2count_out_fname = os.path.join(out_dir, 'lemma{:s}2count.json'.format(etype))
    du.write_json(data_dict=count_dict, out_fname=op2count_out_fname)
    sys.stdout.write('Wrote: {:s}\n'.format(op2count_out_fname))

    if etype == 'op':
        op2lemma_out_fname = os.path.join(out_dir, 'unlemma{:s}2lemma.json'.format(etype))
        du.write_json(data_dict=operation2lemma, out_fname=op2lemma_out_fname,
                      save_space=False)
        sys.stdout.write('Wrote: {:s}\n'.format(op2lemma_out_fname))

    doi_file.close()
    end = time.time()
    if etype == 'op':
        sys.stdout.write('Un-lemmatized entities: {:d}\n'.format(len(operation2lemma)))
    sys.stdout.write('Entities of etype: {:s}: {:d}\n'.format(etype, len(count_dict)))
    sys.stdout.write('Took: {:.4f}s\n'.format(end-start))


if __name__ == '__main__':
    type = sys.argv[1]
    if type in ['op', 'ent']:
        # This in_path has a subdir "data" with actual data in it.
        in_path = '/iesl/canvas/smysore/material_science_framex/datasets_raw/' \
                  'predsynth-ner_parsed_papers-framex-filtered'
        in_doi_file = '/iesl/canvas/smysore/material_science_framex/datasets_raw/' \
                      'predsynth-ner_parsed_papers-framex-filtered/' \
                      'predsynth-ner_parsed_papers-framex-filtered-fnames.txt'
        out_path = '/iesl/canvas/smysore/material_science_framex/datasets_raw/' \
                   'predsynth-ner_parsed_papers-framex-filtered'
        form_count_dict(in_path, in_doi_file, etype=type, out_dir=out_path)
    else:
        sys.stderr.write('Invalid type count requested.')
