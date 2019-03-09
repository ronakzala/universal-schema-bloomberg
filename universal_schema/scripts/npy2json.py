"""
Convert a passed numpy array to a json file. Using this to write a json file
of the embeddings out. This is what the pretrained initialization code expects.
"""
from __future__ import unicode_literals
import os
import sys
import numpy as np
import codecs
import json


def npy2json(word2idx_fname, in_fname, out_fname):
    """
    Create a word2emb file.
    :param word2idx_fname: map from string to integer.
    :param in_fname: numpy array of embeddings.
    :param out_fname: json file to write word2emb file.
    :return: None.
    """
    with codecs.open(word2idx_fname, 'r', 'utf-8') as fp:
        word2idx = json.load(fp)
        idx2word = dict([(val, key) for key, val in word2idx.items()])
    embeddings = np.load(in_fname)
    # vocab x embedding dimension.
    idxs = embeddings.shape[0]
    embedding_json = {}
    for i in xrange(idxs):
        embedding_json[idx2word[i]] = map(float, list(embeddings[i, :]))

    with codecs.open(out_fname, 'w', 'utf-8') as fp:
        json.dump(embedding_json, fp, sort_keys=True)
        sys.stdout.write('Wrote: {:s}\n'.format(fp.name))


def embedding2json(in_fname, out_fname):
    """
    Create a word2emb file.
    :param in_fname: text file with embeddings.
    :param out_fname: json file to write word2emb file.
    :return: None.
    """
    embperline_file = codecs.open(in_fname, 'r', 'utf-8')
    embedding2word = dict()
    count = 0
    for empline in embperline_file:
        empline = empline.strip()
        t = empline.split()
        word = t[0]
        w_embedding = t[1:]
        embedding2word[word] = map(float, w_embedding)
        if count % 1000 == 0:
           print('Processing word {:d}'.format(count))
        count += 1

    with codecs.open(out_fname, 'w', 'utf-8') as fp:
        json.dump(embedding2word, fp, sort_keys=True)
        sys.stdout.write('Wrote: {:s}\n'.format(fp.name))


def msfasttext2json(in_fname, out_fname):
    """
    Read pretrained gensim format fasttext embeddings for materials science and write
    json with the pretrained embeddings to use in model classes.
    :param in_fname: gensim format fasttext embeddings for materials science.
    :param out_fname: json file to write out.
    :return: None.
    """
    from gensim.models.keyedvectors import KeyedVectors

    embeddings = KeyedVectors.load(in_fname)
    embeddings.bucket = 2000000
    print('Read: {:s}'.format(in_fname))
    print('Vocab size: {:d}'.format(len(embeddings.wv.index2word)))

    count = 0
    word2embedding = {}
    for word in embeddings.wv.index2word:
        word2embedding[word] = map(float, list(embeddings[word]))
        count += 1
        if count % 1000 == 0:
            print('Processing word: {:d}'.format(count))

    with codecs.open(out_fname, 'w', 'utf-8') as fp:
        json.dump(word2embedding, fp)
        sys.stdout.write('Wrote: {:s}\n'.format(fp.name))


def nwfasttext2json(in_fname, out_fname):
    """
    Read pretrained text file with fasttext embeddings for general english text
    and write json with the pretrained embeddings to use in model classes.
    :param in_fname: gensim format fasttext embeddings for english.
    :param out_fname: json file to write out.
    :return: None.
    """
    import io

    fin = io.open(in_fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    vocab_size, emb_dim = map(int, fin.readline().split())
    print('Read: {:s}'.format(in_fname))
    print('Vocab size: {:d}'.format(vocab_size))
    word2embedding = {}
    count = 0
    for line in fin:
        token = line.rstrip().split(' ')
        token_str = token[0].lower()
        # The tokens in the file are sorted by frequency and arent lower cased.
        # If the token being read is already in the map then move on. This way the
        # embedding saved it the one for the more frequent of the lower or
        # uppercased tokens.
        if token_str not in word2embedding:
            word2embedding[token_str] = map(float, token[1:])
        else:
            continue
        count += 1
        if count % 1000 == 0:
            print('Processing word: {:d}'.format(count))

    with codecs.open(out_fname, 'w', 'utf-8') as fp:
        json.dump(word2embedding, fp)
        sys.stdout.write('Wrote: {:s}\n Lower-cased size: {:d}\n'.
                         format(fp.name, len(word2embedding)))


if __name__ == '__main__':
    if sys.argv[1] == 'npy2json':
        word2idx = '/iesl/canvas/smysore/material_science_framex/datasets_proc/mementvs/op2idx-full.json'
        in_fname = '/iesl/canvas/smysore/material_science_framex/model_runs/train_model-mementvs-2018_05_18-15_52_43-5m-add/learnt_row_embeddings.npy'
        out_fname = '/iesl/canvas/smysore/material_science_framex/datasets_proc/embeddings/50d.embed.json'
        npy2json(word2idx, in_fname, out_fname)
    elif sys.argv[1] == 'glove2json':
        in_fname = '/iesl/canvas/smysore/material_science_framex/datasets_proc/anyt/embeddings/pretrained_glove/glove.6B.50d.txt'
        out_fname = '/iesl/canvas/smysore/material_science_framex/datasets_proc/anyt/embeddings/pretrained_glove/glove.6B.50d.json'
        embedding2json(in_fname, out_fname)
    elif sys.argv[1] == 'msft2json':
        gensim_embeds = '/iesl/canvas/smysore/material_science_framex/datasets_proc/ms500k' \
                        '/embeddings/fasttext-march2018/fasttext_embeddings-MINIFIED.model'
        out_fname = '/iesl/canvas/smysore/material_science_framex/datasets_proc/ms500k' \
                    '/embeddings/fasttext-march2018/100d.embed.json'
        msfasttext2json(in_fname=gensim_embeds, out_fname=out_fname)
    elif sys.argv[1] == 'nwft2json':
        ft_embeds = '/iesl/canvas/smysore/material_science_framex/datasets_proc/anyt/' \
                    'embeddings/fasttext-feb2019/wiki-news-300d-1M-subword.vec'
        out_fname = '/iesl/canvas/smysore/material_science_framex/datasets_proc/anyt/' \
                    'embeddings/fasttext-feb2019/300d.embed.json'
        nwfasttext2json(in_fname=ft_embeds, out_fname=out_fname)
    else:
        print('Unknown command.')

