from __future__ import unicode_literals
from __future__ import print_function
import os, sys
import logging
import codecs
import json

import torch

# Write unicode to stdout.
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)


def init_onehot(feat2idx):
    """
    Initialize features as one hot embeddings which then just get learnt.
    :param feat2idx: dict; mapping from a categorical feature to an integer.
        <pad> needs to be 0.
    :return:
        embed: a torch.nn.Embedding with all embeddings. Of size len(feat2idx)+1
    """
    vocab_size = len(feat2idx)
    embedding_dim = len(feat2idx)
    # Make a matrix for the voacb words.
    padding_idx = 0
    # Initialize with all zeros.
    initialized = torch.zeros(vocab_size, vocab_size)

    for word, idx in feat2idx.iteritems():
        # Set the padding tok to zeros.
        if idx == padding_idx:
            continue
        # Set other embeddings to be one hots.
        initialized[idx, idx] = 1.0

    logging.info('Initialized with one hots; feat vocab size: {:d}; embedding dim: {:d}'.
                 format(*initialized.size()))
    embed = torch.nn.Embedding(vocab_size, embedding_dim)
    embed.weight = torch.nn.Parameter(initialized)
    return embed


def init_embeddings(embed_path, word2idx, embedding_dim):
    """
    Initialize the store for embeddings with the pre-trained embeddings
    or random values.
    :param embed_path: string; path to json file mapping word to a
        pretrained embedding.
    :param word2idx: dict; mapping from a word to an integer. <pad> needs to be 0.
    :param embedding_dim: int; size of the embedding.
    :return:
        embed: a torch.nn.Embedding with all embeddings.
    """
    vocab_size = len(word2idx)
    # Make a matrix for the voacb words.
    padding_idx = 0
    # Initialize from a normal distribution.
    pretrained = torch.randn(vocab_size, embedding_dim) * 0.01
    # Set the padding token embedding to zero. This is necessary because we want
    # to initialize with our set of initialized values. If you used nn.Embedding
    # out of the box it would already be initialized with a normal and have pad
    # set to zero.
    pretrained[0] = torch.FloatTensor(embedding_dim).zero_()
    # If no embedding file is provided return randomly initialized embeddings.
    if embed_path is None:
        logging.info('Initialized embeddings from torch.randn.{}'.format(pretrained.size()))
        embed = torch.nn.Embedding(vocab_size, embedding_dim)
        embed.weight = torch.nn.Parameter(pretrained)
        return embed

    # If not then read in the embedding file.
    embed_file = os.path.join(embed_path, '{:d}d.embed.json'.format(embedding_dim))
    with codecs.open(embed_file, 'r', 'utf-8') as fp:
        word2emb = json.load(fp)
        logging.info('Read embeddings: {:s}'.format(fp.name))
    mwe_oov_count = 0
    sinw_oov_count = 0
    mwe_count = 0
    sinw_count = 0
    for word, idx in word2idx.iteritems():
        found_pretrained = False
        # Set the padding tok to zeros.
        if idx == padding_idx:
            pretrained[idx] = torch.FloatTensor(embedding_dim).zero_()
            continue
        # Since the words are actually entities they can be multi token.
        toks = word.split()
        if len(toks) > 1:
            mwe_count += 1
            # The pretrained entity rep for a multiword entity is the average
            # of the pretrained reps of the individual tokens.
            summed_pt = torch.FloatTensor(embedding_dim).zero_()
            for tok in toks:
                # If the token of the entity is in the pretrained vocab use it
                # else use a random value.
                if tok in word2emb:
                    found_pretrained = True
                    tok_pt = torch.FloatTensor(word2emb[tok])
                else:
                    tok_pt = torch.randn(embedding_dim) * 0.01
                summed_pt += torch.FloatTensor(tok_pt)
            summed_pt = summed_pt/len(toks)
            pretrained[idx] = summed_pt
        else:
            sinw_count += 1
            if word in word2emb:
                found_pretrained = True
                pretrained[idx] = torch.FloatTensor(word2emb[word])
        # If word not in pretrained vocab keep track of it.
        if not found_pretrained:
            if len(toks) > 1:
                mwe_oov_count += 1
            else:
                sinw_oov_count += 1

    logging.info('Initialized vocab with pretrained: {:d}; {:d}/{:d} mwe words not in vocab;'
                 ' {:d}/{:d} sinw words not in vocab.'
                 .format(vocab_size, mwe_oov_count, mwe_count, sinw_oov_count, sinw_count))
    embed = torch.nn.Embedding(vocab_size, embedding_dim)
    embed.weight = torch.nn.Parameter(pretrained)
    return embed


def init_latent_types(type2idx, latent_width, embedding_dim):
    """
    Return a set of flattened matrices as a torch.nn.Embedding.
    :param type2idx: dict(typestr:int); type string to int mapping;
        <pad> needs to be 0.
    :param latent_width: int; the dimension of the space into which embeddings
        will get mapped.
    :param embedding_dim: int; the embedding dimension of the word embeddings.
    :return:
        flat_latent_mat: a torch.nn.Embedding with the matrices
            flattened to vectors.
    """
    num_types = len(type2idx)
    # Initialize from a normal distribution. I'm not sure if this is the "correct"
    # thing to do because dont you want every matrix to come from a 2d MVN instead
    # of one 3d MVN.
    random_normals = torch.randn(num_types, latent_width*embedding_dim) * 0.01
    # Set the padding token embedding to zero. This is necessary because we want
    # to initialize with our set of initialized values. If you used nn.Embedding
    # out of the box it would already be initialized with a normal and have pad
    # set to zero.
    random_normals[0] = torch.FloatTensor(latent_width*embedding_dim).zero_()
    logging.info('Initialized latent types from torch.randn: {}'.
                 format(random_normals.size()))
    # Flattened matrix.
    latent_mats = torch.nn.Embedding(num_types, latent_width*embedding_dim)
    latent_mats.weight = torch.nn.Parameter(random_normals)
    return latent_mats


if __name__ == '__main__':
    d={'<pad>':0, 'subj': 1, 'dobj':2, 'xcomp':3, 'ccomp': 4}
    print(init_onehot(d))