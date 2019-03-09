"""
Utilities to feed and initialize the models.
"""
from __future__ import unicode_literals
from __future__ import print_function
import os, sys
import logging
import codecs

import torch

# Write unicode to stdout.
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)


def batched_loss(model, batcher, batch_size, ex_fnames, num_examples):
    """
    Make predictions batch by batch. Dont do any funky shuffling shit.
    :param model: the model object with a predict method.
    :param batcher: reference to model_utils.Batcher class.
    :param batch_size: int; number of docs to consider in a batch.
    :param ex_fnames: dict; which the batcher understands as having example
        file names.
    :param num_examples: int; number of examples in above files.
    :return: loss: float; total loss for the data passed.
    """
    # Intialize batcher but dont shuffle.
    loss_batcher = batcher(ex_fnames=ex_fnames, num_examples=num_examples,
                           batch_size=batch_size)
    loss = torch.FloatTensor([0])
    if torch.cuda.is_available():
        loss = loss.cuda()
    iteration = 0
    logging.info('Dev pass; Num batches: {:d}'.format(loss_batcher.num_batches))
    for batch_ids, batch_dict in loss_batcher.next_batch():
        # Pass batches forward but in inference mode.
        batch_objective = model.forward(batch_dict=batch_dict, inference=True)
        # Objective is a variable; Do your summation on the GPU.
        loss += batch_objective.data
        if iteration % 100 == 0:
            logging.info('\tDev pass; Iteration: {:d}/{:d}'.
                         format(iteration, loss_batcher.num_batches))
        iteration += 1
    if torch.cuda.is_available():
        loss = float(loss.cpu().numpy())
    return loss


def batched_predict(model, batcher, batch_size, ex_fnames, num_examples):
    """
    Make predictions batch by batch. Dont do any funky shuffling shit.
    :param model: the model object with a predict method.
    :param batcher: reference to model_utils.Batcher class.
    :param batch_size: int; number of docs to consider in a batch.
    :param pos_fname: string; path to the file with the positive examples.
    :param num_examples: int; number of examples in above file.
    :param return_reps: bool; says if learnt representations should be returned.
    :return: preds: numpy array; predictions on int_mapped_X.
    """
    # Intialize batcher.
    predict_batcher = batcher(ex_fnames=ex_fnames, num_examples=num_examples,
                              batch_size=batch_size)
    iteration = 0
    for batch_doc_ids, batch_dict in predict_batcher.next_batch():
        # Make a prediction.
        # this can be: batch_probs, batch_col_rep, batch_row_rep
        # or: batch_probs, batch_col_rep, batch_row_rep, batch_role_rep, batch_arg_lens
        # having it be a tuple allows this function to be reused.
        all_pred_items = model.predict(batch_dict=batch_dict)
        if iteration % 100 == 0:
            logging.info('\tPredict pass; Iteration: {:d}/{:d}'.
                         format(iteration, predict_batcher.num_batches))
        iteration += 1
        # Map int mapped tokens back to strings.
        yield batch_doc_ids, all_pred_items
