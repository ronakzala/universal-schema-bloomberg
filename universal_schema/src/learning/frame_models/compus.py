"""
Variants of the universal schema model.
"""
from __future__ import print_function
import sys
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional

import compvschema as cvs


class LatentFeatureUS(cvs.GenericCompVS):
    """
    The latent feature universal schema model from the 2013 US paper:
    http://aclweb.org/anthology/N13-1008
    With the slight modification that entity-pair representations are simply
    the entity embeddings concated together.
    Rows are relations, columns are entity pairs.
    """
    def __init__(self, row2idx, col2idx, embedding_path=None, rel_dim=50,
                 arg_dim=25, dropout=0.3, size_average=False, criterion='bpr'):
        cvs.GenericCompVS.__init__(self, row2idx=row2idx, col2idx=col2idx,
                                   embedding_path=embedding_path,
                                   rel_dim=rel_dim, arg_dim=arg_dim,
                                   size_average=size_average, criterion=criterion)
        self.in_drop = torch.nn.Dropout(p=dropout)

    def forward(self, batch_dict, inference=False):
        return self.forward_vs(batch_dict, inference)

    def forward_vs(self, batch_dict, inference=False):
        """
        Pass through a forward pass and return the loss.
        :param batch_dict: dict; of the form:
            {'batch_cr': (batch_col_row) dict of the form:
                {'col': Torch Tensor; col elements; this will depend on what
                    the batcher implements, but it will be the entity-pair in the
                    column.
                 'row': Torch Tensor; row elements; the relations.}
            'batch_neg': (batch_row_neg) dict of the form:
                {'col': Torch Tensor; negative column elements.}
        :param inference: boolean; if True do pure inference; turn off dropout and dont store gradients.
        :return: loss; torch Variable.
        """
        batch_cr, batch_neg = batch_dict['batch_cr'], batch_dict['batch_neg']
        col, row = batch_cr['col'], batch_cr['row']
        col_neg = batch_neg['col']

        # Get the column and row representations.
        row_rep = self._row_compose(row=row, inference=inference)
        col_rep = self._col_compose(col=col, inference=inference)
        col_neg_rep = self._col_compose(col=col_neg, inference=inference)
        # The compatability between the rows and the columns of
        # positive examples:
        comp_score_pos = torch.sum(row_rep * col_rep, dim=1)
        comp_score_neg = torch.sum(row_rep * col_neg_rep, dim=1)
        loss_val = self.criterion_bpr(true_ex_scores=comp_score_pos,
                                      false_ex_scores=comp_score_neg)
        return loss_val

    def _col_compose(self, col, inference=False):
        """
        Look up embeddings for column elements, concat the reps for the pair and return.
        :param col: Torch Tensor; the padded and sorted-by-length entities.
        :param inference: boolean; if True do pure inference; turn off dropout and dont store gradients.
        :return: agg_hidden; Torch Variable with the hidden representations for the batch.
        """
        col = Variable(col, volatile=inference)
        if torch.cuda.is_available():
            col = col.cuda()
        # Look up embeddings: batch_size x 2 x arg_dim
        col_embeds = self.col_embeddings(col)
        if inference == False:
            col_embeds = self.in_drop(col_embeds)
        # This can be wrong based on how the col_embeds is layed out check to
        # make sure it works.
        cat_embeds = torch.cat([col_embeds[:, 0, :],
                                col_embeds[:, 1, :]], dim=1)
        return cat_embeds

    def predict(self, batch_dict):
        return self.predict_vs(batch_dict)

    def predict_vs(self, batch_dict):
        """
        Pass through a forward pass and compute scores for the batch rows and
        columns.
        :param batch_dict: dict; of the form:
            {'batch_cr':
                {'col': Torch Tensor; col elements; this will depend on what
                    the batcher implements, but it will be the entity-pair in the
                    column.
                 'row': Torch Tensor; row elements; the relations.}}
        :return:
            probs: numpy array(batch_size,); probs of the examples in the batch
                in original order.
            col_reps: numpy array(batch_size, rep_size); representations of the
                columns in original order.
            row_reps: numpy array(batch_size, rep_size); representations of the
                rows in original order.
        """
        batch_cr = batch_dict['batch_cr']
        col, row = batch_cr['col'], batch_cr['row']

        # Pass the col and row through the appropriate lstms.
        col_reps = self._col_compose(col=col, inference=True)
        row_reps = self._row_compose(row=row, inference=True)
        # Compute prob for col-row pair.
        comp_score = torch.sum(col_reps * row_reps, dim=1)
        probs = torch.exp(functional.logsigmoid(comp_score))

        # Make numpy arrays and return.
        if torch.cuda.is_available():
            probs = probs.cpu().data.numpy()
            col_reps = col_reps.cpu().data.numpy()
            row_reps = row_reps.cpu().data.numpy()
        else:
            probs = probs.data.numpy()
            col_reps = col_reps.data.numpy()
            row_reps = row_reps.data.numpy()

        assert(probs.shape[0] == col_reps.shape[0] == row_reps.shape[0])
        return probs, col_reps, row_reps
