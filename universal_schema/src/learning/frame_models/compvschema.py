"""
The general model class which all the variants are derived from.
"""
import os
import sys
import numpy as np
import torch
from torch.autograd import Variable

# Add the upper level directory to the path.
# This is hack but its fine for now I guess.: https://stackoverflow.com/a/7506029/3262406
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from models_common import model_utils as mu
from models_common import loss


class GenericCompVS(torch.nn.Module):
    """
    Use some composition function to compose elements along the columns to score
    likely verb and type combinations.
    """
    def __init__(self, row2idx, col2idx, embedding_path, rel_dim=50,
                 arg_dim=25, size_average=False, criterion='bpr'):
        """
        :param row2idx: dict(str:int) relations.
        :param col2idx: dict(str:int) individual entities though a column
            is technically a pair of entities.
        :param embedding_path: string; path to some pretrained embeddings to use
            for the row and cols. If you want them to be separate, implement it.
            Initialized randomly if this is None.
        :param rel_dim: int; dimension of the row embeddings.
        :param arg_dim: int; dimension of the argument embeddings.
        :param size_average: bool; whether the loss must be averaged across the batch.
        :param criterion: strong; {'bpr'/'nce'}
        """
        # Calling it this way so that the multiple inheritance in the multi class
        # model works the way I expect. Hopefully.
        # https://stackoverflow.com/a/29312145/3262406
        torch.nn.Module.__init__(self)
        self.size_average = size_average
        self.rel_dim = rel_dim
        # Use the argument embedding size if it isnt None else use the row emb size.
        self.arg_dim = arg_dim if arg_dim else rel_dim
        # In this model the embeddings for rows should be twice the size of
        # the arg_dims because the final col dim is the args concated.
        assert (self.rel_dim == 2*self.arg_dim)

        # Voacb info.
        self.row2idx = row2idx
        self.col2idx = col2idx
        self.row_vocab_size = len(self.row2idx)
        self.col_vocab_size = len(self.col2idx)

        # Define the elements of the architecture.
        self.row_embeddings = mu.init_embeddings(
            embed_path=embedding_path, word2idx=row2idx,
            embedding_dim=self.rel_dim)
        self.col_embeddings = mu.init_embeddings(
            embed_path=embedding_path, word2idx=col2idx,
            embedding_dim=self.arg_dim)
        if criterion == 'bpr':
            # TODO: Rename this criterion to be more meaningful and generic. --med-pri.
            self.criterion_bpr = loss.BPRLoss(size_average=size_average)
        elif criterion == 'nce':
            self.criterion_bpr = loss.NCELoss(size_average=size_average)
        else:
            raise ValueError('Unknown criterion: {}'.format(criterion))

    def forward(self, batch_dict, inference=False):
        """
        Pass through a forward pass and return the loss. This needs to be
        implemented for any model subclassing this class.
        :param batch_dict:
            {pos: (batch_col_row) dict with the rows, columns and any
            other data you need.
            neg: (batch_col_row) dict with the rows, columns and any
            other data for the negative examples that.}
        :param inference: boolean; if True do pure inference;
            turn off dropout and dont store gradients.
        :return: loss; torch Variable.
        """
        raise NotImplementedError

    def _row_compose(self, row, inference=False):
        """
        Return the row representations. In this model this is just an embedding
        lookup. And remains the same across all the verb-schema models.
        :param row: Torch Tensor; int mapped row elements.
        :param inference: boolean; if True do pure inference; turn off dropout
            and dont store gradients.
        :return: embeds; Torch Variable with the relation representations for the batch.
        """
        # Make them to Variables and move to GPU.
        row = Variable(row, volatile=inference)
        if torch.cuda.is_available():
            row = row.cuda()
        # Pass forward.
        embeds = self.row_embeddings(row)
        return embeds

    def _col_compose(self, *input):
        """
        This defines any composition that one may choose to use. Any model subclassing
        this should implement this method.
        :return:
        """
        raise NotImplementedError

    def predict(self, batch_dict):
        """
        Pass through a forward pass and compute scores for the batch rows and
        columns.
       :param batch_dict: dict with the rows, columns and any
            other data you need.
        :return: predictions; in some form that one of the utils for writing out
            knows what to do with.
        """
        raise NotImplementedError
