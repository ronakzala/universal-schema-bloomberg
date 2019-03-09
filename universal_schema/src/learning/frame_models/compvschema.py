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
    def __init__(self, row2idx, col2idx, embedding_path=None, embedding_dim=50,
                 hidden_dim=50, arg_embedding_dim=None, size_average=False,
                 criterion='bpr'):
        """
        :param row2idx: dict(str:int)
        :param col2idx: dict(str:int)
        :param embedding_path: string; path to the row embeddings.
        :param embedding_dim: int; dimension of the row embeddings.
        :param hidden_dim: int; dimension of the final column embedding.
        :param arg_embedding_dim: int; dimension of the argument embeddings.
        :param size_average: bool; whether the loss must be averaged across the batch.
        :param criterion: strong; {'bpr'/'nce'}
        """
        # Calling it this way so that the multiple inheritance in the multi class
        # model works the way I expect. Hopefully.
        # https://stackoverflow.com/a/29312145/3262406
        torch.nn.Module.__init__(self)
        self.size_average = size_average
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        # Use the argument embedding size if it isnt None else use the row emb size.
        self.arg_embedding_dim = arg_embedding_dim if arg_embedding_dim else embedding_dim
        # In this model the embeddings for rows should be the same
        # as that for the representations of the columns.
        assert (self.embedding_dim == self.hidden_dim)

        # Voacb info.
        self.row2idx = row2idx
        self.col2idx = col2idx
        self.row_vocab_size = len(self.row2idx)
        self.col_vocab_size = len(self.col2idx)

        # Define the elements of the architecture.
        self.row_embeddings = mu.init_embeddings(
            embed_path=embedding_path, word2idx=row2idx,
            embedding_dim=self.embedding_dim)
        self.col_embeddings = mu.init_embeddings(
            embed_path=embedding_path, word2idx=col2idx,
            embedding_dim=self.arg_embedding_dim)
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

    def _row_compose(self, row, row_refs, inference=False):
        """
        Return the row representations. In this model this is just an embedding
        lookup. And remains the same across all the verb-schema models.
        :param row: Torch Tensor; the padded and sorted-by-length entities.
        :param row_refs: list(int); ints saying which seq in row came
                    from which document. ints in range [0, len(docs)]}
        :param inference: boolean; if True do pure inference; turn off dropout and dont store gradients.
        :return: agg_hidden; Torch Variable with the hidden representations for the batch.
        """
        total_examples = row.size(0)  # Batch size.
        # Make the doc masks; there must be an easier way to do this. :/
        row_refs = np.array(row_refs)
        row_masks = np.zeros((total_examples, total_examples, self.hidden_dim))
        for ref in xrange(total_examples):
            row_masks[ref, row_refs == ref, :] = 1.0
        row_masks = torch.FloatTensor(row_masks)
        # Make them to Variables and move to GPU.
        row, row_masks = Variable(row, volatile=inference), \
                         Variable(row_masks, volatile=inference)
        if torch.cuda.is_available():
            row = row.cuda()
            row_masks = row_masks.cuda()
        # Pass forward.
        embeds = self.row_embeddings(row)
        # Put the hidden vectors into the unsorted order >_<.
        agg_hidden = torch.sum(embeds * row_masks, dim=1)

        return agg_hidden

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
       :param batch_dict: (batch_col_row) dict with the rows, columns and any
            other data you need.
        :return: loss; torch Variable.
        """
        raise NotImplementedError
