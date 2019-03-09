"""
This is the compositional universal schema model used to learn verb/operation
schemas.
"""
from __future__ import print_function
import sys
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional

import compvschema as cvs


class CompVS(cvs.GenericCompVS):
    """
    Use LSTMs to compose elements along the columns to score likely
    verb and type combinations.
    """
    def __init__(self, row2idx, col2idx, lstm_comp, embedding_path=None, num_layers=1,
                 embedding_dim=50, hidden_dim=50, dropout=0.3, size_average=False):
        cvs.GenericCompVS.__init__(self, row2idx=row2idx, col2idx=col2idx,
                                   embedding_path=embedding_path,
                                   embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                                   size_average=size_average)
        self.num_layers = num_layers

        # Define the elements of the architecture.
        # How the vectors from the lstm should be used used to get tuple
        # representations. Valid choices are 'max', 'sum' or 'hidden'. Max
        # and add combine the output at each time step using those operations.
        # 'sum' isnt implemented yet.
        self.lstm_comp = lstm_comp
        self.in_drop = torch.nn.Dropout(p=dropout)
        # If asked to use lstm composition then initialize it.
        if self.lstm_comp in ['hidden', 'max']:
            # The column lstm to compose column elements (types or type,ents)
            self.col_lstm = torch.nn.LSTM(self.embedding_dim, self.hidden_dim)
        # If theres nothing transforming the tuple representation to the same sized
        # vec as the trigger embedding then make this assertion.
        elif self.lstm_comp == 'add':
            pass
        # Dropout at the lstm hidden outputs.
        self.colh_drop = torch.nn.Dropout(p=dropout)

    def forward(self, batch_dict, inference=False):
        return self.forward_vs(batch_dict, inference)

    def forward_vs(self, batch_dict, inference=False):
        """
        Pass through a forward pass and return the loss.
        :param batch_dict: dict; of the form:
            {'batch_cr': (batch_col_row) dict of the form:
                {'col': Torch Tensor; the padded and sorted-by-length col elements.
                 'row': Torch Tensor; the padded and sorted-by-length entities.
                 'col_lens': list(int); lengths of all sequences in 'col'.
                 'row_lens': list(int); lengths of all sequences in 'row'.
                 'sorted_colrefs': list(int); ints saying which seq in col came
                        from which document. ints in range [0, len(docs)]
                 'sorted_rowrefs': list(int); ints saying which seq in row came
                        from which document. ints in range [0, len(docs)]}
            'batch_neg': (batch_row_neg) dict of the form:
                {'row': Torch Tensor; the padded and sorted-by-length entities.
                 'row_lens': list(int); lengths of all sequences in 'col'.
                 'sorted_rowrefs': list(int); ints saying which seq in row came
                        from which document. ints in range [0, len(docs)]}}
        :param inference: boolean; if True do pure inference; turn off dropout and dont store gradients.
        :return: loss; torch Variable.
        """
        batch_cr, batch_neg = batch_dict['batch_cr'], batch_dict['batch_neg']
        col, row, col_lens, row_lens, col_refs, row_refs = \
            batch_cr['col'], batch_cr['row'], batch_cr['col_lens'],\
            batch_cr['row_lens'], batch_cr['sorted_colrefs'],\
            batch_cr['sorted_rowrefs']
        col_neg, col_neg_lens, col_neg_refs = \
            batch_neg['col'], batch_neg['col_lens'], batch_neg['sorted_colrefs']

        # Pass the col and row through the appropriate lstms.
        col_rep = self._col_compose(col=col, col_refs=col_refs,
                                    col_lengths=col_lens, inference=inference)
        row_rep = self._row_compose(row=row, row_refs=row_refs, inference=inference)
        col_neg_rep = self._col_compose(col=col_neg, col_refs=col_neg_refs,
                                        col_lengths=col_neg_lens, inference=inference)
        # At this point the stuff in the hidden vectors is assumed to be
        # aligned. The compatability between the rows and the columns of
        # positive examples:
        comp_score_pos = torch.sum(row_rep * col_rep, dim=1)
        comp_score_neg = torch.sum(row_rep * col_neg_rep, dim=1)
        loss_val = self.criterion_bpr(true_ex_scores=comp_score_pos,
                                      false_ex_scores=comp_score_neg)
        return loss_val

    def _col_compose(self, col, col_refs, col_lengths, inference=False):
        """
        Pass through the col lstm and return its representation for a batch of
        col.
        :param col: Torch Tensor; the padded and sorted-by-length entities.
        :param col_refs: list(int); ints saying which seq in col came
                    from which document. ints in range [0, len(docs)]}
        :param col_lengths: list(int); lengths of all sequences in 'col'.
        :param inference: boolean; if True do pure inference; turn off dropout and dont store gradients.
        :return: agg_hidden; Torch Variable with the hidden representations for the batch.
        """
        total_examples = col.size(0)  # Batch size.
        if self.lstm_comp in ['max', 'hidden']:
            # Make initialized hidden and cell states for lstm.
            col_h0 = torch.zeros(self.num_layers, total_examples, self.hidden_dim)
            col_c0 = torch.zeros(self.num_layers, total_examples, self.hidden_dim)
        # Make the doc masks; there must be an easier way to do this. :/
        col_refs = np.array(col_refs)
        col_masks = np.zeros((total_examples, total_examples, self.hidden_dim))
        for ref in xrange(total_examples):
            col_masks[ref, col_refs == ref, :] = 1.0
        col_masks = torch.FloatTensor(col_masks)
        # Make variable of the lengths but only use in additive combination.
        col_lengths_var = Variable(torch.FloatTensor(col_lengths),
                                   volatile=inference)
        # Make all model variables to Variables and move to the GPU.
        if self.lstm_comp in ['max', 'hidden']:
            col_h0, col_c0 = Variable(col_h0, volatile=inference),\
                             Variable(col_c0, volatile=inference)
        col, col_masks = Variable(col, volatile=inference), \
                         Variable(col_masks, volatile=inference)
        if torch.cuda.is_available():
            col = col.cuda()
            if self.lstm_comp in ['max', 'hidden']:
                col_h0, col_c0 = col_h0.cuda(), col_c0.cuda()
            col_masks = col_masks.cuda()
            col_lengths_var = col_lengths_var.cuda()
        # Pass forward.
        embeds = self.col_embeddings(col)
        if inference == False:
            embeds = self.in_drop(embeds)

        if self.lstm_comp in ['max', 'hidden']:
            packed = torch.nn.utils.rnn.pack_padded_sequence(embeds, col_lengths,
                                                             batch_first=True)
            packed_out, (hidden, cell) = self.col_lstm(packed, (col_h0, col_c0))

            # Do a somewhat order invariant max operation on the outputs at each time
            # step.
            if self.lstm_comp == 'max':
                out, _ = torch.nn.utils.rnn.pad_packed_sequence(sequence=packed_out,
                                                                batch_first=True)
                out.contiguous()
                # Out is (batch_size x seq_len x hidden_dim)
                out_rep, out_idxs = torch.max(out, dim=1)
            # If self.lstm_comp is None then just use the final hidden state of the
            # lstm.
            elif self.lstm_comp == 'hidden':
                out_rep = hidden
        elif self.lstm_comp in ['add']:
            # Divide by the appropriate sequence lengths.
            out_rep = torch.sum(embeds, dim=1).div(col_lengths_var.unsqueeze(dim=1))
        # Put the hidden vectors into the unsorted order; all except one
        # vec will be zeroed out for each example in the batch.
        agg_hidden = torch.sum(out_rep * col_masks, dim=1)
        if inference == False:
            agg_hidden = self.colh_drop(agg_hidden)
        return agg_hidden

    def predict(self, batch_dict):
        return self.predict_vs(batch_dict)

    def predict_vs(self, batch_dict):
        """
        Pass through a forward pass and compute scores for the batch rows and
        columns.
        :param batch_dict: dict; of the form:
            {'batch_cr':
                {'col': Torch Tensor; the padded and sorted-by-length sentence.
                 'row': Torch Tensor; the padded and sorted-by-length entities.
                 'col_lens': list(int); lengths of all sequences in 'col'.
                 'row_lens': list(int); lengths of all sequences in 'row'.
                 'sorted_colrefs': list(int); ints saying which seq in col came
                        from which document. ints in range [0, len(docs)]
                 'sorted_rowrefs': list(int); ints saying which seq in row came
                        from which document. ints in range [0, len(docs)]}}
        :return:
            probs: numpy array(batch_size,); probs of the examples in the batch
                in original order.
            col_hidden: numpy array(batch_size, rep_size); representations of the
                columns in original order.
            row_hidden: numpy array(batch_size, rep_size); representations of the
                rows in original order.
        """
        batch_cr = batch_dict['batch_cr']
        col, row, col_lens, row_lens, col_refs, row_refs = \
            batch_cr['col'], batch_cr['row'], batch_cr['col_lens'],\
            batch_cr['row_lens'], batch_cr['sorted_colrefs'],\
            batch_cr['sorted_rowrefs']

        total_sents = col.size(0)
        # Pass the col and row through the appropriate lstms.
        col_hidden = self._col_compose(col=col, col_refs=col_refs,
                                       col_lengths=col_lens, inference=True)
        row_hidden = self._row_compose(row=row, row_refs=row_refs, inference=True)
        # At this point the stuff in the hidden vectors is assumed to be
        # aligned. The compatibility between the rows and the columns; point
        # mul the rows and sum.:
        comp_score = torch.sum(col_hidden * row_hidden, dim=1)
        probs = torch.exp(functional.logsigmoid(comp_score))

        # Make numpy arrays and return.
        if torch.cuda.is_available():
            probs = probs.cpu().data.numpy()
            col_hidden = col_hidden.cpu().data.numpy()
            row_hidden = row_hidden.cpu().data.numpy()
        else:
            probs = probs.data.numpy()
            col_hidden = col_hidden.data.numpy()
            row_hidden = row_hidden.data.numpy()

        assert(probs.shape[0] == col_hidden.shape[0] == row_hidden.shape[0]
               == total_sents)
        return probs, col_hidden, row_hidden
