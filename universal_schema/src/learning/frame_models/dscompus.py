"""
A universal schema style model that attempts to learn affinity of predicates
for argument sets based on a deep-argument-set representation. This subclasses
the compvschema model class and implements a deep set model for column
composition.
"""
from __future__ import print_function
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional

from models_common import model_utils as mu
import compvschema as cvs
import dsnet as ds
import sys
import codecs
# Write unicode to stdout.
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)


class DeepSetCVS(cvs.GenericCompVS):
    """
    - Treat the event arguments as a order invariant set.
    """
    def __init__(self, row2idx, col2idx, pred_embedding_dim, arg_embedding_dim,
                 out_arg_embedding_dim, arg_set_dim, dropout,
                 embedding_path=None, size_average=False, criterion='bpr'):
        """
        :param row2idx: dict(string:int); dict mapping the predicates to ints.
        :param col2idx: dict(string:int); dict mapping the arguments to ints.
        :param pred_embedding_dim: int; dimension of the predicate embedding.
        :param arg_embedding_dim: int; dimension of the argument embeddings read in.
        :param out_arg_embedding_dim: int; dimension of the argument embeddings
            inside the deep-set.
        :param arg_set_dim: int; dimension of the final column elements.
        :param dropout: float; use one dropout value everywhere.
        :param embedding_path: string; path to initialize row/predicate embeddings
            from.
        :param size_average: boolean; whether the per-batch loss should get averaged.
        :param criterion: string; {'bpr'/'nce'}
        """
        cvs.GenericCompVS.__init__(self, row2idx=row2idx, col2idx=col2idx,
                                   embedding_path=embedding_path, embedding_dim=pred_embedding_dim,
                                   arg_embedding_dim=arg_embedding_dim, hidden_dim=arg_set_dim,
                                   size_average=size_average, criterion=criterion)
        # Define the elements of the architecture.
        self.in_drop = torch.nn.Dropout(p=dropout)
        # TODO: Allow finer grained control over the dimensions of all the hidden
        # layers in future. --low-pri.
        self.arg_set_net = ds.SetNet(in_elm_dim=arg_embedding_dim, out_elm_dim=out_arg_embedding_dim,
                                     composition_dims=(out_arg_embedding_dim, arg_set_dim))
        # Dropout at the final column outputs.
        self.colh_drop = torch.nn.Dropout(p=dropout)

    def forward(self, batch_dict, inference=False):
        return self.forward_vs(batch_dict, inference)

    def forward_vs(self, batch_dict, inference=False):
        """
        Pass through a forward pass and return the loss.
        :param batch_dict: dict; of the form:
            {'batch_cr': (batch_col_row) dict of the form:
                {'col': Torch Tensor; the padded and sorted-by-length col elements.
                 'row': Torch Tensor; predicates.
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
        :param inference: boolean; if True do pure inference; turn off dropout and
            dont store gradients.
        :return: loss; torch Variable.
        """
        batch_cr, batch_neg = batch_dict['batch_cr'], batch_dict['batch_neg']
        col, row, col_lens, row_lens, col_refs, row_refs = \
            batch_cr['col'], batch_cr['row'], batch_cr['col_lens'], \
            batch_cr['row_lens'], batch_cr['sorted_colrefs'],  batch_cr['sorted_rowrefs']
        col_neg, col_neg_lens, col_neg_refs = \
            batch_neg['col'], batch_neg['col_lens'], batch_neg['sorted_colrefs']

        # Pass the col and row through the appropriate composition functions.
        col_rep, col_elm_reps = self._col_compose(
            col=col, col_refs=col_refs, col_lengths=col_lens, inference=inference)
        row_rep = self._row_compose(row=row, row_refs=row_refs, inference=inference)
        col_neg_rep, _ = self._col_compose(
            col=col_neg,  col_refs=col_neg_refs, col_lengths=col_neg_lens,
            inference=inference)
        # At this point the stuff in the row and columns are aligned.
        comp_score_pos = torch.sum(row_rep * col_rep, dim=1)
        comp_score_neg = torch.sum(row_rep * col_neg_rep, dim=1)
        loss_val = self.criterion_bpr(true_ex_scores=comp_score_pos,
                                      false_ex_scores=comp_score_neg)
        return loss_val

    def _col_compose(self, col, col_refs, col_lengths, inference=False):
        """
        :param col: Torch Tensor; the padded column entities.
        :param col_refs: list(int); ints saying which seq in col came
                    from which document. ints in range [0, len(docs)]}
        :param col_lengths: list(int); lengths of all sequences in 'col'.
        :param inference: boolean; if True do pure inference; turn off dropout and
            don't store gradients.
        :return: agg_hidden; Torch Variable with the hidden representations for
            the batch.
        """
        # Make variable and move to GPU.
        col = Variable(col, volatile=inference)
        if torch.cuda.is_available():
            col = col.cuda()
        # Look up embeddings. This will be: batch_size x max_set_len x arg_embedding_dim
        arg_embeds = self.col_embeddings(col)
        # batch_size x embedding_dim
        if not inference:
            arg_embeds = self.in_drop(arg_embeds)
        # Get set and set elm reps; stuff going into and out of this is all in
        # unsorted order.
        arg_set_reps, arg_elm_reps = self.arg_set_net.forward(elms=arg_embeds)
        if not inference:
            arg_set_reps = self.colh_drop(arg_set_reps)
        return arg_set_reps, arg_elm_reps

    def predict(self, batch_dict):
        return self.predict_vs(batch_dict)

    def predict_vs(self, batch_dict):
        """
        Pass through a forward pass and compute scores for the batch rows and
        columns.
        :param batch_dict: dict of the form:
            {'batch_cr': (batch_col_row) dict of the form:
                {'col': Torch Tensor; the padded and sorted-by-length col elements.
                 'row': Torch Tensor; predicates.
                 'col_lens': list(int); lengths of all sequences in 'col'.
                 'row_lens': list(int); lengths of all sequences in 'row'.
                 'sorted_colrefs': list(int); ints saying which seq in col came
                        from which document. ints in range [0, len(docs)]
                 'sorted_rowrefs': list(int); ints saying which seq in row came
                        from which document. ints in range [0, len(docs)]}}
        :return:
            probs: numpy array(batch_size,); probs of the examples in the batch
                in original order.
            col_reps: numpy array(batch_size, rep_size); representations of the
                columns in original order.
            row_reps: numpy array(batch_size, rep_size); representations of the
                rows in original order.
        """
        batch_cr = batch_dict['batch_cr']
        col, row, col_lens, row_lens, col_refs, row_refs = \
            batch_cr['col'], batch_cr['row'], batch_cr['col_lens'], \
            batch_cr['row_lens'], batch_cr['sorted_colrefs'], batch_cr['sorted_rowrefs']

        batch_size = col.size(0)
        # Pass the col and row through the appropriate lstms.
        col_reps, col_elm_reps = self._col_compose(
            col=col, col_refs=col_refs, col_lengths=col_lens, inference=True)
        row_reps = self._row_compose(row=row, row_refs=row_refs, inference=True)
        # At this point the stuff in the hidden vectors is assumed to be
        # aligned. The compatibility between the rows and the columns; point
        # mul the rows and sum.:
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

        assert (probs.shape[0] == col_reps.shape[0] == row_reps.shape[0] == batch_size)
        return probs, col_reps, row_reps


class DeepEventSetCVS(cvs.GenericCompVS):
    """
    - Treat the event arguments and the predicate together as a order invariant set.
    - The network outputs one score for the candidate extraction instead of there
        being a linear interaction between predicate and argument set.
    """
    def __init__(self, row2idx, col2idx, pred_embedding_dim, arg_embedding_dim,
                 out_arg_embedding_dim, arg_set_dim, dropout,
                 embedding_path=None, size_average=False):
        """
        :param row2idx: dict(string:int); dict mapping the predicates to ints.
        :param col2idx: dict(string:int); dict mapping the arguments to ints.
        :param pred_embedding_dim: int; dimension of the predicate embedding.
        :param arg_embedding_dim: int; dimension of the argument embeddings read in.
        :param out_arg_embedding_dim: int; dimension of the argument embeddings
            inside the deep-set.
        :param arg_set_dim: int; dimension of the final column elements.
        :param dropout: float; use one dropout value everywhere.
        :param embedding_path: string; path to initialize row/predicate embeddings
            from.
        :param size_average: boolean; whether the per-batch loss should get averaged.
        """
        cvs.GenericCompVS.__init__(self, row2idx=row2idx, col2idx=col2idx,
                                   embedding_path=embedding_path, embedding_dim=pred_embedding_dim,
                                   arg_embedding_dim=arg_embedding_dim, hidden_dim=arg_set_dim,
                                   size_average=size_average)
        assert(pred_embedding_dim == arg_embedding_dim)
        # Define the elements of the architecture.
        self.in_drop = torch.nn.Dropout(p=dropout)
        # TODO: Allow finer grained control over the dimensions of all the hidden
        # layers in future. --low-pri.
        self.predarg_set_net = ds.SetNet(in_elm_dim=arg_embedding_dim,
                                         out_elm_dim=out_arg_embedding_dim,
                                         composition_dims=(out_arg_embedding_dim, arg_set_dim))
        # Dropout at the final column outputs.
        self.colh_drop = torch.nn.Dropout(p=dropout)

        # Define a network which predicts a single score.
        self.score_network = torch.nn.Linear(in_features=arg_set_dim, out_features=1, bias=True)

    def forward(self, batch_dict, inference=False):
        return self.forward_vs(batch_dict, inference)

    def forward_vs(self, batch_dict, inference=False):
        """
        Pass through a forward pass and return the loss.
        :param batch_dict: dict; of the form:
            {'batch_cr': (batch_col_row) dict of the form:
                {'col': Torch Tensor; the padded and sorted-by-length col elements.
                 'row': Torch Tensor; predicates.
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
        :param inference: boolean; if True do pure inference; turn off dropout and
            dont store gradients.
        :return: loss; torch Variable.
        """
        batch_cr, batch_neg = batch_dict['batch_cr'], batch_dict['batch_neg']
        col, row, col_lens, row_lens, col_refs, row_refs = \
            batch_cr['col'], batch_cr['row'], batch_cr['col_lens'], \
            batch_cr['row_lens'], batch_cr['sorted_colrefs'],  batch_cr['sorted_rowrefs']
        col_neg, col_neg_lens, col_neg_refs = \
            batch_neg['col'], batch_neg['col_lens'], batch_neg['sorted_colrefs']

        # Get the preds in unsorted order and look up pred embeddings.
        pred_rep = self._row_compose(row=row, row_refs=row_refs, inference=inference)
        # Get a combined deep set rep for the pred arg set.
        predarg_rep, predarg_elm_reps, unsorted_lens = self._event_compose(
            pred_embeds=pred_rep, arg=col, arg_refs=col_refs, arg_lengths=col_lens,
            inference=inference)
        predarg_neg_rep, _, _ = self._event_compose(
            pred_embeds=pred_rep, arg=col_neg,  arg_refs=col_neg_refs,
            arg_lengths=col_neg_lens, inference=inference)
        # Pass the reps through the final layer to get a scalar score.
        comp_score_pos = self.score_network.forward(input=predarg_rep)
        comp_score_neg = self.score_network.forward(input=predarg_neg_rep)
        loss_val = self.criterion_bpr(true_ex_scores=comp_score_pos,
                                      false_ex_scores=comp_score_neg)
        return loss_val

    def _event_compose(self, pred_embeds, arg, arg_refs, arg_lengths, inference=False):
        """
        :param arg: Torch Tensor; the padded and sorted-by-length entities.
        :param arg_refs: list(int); ints saying which seq in arg came
                    from which document. ints in range [0, len(docs)]}
        :param arg_lengths: list(int); lengths of all sequences in 'col'.
        :param inference: boolean; if True do pure inference; turn off dropout and
            don't store gradients.
        :return: agg_hidden; Torch Variable with the hidden representations for
            the batch.
        """
        batch_size, max_set_len = arg.size()
        # Make the masks to put the column elements into an unsorted order.
        arg_refs = np.array(arg_refs)
        unsrt_masks = np.zeros((batch_size, batch_size, max_set_len))
        for ref in xrange(batch_size):
            unsrt_masks[ref, arg_refs == ref, :] = 1.0
        unsrt_masks = torch.LongTensor(unsrt_masks)

        unsorted_paset_lengths = [0]*len(arg_lengths)
        for idx, (ref, seq_len) in enumerate(zip(arg_refs, arg_lengths)):
            # Add 1 because the predicate gets added here too.
            unsorted_paset_lengths[ref] = arg_lengths[idx] + 1

        # Make variable and move to GPU.
        arg, unsrt_masks = Variable(arg, volatile=inference), \
                           Variable(unsrt_masks, volatile=inference)
        # Make variable of the lengths for normalization.
        arg_lengths_var = Variable(torch.FloatTensor(unsorted_paset_lengths),
                                   volatile=inference)

        if torch.cuda.is_available():
            arg = arg.cuda()
            unsrt_masks = unsrt_masks.cuda()
            arg_lengths_var = arg_lengths_var.cuda()
        # Put the column sequences into the unsorted order;
        arg = torch.sum(arg * unsrt_masks, dim=1)
        # Look up embeddings. This will be: batch_size x max_set_len x arg_embedding_dim
        arg_embeds = self.col_embeddings(arg)
        # Concat the predicate to the argument set: batch_size x (max_set_len + 1) x arg_embedding_dim
        # There are no start-stops attached to the sets here.
        predarg_embeds = torch.cat([pred_embeds.unsqueeze(1), arg_embeds], dim=1)
        # batch_size x embedding_dim
        if not inference:
            predarg_embeds = self.in_drop(predarg_embeds)
        # Get set and set elm reps; stuff going into and out of this is all in
        # unsorted order.
        predarg_set_reps, predarg_elm_reps = self.predarg_set_net.forward(elms=predarg_embeds)
        # Divide by the set length so the rep doesnt prefer larger sets.
        predarg_set_reps_norm = predarg_set_reps.div(arg_lengths_var.unsqueeze(dim=1))
        if not inference:
            predarg_set_reps_norm = self.colh_drop(predarg_set_reps_norm)
        return predarg_set_reps_norm, predarg_elm_reps, unsorted_paset_lengths

    def predict(self, batch_dict):
        return self.predict_vs(batch_dict)

    def predict_vs(self, batch_dict):
        """
        Pass through a forward pass and compute scores for the batch rows and
        columns.
        :param batch_dict: dict of the form:
            {'batch_cr': (batch_col_row) dict of the form:
                {'col': Torch Tensor; the padded and sorted-by-length col elements.
                 'row': Torch Tensor; predicates.
                 'col_lens': list(int); lengths of all sequences in 'col'.
                 'row_lens': list(int); lengths of all sequences in 'row'.
                 'sorted_colrefs': list(int); ints saying which seq in col came
                        from which document. ints in range [0, len(docs)]
                 'sorted_rowrefs': list(int); ints saying which seq in row came
                        from which document. ints in range [0, len(docs)]}}
        :return:
            probs: numpy array(batch_size,); probs of the examples in the batch
                in original order.
            col_reps: numpy array(batch_size, rep_size); representations of the
                columns in original order.
            row_reps: numpy array(batch_size, rep_size); representations of the
                rows in original order.
        """
        batch_cr = batch_dict['batch_cr']
        col, row, col_lens, row_lens, col_refs, row_refs = \
            batch_cr['col'], batch_cr['row'], batch_cr['col_lens'], \
            batch_cr['row_lens'], batch_cr['sorted_colrefs'], batch_cr['sorted_rowrefs']

        batch_size = col.size(0)
        # Get the preds in unsorted order and look up pred embeddings.
        pred_rep = self._row_compose(row=row, row_refs=row_refs, inference=True)
        # Get a combined deep set rep for the pred arg set.
        predarg_rep, predarg_elm_reps, unsorted_lens = self._event_compose(
            pred_embeds=pred_rep, arg=col, arg_refs=col_refs, arg_lengths=col_lens,
            inference=True)
        # Pass the reps through the final layer to get a scalar score.
        comp_score = self.score_network.forward(input=predarg_rep)
        probs = torch.exp(functional.logsigmoid(comp_score))

        # Make numpy arrays and return.
        if torch.cuda.is_available():
            probs = probs.cpu().data.numpy()
            predarg_reps = predarg_rep.cpu().data.numpy()
            pred_rep = pred_rep.cpu().data.numpy()
        else:
            probs = probs.data.numpy()
            predarg_reps = predarg_rep.data.numpy()
            pred_rep = pred_rep.data.numpy()

        assert (probs.shape[0] == predarg_reps.shape[0] == pred_rep.shape[0] == batch_size)
        return probs, predarg_reps, pred_rep


class DeepSetSIGeneric_CVS(cvs.GenericCompVS):
    """
    General parent class for the deep-set net which uses side information. Child
    classes must inherit this and define side_info_reps.
    """
    def __init__(self, row2idx, col2idx, pred_embedding_dim, arg_embedding_dim,
                 out_arg_embedding_dim, arg_set_dim, si_dim, dropout, elm_si_comp,
                 embedding_path=None, size_average=False):
        """
        :param row2idx: dict(string:int); dict mapping the predicates to ints.
        :param col2idx: dict(string:int); dict mapping the arguments to ints.
        :param pred_embedding_dim: int; dimension of the predicate embedding.
        :param arg_embedding_dim: int; dimension of the argument embeddings.
        :param out_arg_embedding_dim: int; dimension of the argument embeddings
            inside the deep-set, these are contextualized set element rep
            dimensions where the context is provided by the side info embeddings.
        :param elm_si_comp: string; {'bilinear'/'cat'} says what kind of network
            the deep set should use for element contextualization.
        :param arg_set_dim: int; dimension of the final column elements.
        :param si_dim: int; dimension of the side information supplied.
            (scon, deps, types or combination thereof)
        """
        cvs.GenericCompVS.__init__(self, row2idx=row2idx, col2idx=col2idx,
                                   size_average=size_average, embedding_path=embedding_path,
                                   arg_embedding_dim=arg_embedding_dim, embedding_dim=pred_embedding_dim,
                                   hidden_dim=arg_set_dim)
        # Define the elements of the architecture.
        self.in_drop = torch.nn.Dropout(p=dropout)
        # TODO: Allow finer grained control over the dimensions of all the hidden
        # layers in future. --low-pri.
        self.arg_set_net = ds.SetNet(in_elm_dim=arg_embedding_dim, out_elm_dim=out_arg_embedding_dim,
                                     composition_dims=(out_arg_embedding_dim, arg_set_dim),
                                     elm_si_dim=si_dim, elm_si_comp=elm_si_comp)
        # Dropout at the final column outputs.
        self.colh_drop = torch.nn.Dropout(p=dropout)

    def forward(self, batch_dict, inference=False):
        return self.forward_vs(batch_dict, inference)

    def forward_vs(self, batch_dict, inference=False):
        """
        Pass through a forward pass and return the loss.
        :param batch_dict: dict; of the form:
            {'batch_cr': (batch_col_row) dict of the form:
                {'col': Torch Tensor; the padded and sorted-by-length col elements.
                 'col_si': Torch Tensor; the padded and sorted-by-length col side
                    info. Changes based on what sort of side info it is.
                 'row': Torch Tensor; predicates.
                 'col_lens': list(int); lengths of all sequences in 'col'.
                 'row_lens': list(int); lengths of all sequences in 'row'.
                 'sorted_colrefs': list(int); ints saying which seq in col came
                        from which document. ints in range [0, len(docs)]
                 'sorted_rowrefs': list(int); ints saying which seq in row came
                        from which document. ints in range [0, len(docs)]}
            'batch_neg': (batch_row_neg) dict of the form:
                {'col': Torch Tensor; the padded and sorted-by-length entities.
                 'col_lens': list(int); lengths of all sequences in 'col'.
                 'sorted_colrefs': list(int); ints saying which seq in row came
                        from which document. ints in range [0, len(docs)]}}
        :param inference: boolean; if True do pure inference; turn off dropout and dont store gradients.
        :return: loss; torch Variable.
        """
        batch_cr, batch_neg = batch_dict['batch_cr'], batch_dict['batch_neg']
        col, row, col_lens, row_lens, col_refs, row_refs = \
            batch_cr['col'], batch_cr['row'], batch_cr['col_lens'], \
            batch_cr['row_lens'], batch_cr['sorted_colrefs'], batch_cr['sorted_rowrefs']
        col_neg, col_neg_lens, col_neg_refs = \
            batch_neg['col'], batch_neg['col_lens'], batch_neg['sorted_colrefs']

        # Get the side info embeddings. batch_size x max_set_len x si_dim
        si_pos_reps = self.side_info_reps(batch_dict=batch_cr, inference=inference)
        si_neg_reps = self.side_info_reps(batch_dict=batch_neg, inference=inference)

        # Pass the col and row through the appropriate composition functions.
        col_rep, col_elm_reps = self._col_compose(
            col=col, col_refs=col_refs, si_col=si_pos_reps,
            col_lengths=col_lens, inference=inference)
        row_rep = self._row_compose(row=row, row_refs=row_refs, inference=inference)
        col_neg_rep, _ = self._col_compose(
            col=col_neg, col_refs=col_neg_refs, si_col=si_neg_reps,
            col_lengths=col_neg_lens, inference=inference)
        # At this point the stuff in the hidden vectors is assumed to be
        # aligned. The compatability between the rows and the columns of
        # positive examples:
        comp_score_pos = torch.sum(row_rep * col_rep, dim=1)
        comp_score_neg = torch.sum(row_rep * col_neg_rep, dim=1)
        loss_val = self.criterion_bpr(true_ex_scores=comp_score_pos,
                                      false_ex_scores=comp_score_neg)
        return loss_val

    def side_info_reps(self, batch_dict, inference=False):
        """
        Model using side information must know what the batch dict needs
        to contain for returning the side info reps. And must implement the
        architecture for outputing the side information embeddings.
        :param batch_dict: either a positive batch_cr dict or a batch_neg dict;
            check classes inheriting this class. Should have a tensor under
            'col_si' which is deps or something else.
        :param inference: boolean; if True do pure inference; turn off dropout
            and dont store gradients.
        :return:
            si_pos/neg_reps: Variable(); batch_size x max_set_len x si_dim;
                Everything returned should be in unsorted order.
        """
        raise NotImplementedError

    def _col_compose(self, col, col_refs, si_col, col_lengths, inference=False):
        """
        Even though it says column compose it uses the row element.
        :param col: Torch Tensor; the padded and sorted-by-length entities.
        :param col_refs: list(int); ints saying which seq in col came
                    from which document. ints in range [0, len(docs)]}
        :param si_col: Variable(); batch_size x max_set_len x si_dim;
        :param col_lengths: list(int); lengths of all sequences in 'col'.
        :param inference: boolean; if True do pure inference; turn off dropout and dont store gradients.
        :return: agg_hidden; Torch Variable with the hidden representations for the batch.
        """
        # Make variable of the lengths but only use in additive combination.
        col = Variable(col, volatile=inference)

        if torch.cuda.is_available():
            col = col.cuda()
        # Look up embeddings. This will be: batch_size x max_set_len x embedding_dim
        arg_embeds = self.col_embeddings(col)
        if not inference:
            arg_embeds = self.in_drop(arg_embeds)
        # Get event and role reps; stuff going into and out of this is all in
        # unsorted order.
        arg_set_reps, arg_elm_reps = self.arg_set_net.forward(elms=arg_embeds,
                                                              side_info=si_col)
        if not inference:
            arg_set_reps = self.colh_drop(arg_set_reps)
        return arg_set_reps, arg_elm_reps

    def predict(self, batch_dict):
        return self.predict_vs(batch_dict)

    def predict_vs(self, batch_dict):
        """
        Pass through a forward pass and compute scores for the batch rows and
        columns.
        :param batch_dict: dict of the form:
            {'batch_cr': (batch_col_row) dict of the form:
                {'col': Torch Tensor; the padded and sorted-by-length sentence.
                 'col_si': Torch Tensor; the padded and sorted-by-length col side
                    info. Changes based on what sort of side info it is.
                 'row': Torch Tensor; predicates.
                 'col_lens': list(int); lengths of all sequences in 'col'.
                 'row_lens': list(int); lengths of all sequences in 'row'.
                 'sorted_colrefs': list(int); ints saying which seq in col came
                        from which document. ints in range [0, len(docs)]
                 'sorted_rowrefs': list(int); ints saying which seq in row came
                        from which document. ints in range [0, len(docs)]}}
        :return:
            probs: numpy array(batch_size,); probs of the examples in the batch
                in original order.
            col_reps: numpy array(batch_size, rep_size); representations of the
                columns in original order.
            row_reps: numpy array(batch_size, rep_size); representations of the
                rows in original order.
        """
        batch_cr = batch_dict['batch_cr']
        col, row, col_lens, row_lens, col_refs, row_refs = \
            batch_cr['col'], batch_cr['row'], batch_cr['col_lens'], \
            batch_cr['row_lens'], batch_cr['sorted_colrefs'], batch_cr['sorted_rowrefs']

        batch_size = col.size(0)

        # Get the side info embeddings. batch_size x max_set_len x si_dim
        si_pos_reps = self.side_info_reps(batch_dict=batch_cr, inference=True)
        # Pass the col and row through the appropriate col composition functions.
        col_rep, col_elm_reps = self._col_compose(
            col=col, col_refs=col_refs, si_col=si_pos_reps,
            col_lengths=col_lens, inference=True)
        row_rep = self._row_compose(row=row, row_refs=row_refs, inference=True)
        # At this point the stuff in the set vectors is assumed to be
        # aligned. The compatibility between the rows and the columns; point
        # mul the rows and sum.:
        comp_score = torch.sum(col_rep * row_rep, dim=1)
        probs = torch.exp(functional.logsigmoid(comp_score))

        # Make numpy arrays and return.
        if torch.cuda.is_available():
            probs = probs.cpu().data.numpy()
            col_rep = col_rep.cpu().data.numpy()
            row_rep = row_rep.cpu().data.numpy()
        else:
            probs = probs.data.numpy()
            col_rep = col_rep.data.numpy()
            row_rep = row_rep.data.numpy()

        assert (probs.shape[0] == col_rep.shape[0] == row_rep.shape[0] == batch_size)
        return probs, col_rep, row_rep


class DeepSetSI_CVS(DeepSetSIGeneric_CVS):
    """
    GraphNetSI_CVS with a dependency/entity type of the argument embeddings
    used as the side information.
    """
    def __init__(self, row2idx, col2idx, si2idx, pred_embedding_dim, arg_embedding_dim,
                 out_arg_embedding_dim, arg_set_dim, si_dim, dropout, elm_si_comp,
                 embedding_path=None, size_average=False):
        DeepSetSIGeneric_CVS.\
            __init__(self, row2idx=row2idx, col2idx=col2idx, pred_embedding_dim=pred_embedding_dim,
                     arg_embedding_dim=arg_embedding_dim, out_arg_embedding_dim=out_arg_embedding_dim,
                     arg_set_dim=arg_set_dim, si_dim=si_dim, dropout=dropout, elm_si_comp=elm_si_comp,
                     embedding_path=embedding_path, size_average=size_average)
        self.si_embeddings = mu.init_embeddings(word2idx=si2idx,
                                                embedding_dim=si_dim,
                                                embed_path=None)

    def side_info_reps(self, batch_dict, inference=False):
        """
        Return one-hot representations of the dependencies.
        :param batch_dict: either a positive batch_cr dict or a batch_neg dict;
            Should have a tensor under 'col_feats'. Feats here should be int
            mapped dependencies of the arguments. Batch IS sorted by length
            of arg n-tuple.
        :param inference: boolean; if True do pure inference; turn off dropout
            and don't store gradients.
        :return:
            si_pos/neg_reps: Variable(); batch_size x max_set_len x si_dim;
        """
        # deps should be batch_size x max_set_len
        col_si = batch_dict['col_si']
        # Make Variable and move to gpu.
        col_si = Variable(col_si, volatile=inference)
        if torch.cuda.is_available():
            col_si = col_si.cuda()
        si_embeds = self.si_embeddings(col_si)
        # dep_onehots should be batch_size x max_set_len x si_dim
        return si_embeds


class DeepSetPosition_CVS(DeepSetSIGeneric_CVS):
    """
    GraphNetSI_CVS with relative position of the argument embeddings
    used as the side information.
    This model class unused as of now.
    """
    def __init__(self, row2idx, col2idx, pred_embedding_dim, arg_embedding_dim,
                 out_arg_embedding_dim, arg_set_dim, si_dim, dropout, elm_si_comp,
                 embedding_path=None, size_average=False):
        DeepSetSIGeneric_CVS.\
            __init__(self, row2idx=row2idx, col2idx=col2idx, pred_embedding_dim=pred_embedding_dim,
                     arg_embedding_dim=arg_embedding_dim, out_arg_embedding_dim=out_arg_embedding_dim,
                     arg_set_dim=arg_set_dim, si_dim=si_dim, dropout=dropout, elm_si_comp=elm_si_comp,
                     embedding_path=embedding_path, size_average=size_average)

    def side_info_reps(self, batch_dict, inference=False):
        """
        Return one-hot representations of the dependencies.
        :param batch_dict: either a positive batch_cr dict or a batch_neg dict;
            Should have a tensor under 'col_feats'. Feats here should be int
            mapped dependencies of the arguments. Batch IS sorted by length
            of arg n-tuple.
        :param inference: boolean; if True do pure inference; turn off dropout
            and don't store gradients.
        :return:
            si_pos/neg_reps: Variable(); batch_size x max_set_len x si_dim;
        """
        # deps should be batch_size x max_set_len
        col_si = batch_dict['col_si']
        # Make Variable and move to gpu.
        col_si = Variable(col_si, volatile=inference)
        if torch.cuda.is_available():
            col_si = col_si.cuda()
        # side_info should be batch_size x max_set_len x si_dim so insert one extra dim.
        return col_si.unsqueeze(dim=2)
