"""
Classes to stream int-mapped data from file in batches, pad and sort them (as needed)
and return batch dicts for the models.
"""
from __future__ import unicode_literals
from __future__ import print_function
import codecs
import copy
import random
from collections import defaultdict

import numpy as np
import torch

import data_utils as du
import le_settings as les

import pprint, sys
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)


class GenericBatcher:
    def __init__(self, num_examples, batch_size):
        """
        Maintain batcher variables, state and such. Any batcher for a specific
        model is a subclass of this and implements specific methods that it
        needs.
        - A batcher needs to know how to read from an int-mapped raw-file.
        - A batcher should yield a dict which you model class knows how to handle.
        :param num_examples: the number of examples in total.
        :param batch_size: the number of examples to have in a batch.
        """
        # Batch sizes book-keeping.
        self.full_len = num_examples
        self.batch_size = batch_size
        if self.full_len > self.batch_size:
            self.num_batches = int(np.ceil(float(self.full_len) / self.batch_size))
        else:
            self.num_batches = 1

        # Get batch indices.
        self.batch_start = 0
        self.batch_end = self.batch_size

    def next_batch(self):
        """
        This should yield the dict which your model knows how to make sense of.
        :return:
        """
        pass

    @staticmethod
    def raw_batch_from_file(ex_file, num_batches, to_read_count):
        """
        Implement whatever you need for reading a raw batch of examples.
        Read the next batch from the file.
        :param ex_file: File-like with a next() method.
        :param num_batches: int; number of batches to read.
        :param to_read_count: int; number of examples to read from the file.
        :return:
        """
        pass


class RCBatcher(GenericBatcher):
    """
    A batcher to feed a model which inputs row-column as a single example.
    """
    # Whether the padding on sequences should be ignored.
    # (If yes simply ignores the first and last elm, make sure data has
    # actually start-stop.)
    ignore_ss = False

    def __init__(self, ex_fnames, num_examples, batch_size):
        """
        Batcher class for the universal-schema-like models which need a positive
        and a negative example.
        :param ex_fnames: dict('pos_ex_fname': str, 'neg_ex_fname': str)
        :param num_examples: int.
        :param batch_size: int.
        """
        GenericBatcher.__init__(self, num_examples=num_examples,
                                batch_size=batch_size)
        pos_ex_fname = ex_fnames['pos_ex_fname']
        neg_ex_fname = ex_fnames.get('neg_ex_fname', None)
        # Check that a file with negative examples has been provided.
        self.train_mode = True if neg_ex_fname!=None else False

        # Access the file with the positive and negative examples.
        self.pos_ex_file = codecs.open(pos_ex_fname, 'r', 'utf-8')
        if self.train_mode:
            self.neg_ex_file = codecs.open(neg_ex_fname, 'r', 'utf-8')

    def next_batch(self):
        """
        Yield the next batch. Based on whether its train_mode or not yield a
        different set of items.
        :return:
            batch_doc_ids: list; with the doc_ids corresponding to the
                    examples in the batch.
            In train mode:
                batch_dict:
                    {'batch_cr': dict; of the form returned by pad_sort_data.
                    'batch_neg': dict; of the form returned by pad_sort_data.}
            else:
                batch_dict:
                    {'batch_cr': dict; of the form returned by pad_sort_data.}
        """
        for nb in xrange(self.num_batches):
            # Read the batch of int-mapped data from the file.
            if self.batch_end < self.full_len:
                cur_batch_size = self.batch_size
            else:
                cur_batch_size = self.full_len - self.batch_start
            batch_doc_ids, batch_row_raw, batch_col_raw = \
                RCBatcher.raw_batch_from_file(self.pos_ex_file, cur_batch_size).next()
            if self.train_mode:
                _, _, batch_col_raw_neg = \
                    RCBatcher.raw_batch_from_file(self.neg_ex_file, cur_batch_size).next()
            self.batch_start = self.batch_end
            self.batch_end += self.batch_size
            # Process the batch for feeding into rnn models; sort the batch and
            # pad shorter examples to be as long as the longest one.
            if self.train_mode:
                batch_cr, batch_neg = RCBatcher.pad_sort_batch(raw_feed={
                    'im_row_raw': batch_row_raw,
                    'im_col_raw': batch_col_raw,
                    'im_col_raw_neg': batch_col_raw_neg
                }, ignore_ss=self.ignore_ss, sort_by_seqlen=False)
                batch_dict = {
                    'batch_cr': batch_cr,
                    'batch_neg': batch_neg
                }
            else:
                batch_cr = RCBatcher.pad_sort_batch(raw_feed={
                    'im_row_raw': batch_row_raw,
                    'im_col_raw': batch_col_raw
                }, ignore_ss=self.ignore_ss, sort_by_seqlen=False)
                batch_dict = {
                    'batch_cr': batch_cr
                }
            yield batch_doc_ids, batch_dict

    @staticmethod
    def raw_batch_from_file(ex_file, to_read_count):
        """
        Read the next batch from the file.
        :param ex_file: File-like with a next() method.
        :param to_read_count: int; number of lines to read from the file.
        :return:
            read_ex_rows: list(list(int)); a list where each element is a list
                which is the int mapped example row.
            read_ex_cols: list(list(int)); a list where each element is a list
                which is the int mapped example col.
        """
        # Initial values.
        read_ex_count = 0
        read_ex_docids = []
        read_ex_rows = []
        read_ex_cols = []
        # Read content from file until the file content is exhausted.
        for ex in du.read_json(ex_file):
            # If it was possible to read a valid example.
            if ex:
                read_ex_docids.append(ex['doc_id'])
                read_ex_rows.append(ex['row'])
                read_ex_cols.append(ex['col'])
                read_ex_count += 1
            if read_ex_count == to_read_count:
                yield read_ex_docids, read_ex_rows, read_ex_cols
                # Once execution is back here empty the lists and reset counters.
                read_ex_count = 0
                read_ex_docids = []
                read_ex_rows = []
                read_ex_cols = []

    @staticmethod
    def pad_sort_batch(raw_feed, ignore_ss, sort_by_seqlen=True):
        """
        Pad the data and sort such that the sentences are sorted in descending order
        of sentence length. Jumble all the sentences in this sorting but also
        maintain a list which says which sentence came from which document of the
        same length as the total number of sentences with elements in
        [0, len(int_mapped_docs)]
        :param raw_feed: dict; a dict with the set of things you want to feed
            the model. Here the elements are:
            im_row_raw: list(list(int)); a list where each element is a list
                    which is the int mapped example row.
            im_col_raw: list(list(int)); a list where each element is a list
                    which is the int mapped example col.
            im_col_raw_neg: list(list(int)); a list where each element is a list
                which is the int mapped example col. But this is a random set of
                col negative examples not corresponding to the rows.
        :param ignore_ss: boolean; Whether the start-stop on sequences should be
            ignored or not.
        :param sort_by_seqlen: boolean; Optionally allow sorting to be turned off.
        :return:
            colrow_ex: (batch_col_row) dict of the form:
                {'col': Torch Tensor; the padded and sorted-by-length col elements.
                 'row': Torch Tensor; the padded and sorted-by-length row elements.
                 'col_lens': list(int); lengths of all sequences in 'col'.
                 'row_lens': list(int); lengths of all sequences in 'row'.
                 'sorted_colrefs': list(int); ints saying which seq in col came
                        from which document. ints in range [0, len(docs)]
                 'sorted_rowrefs': list(int); ints saying which seq in row came
                        from which document. ints in range [0, len(docs)]}
            colneg_ex: (batch_row_neg) dict of the form:
                {'col': Torch Tensor; the padded and sorted-by-length entities.
                 'col_lens': list(int); lengths of all sequences in 'col'.
                 'sorted_colrefs': list(int); ints saying which seq in row came
                        from which document. ints in range [0, len(docs)]}
        """
        # Unpack arguments.
        im_row_raw = raw_feed['im_row_raw']
        im_col_raw = raw_feed['im_col_raw']
        im_col_raw_neg = raw_feed.get('im_col_raw_neg', None)
        # Do whatever.
        assert (len(im_col_raw) == len(im_row_raw))
        # If there is no data in the batch the model computes a zero loss. This never
        # happens in a purely rc model, happens in the emdsentvssup model.
        if len(im_row_raw) == 0:
            if isinstance(im_col_raw_neg, list):
                return None, None
            else:
                return None
        col, col_lens, sorted_colrefs = RCBatcher.pad_sort_ex_seq(
            im_col_raw, ignore_ss=ignore_ss, sort_by_seqlen=sort_by_seqlen)
        row, row_lens, sorted_rowrefs = RCBatcher.pad_sort_ex_seq(
            im_row_raw, ignore_ss=ignore_ss, sort_by_seqlen=sort_by_seqlen)
        colrow_ex = {'col': col,
                     'row': row,
                     'col_lens': col_lens,
                     'row_lens': row_lens,
                     'sorted_colrefs': sorted_colrefs,
                     'sorted_rowrefs': sorted_rowrefs}
        if im_col_raw_neg:
            assert (len(im_col_raw) == len(im_row_raw) == len(im_col_raw_neg))
            col_neg, col_neg_lens, sorted_neg_colrefs = RCBatcher.pad_sort_ex_seq(
                im_col_raw_neg, ignore_ss=ignore_ss, sort_by_seqlen=sort_by_seqlen)
            colneg_ex = {'col': col_neg,
                         'col_lens': col_neg_lens,
                         'sorted_colrefs': sorted_neg_colrefs}
            return colrow_ex, colneg_ex

        return colrow_ex

    @staticmethod
    def pad_sort_ex_seq(im_ex_seq, ignore_ss, sort_by_seqlen=True, pad_int=0):
        """
        Pad and sort the passed list of sequences one corresponding to each
        example.
        :param im_ex_seq: list(list(int)); can be anything; each sublist can be
            a different length.
        :param ignore_ss: boolean; Whether the start-stop on sequences should be
            ignored or not.
        :param sort_by_seqlen: boolean; Optionally allow sorting to be turned off.
        :param pad_int: int; int value to use for padding.
        :return:
            ex_seq_padded: torch.Tensor(len(im_ex_seq), len_of_longest_seq)
            sorted_lengths: list(int); lengths of sequences in im_ex_seq. Sorted.
            sorted_ref: list(int); indices of im_ex_seq elements in sorted order.
        """
        doc_ref = range(len(im_ex_seq))
        max_seq_len = max([len(l) for l in im_ex_seq])
        # Get sorted indices.
        if sort_by_seqlen:
            sorted_indices = sorted(range(len(im_ex_seq)),
                                    key=lambda k: -len(im_ex_seq[k]))
        else:
            sorted_indices = range(len(im_ex_seq))

        if ignore_ss:
            # If its the operation/row (single element long) then there's no need to
            # ignore start/stops.
            if len(im_ex_seq[sorted_indices[0]]) > 1:
                max_length = max_seq_len - 2
            else:
                max_length = max_seq_len
        else:
            max_length = max_seq_len

        # Make the padded sequence.
        ex_seq_padded = torch.LongTensor(len(im_ex_seq), max_length).zero_()
        if pad_int != 0:
            ex_seq_padded = ex_seq_padded + pad_int
        # Make the sentences into tensors sorted by length and place then into the
        # padded tensor.
        sorted_ref = []
        sorted_lengths = []
        for i, sent_i in enumerate(sorted_indices):
            seq = im_ex_seq[sent_i]
            if ignore_ss:
                if len(seq) > 1:
                    # Ignore the start and stops in the int mapped data.
                    seq = seq[1:-1]
                else:
                    seq = seq
            else:
                seq = seq
            tt = torch.LongTensor(seq)
            length = tt.size(0)
            ex_seq_padded[i, 0:length] = tt
            # Rearrange the doc refs.
            sorted_ref.append(doc_ref[sent_i])
            # Make this because packedpadded seq asks for it.
            sorted_lengths.append(length)

        return ex_seq_padded, sorted_lengths, sorted_ref


class RCSideInfoBatcher(RCBatcher):
    """
    A batcher to feed a model which inputs row-column-column side info as a single
    example.
    """
    # Make this a class variable so that it can be set before the class is instantiated.
    side_info = 'types'

    def next_batch(self):
        """
        Yield the next batch. Based on whether its train_mode or not yield a
        different set of items.
        :return:
            batch_doc_ids: list; with the doc_ids corresponding to the
                    examples in the batch.
            In train mode:
                batch_dict:
                    {'batch_cr': dict; of the form returned by pad_sort_data.
                    'batch_neg': dict; of the form returned by pad_sort_data.}
            else:
                batch_dict:
                    {'batch_cr': dict; of the form returned by pad_sort_data.}
        """
        for nb in xrange(self.num_batches):
            # Read the batch of int-mapped data from the file.
            if self.batch_end < self.full_len:
                cur_batch_size = self.batch_size
            else:
                cur_batch_size = self.full_len - self.batch_start
            batch_doc_ids, batch_row_raw, batch_col_raw, batch_col_si = \
                self.raw_batch_from_file(self.pos_ex_file, cur_batch_size, side_info=self.side_info).next()
            if self.train_mode:
                _, _, batch_col_raw_neg, batch_col_neg_si = \
                    self.raw_batch_from_file(self.neg_ex_file, cur_batch_size, side_info=self.side_info).next()
            self.batch_start = self.batch_end
            self.batch_end += self.batch_size
            # Process the batch for feeding into models; sort the batch and
            # pad shorter examples to be as long as the longest one.
            if self.train_mode:
                batch_cr, batch_neg = RCSideInfoBatcher.pad_sort_batch(raw_feed={
                    'im_row_raw': batch_row_raw,
                    'im_col_raw': batch_col_raw,
                    'im_col_si': batch_col_si,
                    'im_col_raw_neg': batch_col_raw_neg,
                    'im_col_si_neg': batch_col_neg_si
                }, ignore_ss=self.ignore_ss, sort_by_seqlen=False)
                batch_dict = {
                    'batch_cr': batch_cr,
                    'batch_neg': batch_neg
                }
            else:
                batch_cr = RCSideInfoBatcher.pad_sort_batch(raw_feed={
                    'im_row_raw': batch_row_raw,
                    'im_col_raw': batch_col_raw,
                    'im_col_si': batch_col_si
                }, ignore_ss=self.ignore_ss, sort_by_seqlen=False)
                batch_dict = {
                    'batch_cr': batch_cr
                }
            yield batch_doc_ids, batch_dict

    @staticmethod
    def raw_batch_from_file(ex_file, to_read_count, side_info):
        """
        Read the next batch from the file.
        :param ex_file: File-like with a next() method.
        :param to_read_count: int; number of lines to read from the file.
        :param side_info: string; says what column side information should also
            be read by the batcher. Expects the raw data to have a list of the
            same length as 'col' called 'col_{side_info}'.
            Can be: types/deps
        :return:
            read_ex_rows: list(list(int)); a list where each element is a list
                which is the int mapped example row.
            read_ex_cols: list(list(int)); a list where each element is a list
                which is the int mapped example col.
            read_ex_col_si: list(list(int)); a list where each element is a
                list which is the int mapped example col types.
        """
        # Initial values.
        read_ex_count = 0
        read_ex_docids = []
        read_ex_rows = []
        read_ex_cols = []
        read_ex_col_si = []
        # Read content from file until the file content is exhausted.
        for ex in du.read_json(ex_file):
            # If it was possible to read a valid example.
            if ex:
                read_ex_docids.append(ex['doc_id'])
                read_ex_rows.append(ex['row'])
                read_ex_cols.append(ex['col'])
                read_ex_col_si.append(ex['col_{:s}'.format(side_info)])
                read_ex_count += 1
            if read_ex_count == to_read_count:
                yield read_ex_docids, read_ex_rows, read_ex_cols, read_ex_col_si
                # Once execution is back here empty the lists and reset counters.
                read_ex_count = 0
                read_ex_docids = []
                read_ex_rows = []
                read_ex_cols = []
                read_ex_col_si = []

    @staticmethod
    def pad_sort_batch(raw_feed, ignore_ss, sort_by_seqlen=True):
        """
        Pad the data and sort such that the sentences are sorted in descending order
        of sentence length. Jumble all the sentences in this sorting but also
        maintain a list which says which sentence came from which document of the
        same length as the total number of sentences with elements in
        [0, len(int_mapped_docs)]
        :param raw_feed: dict; a dict with the set of things you want to feed
            the model. Here the elements are:
            im_row_raw: list(list(int)); a list where each element is a list
                    which is the int mapped example row.
            im_col_raw: list(list(int)); a list where each element is a list
                    which is the int mapped example col.
            im_col_si: list(list(int)); a list where each element is a list
                    which is the int mapped example cols entity type/dependency.
            im_col_raw_neg: list(list(int)); a list where each element is a list
                which is the int mapped example col. But this is a random set of
                col negative examples not corresponding to the rows.
            im_col_si_neg: list(list(int)); a list where each element is a list
                which is the int mapped example cols (entity) type. But this
                is a random set of col negative examples not corresponding to
                the rows.
        :param ignore_ss: boolean; Whether the start-stop on sequences should be
            ignored or not.
        :param sort_by_seqlen: boolean; Optionally allow sorting to be turned off.
        :return:
            colrow_ex: (batch_col_row) dict of the form:
                {'col': Torch Tensor; the padded and sorted-by-length col elements.
                 'col_si': Torch Tensor; the padded and sorted-by-length col elements.
                 'row': Torch Tensor; the padded and sorted-by-length row elements.
                 'col_lens': list(int); lengths of all sequences in 'col'.
                 'row_lens': list(int); lengths of all sequences in 'row'.
                 'sorted_colrefs': list(int); ints saying which seq in col came
                        from which document. ints in range [0, len(docs)]
                 'sorted_rowrefs': list(int); ints saying which seq in row came
                        from which document. ints in range [0, len(docs)]}
            colneg_ex: (batch_row_neg) dict of the form:
                {'col': Torch Tensor; the padded and sorted-by-length entities.
                 'col_si': Torch Tensor; the padded and sorted-by-length entity side info.
                 'col_lens': list(int); lengths of all sequences in 'col'.
                 'sorted_colrefs': list(int); ints saying which seq in row came
                        from which document. ints in range [0, len(docs)]}
        """
        # Unpack arguments.
        im_row_raw = raw_feed['im_row_raw']
        im_col_raw = raw_feed['im_col_raw']
        im_col_si = raw_feed['im_col_si']
        im_col_raw_neg = raw_feed.get('im_col_raw_neg', None)
        im_col_si_neg = raw_feed.get('im_col_si_neg', None)
        # Do whatever.
        assert (len(im_col_raw) == len(im_row_raw))
        col, col_lens, sorted_colrefs = RCBatcher.pad_sort_ex_seq(
            im_col_raw, ignore_ss=ignore_ss, sort_by_seqlen=sort_by_seqlen)
        row, row_lens, sorted_rowrefs = RCBatcher.pad_sort_ex_seq(
            im_row_raw, ignore_ss=ignore_ss, sort_by_seqlen=sort_by_seqlen)
        col_si, _, _ = RCBatcher.pad_sort_ex_seq(
            im_col_si, ignore_ss=ignore_ss, sort_by_seqlen=sort_by_seqlen)
        colrow_ex = {'col': col,
                     'row': row,
                     'col_si': col_si,
                     'col_lens': col_lens,
                     'row_lens': row_lens,
                     'sorted_colrefs': sorted_colrefs,
                     'sorted_rowrefs': sorted_rowrefs}
        if im_col_raw_neg:
            assert (len(im_col_raw) == len(im_row_raw) == len(im_col_raw_neg)
                    == len(im_col_si) == len(im_col_si_neg))
            col_neg, col_neg_lens, sorted_neg_colrefs = RCBatcher.pad_sort_ex_seq(
                im_col_raw_neg, ignore_ss=ignore_ss, sort_by_seqlen=sort_by_seqlen)
            col_si, _, _ = RCBatcher.pad_sort_ex_seq(
                im_col_si_neg, ignore_ss=ignore_ss, sort_by_seqlen=sort_by_seqlen)
            colneg_ex = {'col': col_neg,
                         'col_si': col_si,
                         'col_lens': col_neg_lens,
                         'sorted_colrefs': sorted_neg_colrefs}
            return colrow_ex, colneg_ex

        return colrow_ex


class RCPositionBatcher(RCSideInfoBatcher):
    """
    A batcher to feed a model which inputs row-column-column and position as a
    single example.
    """
    # The maximum distance of the argument from the predicate. Has to be set from
    # the outside.
    max_arg_pos = None

    def raw_batch_from_file(self, ex_file, to_read_count, side_info):
        """
        Read the next batch from the file.
        :param ex_file: File-like with a next() method.
        :param to_read_count: int; number of lines to read from the file.
        :param side_info: string; says what column side information should also
            be read by the batcher. Expects the raw data to have a list of the
            same length as 'col' called 'col_{side_info}'.
            Can be: types/deps
        :return:
            read_ex_rows: list(list(int)); a list where each element is a list
                which is the int mapped example row.
            read_ex_cols: list(list(int)); a list where each element is a list
                which is the int mapped example col.
            read_ex_col_id: list(list(int)); a list where each element is a
                list which is the int mapped example col ids (string or int
                based on dataset >_<).
            read_ex_row_id: list(int); a list of batch size long with the row_ids
                for the examples in the batch.
        """
        # Initial values.
        read_ex_count = 0
        read_ex_docids = []
        read_ex_rows = []
        read_ex_cols = []
        read_ex_col_si = []
        # Read content from file until the file content is exhausted.
        for ex in du.read_json(ex_file):
            # If it was possible to read a valid example.
            if ex:
                read_ex_docids.append(ex['doc_id'])
                read_ex_rows.append(ex['row'])
                read_ex_cols.append(ex['col'])
                col_si = self.convert_id2relpos(ex_col_ids=ex['col_ids'],
                                                ex_row_id=ex['row_id'][0],
                                                max_arg_pos=self.max_arg_pos)
                read_ex_col_si.append(col_si)
                read_ex_count += 1
            if read_ex_count == to_read_count:
                yield read_ex_docids, read_ex_rows, read_ex_cols, read_ex_col_si
                # Once execution is back here empty the lists and reset counters.
                read_ex_count = 0
                read_ex_docids = []
                read_ex_rows = []
                read_ex_cols = []
                read_ex_col_si = []

    @staticmethod
    def convert_id2relpos(ex_col_ids, ex_row_id, max_arg_pos):
        """
        Given the example column ids and the row id compute the difference between
        the two and return a list of length ex_col_ids with the relative distances.
        The start stop tokens are expected to be the same as the predicate so they
        get set to zero when the predicate id is subtracted from it.
        :param ex_col_ids: list(int/string); token indices of the column elements.
        :param ex_row_id: int/string; token index of the predicate.
        :param max_arg_pos: int; maximum distance of the argument from the predicate
            beyond which distances get collapsed into the max value.
        :return: ex_col_rel_pos: list(int); relative position of the argument wrt to
            the predicate.
        """
        # Sometimes the ids are strings of the type 'T{:d}' else they're ints.
        if not isinstance(ex_row_id, int):
            ex_row_id = int(ex_row_id[1:])
            ex_col_rel_pos = [int(strid[1:])-ex_row_id for strid in ex_col_ids]
        else:
            ex_col_rel_pos = [col_id-ex_row_id for col_id in ex_col_ids]
        bucketed_ex_col_rel_pos = [2]  # The start token.
        for dist in ex_col_rel_pos[1:-1]:  # Look at things between start-stop.
            if dist == 0:
                bucketed_ex_col_rel_pos.append(dist)
            elif dist > 0 and dist >= max_arg_pos:
                bucketed_ex_col_rel_pos.append(2*max_arg_pos+4-1)
            elif dist < 0 and dist <= -max_arg_pos:
                bucketed_ex_col_rel_pos.append(1+4-1)
            else:
                bucketed_ex_col_rel_pos.append((dist + max_arg_pos)//3 + 4 - 1)
        bucketed_ex_col_rel_pos.append(3)  # The stop symbol.
        assert(len(bucketed_ex_col_rel_pos) == len(ex_col_rel_pos))
        return bucketed_ex_col_rel_pos

