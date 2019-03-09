"""
Tests for functions in model_utils and batcher.
"""
from __future__ import print_function
from __future__ import unicode_literals
import os
import sys
import codecs
import pprint
# Add the upper level directory to the path.
# This is hack but its fine for now I guess.: https://stackoverflow.com/a/7506029/3262406
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from learning import batchers, predict_utils
from learning.models_common import model_utils

# Write unicode to stdout.
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)


if __name__ == '__main__':
    if sys.argv[1] == 'test_rcpad_sort':
        test_row = [[1], [2], [3], [4], [5]]
        test_col = [[2, 21, 3], [2, 1, 22, 31, 3], [1, 31, 4, 1, 52, 3], [2, 4, 3], [2, 5, 9, 10, 11, 12, 3]]
        print(test_row)
        print(test_col)
        batch_dict = batchers.RCBatcher.pad_sort_batch(raw_feed={
            'im_row_raw': test_row,
            'im_col_raw': test_col
        })
        pprint.pprint(batch_dict)

        test_row = [[2, 1, 1, 3], [2, 2, 1, 3], [2, 3, 3, 4, 3], [2, 4, 3], [2, 5, 3]]
        test_col = [[2, 1, 2, 3], [2, 1, 2, 3, 3], [2, 3, 4, 1, 5, 3], [2, 4, 3], [2, 5, 9, 10, 11, 12, 3]]
        print(test_row)
        print(test_col)
        batch_dict = batchers.RCBatcher.pad_sort_batch(raw_feed={
            'im_row_raw': test_row,
            'im_col_raw': test_col
        })
        pprint.pprint(batch_dict)
    elif sys.argv[1] == 'test_rctypepad_sort':
        test_row = [[1], [2], [3], [4], [5]]
        test_col = [[2, 1, 2, 3], [2, 1, 2, 3, 3], [2, 3, 4, 1, 5, 3], [2, 4, 3], [2, 5, 9, 10, 11, 12, 3]]
        test_col_types = [[2, 1, 2, 3], [2, 1, 2, 3, 3], [2, 3, 4, 1, 5, 3], [2, 4, 3], [2, 5, 9, 10, 11, 12, 3]]
        print(test_row)
        print(test_col)
        batch_dict = batchers.RCLatTypeBatcher.pad_sort_batch(raw_feed={
            'im_row_raw': test_row,
            'im_col_raw': test_col,
            'im_col_types': test_col_types
        }, side_info='types')
        pprint.pprint(batch_dict)

        test_row = [[2, 1, 1, 3], [2, 2, 1, 3], [2, 3, 3, 4, 3], [2, 4, 3], [2, 5, 3]]
        test_col = [[2, 1, 2, 3], [2, 1, 2, 3, 3], [2, 3, 4, 1, 5, 3], [2, 4, 3], [2, 5, 9, 10, 11, 12, 3]]
        test_col_types = [[2, 1, 2, 3], [2, 1, 2, 3, 3], [2, 3, 4, 1, 5, 3], [2, 4, 3], [2, 5, 9, 10, 11, 12, 3]]
        print(test_row)
        print(test_col)
        batch_dict = batchers.RCLatTypeBatcher.pad_sort_batch(raw_feed={
            'im_row_raw': test_row,
            'im_col_raw': test_col,
            'im_col_types': test_col_types
        }, side_info='types')
        pprint.pprint(batch_dict)
    elif sys.argv[1] == 'test_rcsconbatcher':
        int_mapped_path = sys.argv[2]
        dev_path = os.path.join(int_mapped_path, 'dev-im-full.json')
        test_batcher = batchers.RCSentConBatcher(ex_fnames={'pos_ex_fname':dev_path},
                                                 num_examples=5, batch_size=5)
        num_batches = 0
        for docids, batch_dict in test_batcher.next_batch():
            cr = batch_dict['batch_cr']
            num_batches += 1
            print(docids[:3])
            print(cr['row'])
            print(cr['col'])
            print(cr['col_scon_flat'].numpy())
            print(cr['col_scon_flat_lens'])
            print(cr['sorted_scon_flat_refs'])
        print(num_batches)
        print(test_batcher.num_batches)
    elif sys.argv[1] == 'test_script_sort':
        test_seq = [[4, 5, 6], [29, 10, 200, 4000], [28, 10472, 39, 9, 10], [30, 39, 492]]
        print(test_seq)
        batch_dict = batchers.ScriptLMBatcher.pad_sort_batch(raw_feed={
            'im_raw': test_seq,
        })
        pprint.pprint(batch_dict)
    elif sys.argv[1] == 'test_rcbatcher':
        int_mapped_path = sys.argv[2]
        ex_fnames = {
            'pos_ex_fname': os.path.join(int_mapped_path, 'dev-im-full.json'),
            'neg_ex_fname': os.path.join(int_mapped_path, 'dev-im-full.json')
        }
        test_batcher = batchers.RCBatcher(ex_fnames=ex_fnames, num_examples=20,
                                          batch_size=4)
        num_batches = 0
        for docids, batch in test_batcher.next_batch():
            num_batches += 1
            print(docids)
            print(batch)
        print(num_batches)
        print(test_batcher.num_batches)
    elif sys.argv[1] == 'test_rcposbatcher':
        int_mapped_path = sys.argv[2]
        ex_fnames = {
            'pos_ex_fname': os.path.join(int_mapped_path, 'dev-im-full.json'),
            'neg_ex_fname': os.path.join(int_mapped_path, 'dev-neg-im-full.json')
        }
        test_batcher = batchers.RCPositionBatcher(ex_fnames=ex_fnames, num_examples=20,
                                                  batch_size=4)
        test_batcher.max_arg_pos = 20
        num_batches = 0
        for docids, batch in test_batcher.next_batch():
            num_batches += 1
            print(docids)
            print(batch)
        print(num_batches)
        print(test_batcher.num_batches)
    elif sys.argv[1] == 'test_scriptbatcher':
        int_mapped_path = sys.argv[2]
        ex_fnames = {
            'ex_fname': os.path.join(int_mapped_path, 'test-debug-im-full.json')
        }
        test_batcher = batchers.ScriptLMBatcher(ex_fnames=ex_fnames, num_examples=6,
                                                batch_size=2)
        num_batches = 0
        for docids, batch in test_batcher.next_batch():
            num_batches += 1
            print(docids)
            # Print the doi of the longest sequence. per batch.
            # print(docids[batch['sorted_seqrefs'][0]])
            print(batch['in_seq'].numpy())
        print(num_batches)
        print(test_batcher.num_batches)
    elif sys.argv[1] == 'test_scriptrcbatcher':
        int_mapped_path = sys.argv[2]
        ex_fnames = {
            'pos_ex_fname': os.path.join(int_mapped_path, 'test-debug-im-full.json'),
            'neg_ex_fname': os.path.join(int_mapped_path, 'test-debug-im-full.json')
        }
        test_batcher = batchers.RC_ScriptLMBatcher(ex_fnames=ex_fnames, num_examples=6,
                                                   batch_size=2)
        num_batches = 0
        for docids, batch_mt in test_batcher.next_batch():
            num_batches += 1
            print(docids)
            # Print the doi of the longest sequence. per batch.
            # print(docids[batch['sorted_seqrefs'][0]])
            print(batch_mt['batch_lm'].keys())
            print(batch_mt['batch_vs'].keys())
        print(num_batches)
        print(test_batcher.num_batches)
    elif sys.argv[1] == 'test_multinegbatcher':
        int_mapped_path = sys.argv[2]
        ex_fnames = {
            'pos_ex_fname': os.path.join(int_mapped_path, 'test-debug-im-full.json'),
            'neg_ex_fname': os.path.join(int_mapped_path, 'test-neg-debug-im-full.json')
        }
        test_batcher = batchers.MultiNegSIBatcher(ex_fnames=ex_fnames, num_examples=20,
                                                  batch_size=4)
        test_batcher.ignore_ss = False
        num_batches = 0
        for docids, batch in test_batcher.next_batch():
            num_batches += 1
            print(docids)
            print(batch['batch_cr'])
            print(batch['batch_neg'])
        print(num_batches)
        print(test_batcher.num_batches)
    elif sys.argv[1] == 'test_embatcher':
        int_mapped_path = sys.argv[2]
        ex_fnames = {
            'pos_ex_fname': os.path.join(int_mapped_path, 'test-im-full.json'),
        }
        test_batcher = batchers.EMBatcher(ex_fnames=ex_fnames, num_examples=20,
                                          batch_size=4)
        num_batches = 0
        for docids, batch in test_batcher.next_batch():
            num_batches += 1
            print(docids)
            print(batch)
        print(num_batches)
        print(test_batcher.num_batches)
    elif sys.argv[1] == 'test_emposbatcher':
        int_mapped_path = sys.argv[2]
        ex_fnames = {
            'pos_ex_fname': os.path.join(int_mapped_path, 'test-im-full.json'),
        }
        test_batcher = batchers.EMPositionBatcher(ex_fnames=ex_fnames, num_examples=20,
                                                  batch_size=4)
        test_batcher.max_arg_pos = 20
        num_batches = 0
        for docids, batch in test_batcher.next_batch():
            num_batches += 1
            print(docids)
            print(batch)
        print(num_batches)
        print(test_batcher.num_batches)
    elif sys.argv[1] == 'test_emsupbatcher':
        int_mapped_path = sys.argv[2]
        ex_fnames = {
            'pos_ex_fname': os.path.join(int_mapped_path, 'test-im-full.json')
        }
        test_batcher = batchers.EMSupBatcher(ex_fnames=ex_fnames, num_examples=20,
                                             batch_size=4)
        num_batches = 0
        for docids, batch in test_batcher.next_batch():
            num_batches += 1
            print(docids)
            print(batch)
        print(num_batches)
        print(test_batcher.num_batches)
    elif sys.argv[1] == 'test_emdumbatcher':
        int_mapped_path = sys.argv[2]
        ex_fnames = {
            'pos_ex_fname': os.path.join(int_mapped_path, 'test-im-full.json'),
            'neg_ex_fname': os.path.join(int_mapped_path, 'test-neg-im-full.json')
        }
        test_batcher = batchers.EMDumBatcher(ex_fnames=ex_fnames, num_examples=20,
                                             batch_size=4)
        num_batches = 0
        for docids, batch in test_batcher.next_batch():
            num_batches += 1
            print(docids)
            print(batch)
        print(num_batches)
        print(test_batcher.num_batches)
    elif sys.argv[1] == 'test_supbatcher':
        int_mapped_path = sys.argv[2]
        ex_fnames = {
            'pos_ex_fname': os.path.join(int_mapped_path, 'test-im-full.json'),
            'neg_ex_fname': os.path.join(int_mapped_path, 'test-neg-im-full.json')
        }
        test_batcher = batchers.SupBatcher(ex_fnames=ex_fnames, num_examples=20,
                                           batch_size=4)
        num_batches = 0
        for docids, batch in test_batcher.next_batch():
            num_batches += 1
            print(docids)
            print(batch)
        print(num_batches)
        print(test_batcher.num_batches)
    elif sys.argv[1] == 'test_init_embed':
        embed_path = None
        word2idx = {'<pad>': 0, 'add': 1, 'dissolve': 2, 'alcohol': 3}
        embeds = model_utils.init_embeddings(embed_path=embed_path,
                                             word2idx=word2idx, embedding_dim=5)
        print(embeds.weight)
        print(embeds.num_embeddings, embeds.embedding_dim, embeds.padding_idx)
    elif sys.argv[1] == 'test_init_lt':
        import torch
        from torch.autograd import Variable
        t2i = {'amt_unit': 0, 'amt_misc': 1, 'cnd_unit': 2, 'cnd_misc': 3,
               'material': 4, 'target': 5, 'descriptor': 6}
        e = model_utils.init_latent_types(type2idx=t2i, latent_width=2, embedding_dim=5)
        print(e)
        t = Variable(torch.LongTensor([1, 2, 3]))
        s = e(t)
        print(s)
        print(e.weight.data.numpy())
    elif sys.argv[1] == 'test_unpad_unsort':
        test_row = [[1], [2], [3], [4], [5]]
        test_col = [[1, 2], [1, 2, 3], [3, 4, 1, 5], [4], [5, 9, 10, 11, 12]]
        print(test_row)
        print(test_col)
        batch_dict = batchers.RCBatcher.pad_sort_batch(raw_feed={
            'im_row_raw': test_row,
            'im_col_raw': test_col
        })
        pprint.pprint(batch_dict)
        print(predict_utils.unpad_unsort_batch(batch_dict['col'],
                                               batch_dict['col_lens'],
                                               batch_dict['sorted_colrefs']))
        print(predict_utils.unpad_unsort_batch(batch_dict['row'],
                                               batch_dict['row_lens'],
                                               batch_dict['sorted_rowrefs']))
    else:
        sys.stderr.write('Unknown argument.\n')
