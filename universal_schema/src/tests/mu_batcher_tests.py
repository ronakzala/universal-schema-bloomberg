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
    elif sys.argv[1] == 'test_init_embed':
        embed_path = None
        word2idx = {'<pad>': 0, 'add': 1, 'dissolve': 2, 'alcohol': 3}
        embeds = model_utils.init_embeddings(embed_path=embed_path,
                                             word2idx=word2idx, embedding_dim=5)
        print(embeds.weight)
        print(embeds.num_embeddings, embeds.embedding_dim, embeds.padding_idx)
    else:
        sys.stderr.write('Unknown argument.\n')
