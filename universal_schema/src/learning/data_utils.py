"""
General utilities; reading files and such.
"""
from __future__ import unicode_literals
from __future__ import print_function
import os, sys
import logging
import time
import itertools
import codecs, json

# Use mpl on remote.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tables
import predict_utils as mu


# Write unicode to stdout.
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)


def read_json(json_file):
    """
    Read per line JSON and yield.
    :param json_file: File-like with a next() method.
    :return: yield one json object.
    """
    for json_line in json_file:
        # Try to manually skip bad chars.
        # https://stackoverflow.com/a/9295597/3262406
        try:
            f_dict = json.loads(json_line.replace('\r\n', '\\r\\n'),
                                encoding='utf-8')
            yield f_dict
        # Skip case which crazy escape characters.
        except ValueError:
            raise


def plot_train_hist(y_vals, checked_iters, fig_path, ylabel):
    """
    Plot y_vals against the number of iterations.
    :param y_vals: list; values along the y-axis.
    :param checked_iters: list; len(y_vals)==len(checked_iters); the iterations
        the values in y_vals correspond to.
    :param fig_path: string; the directory to write the plots to.
    :param ylabel: string; the label for the y-axis.
    :return: None.
    """
    # If there is nothing to plot just return.
    if len(checked_iters) <= 3:
        return
    x_vals = np.array(checked_iters)
    y_vals = np.vstack(y_vals)
    plt.plot(x_vals, y_vals, '-', linewidth=2)
    plt.xlabel('Training iteration')
    plt.ylabel(ylabel)
    plt.title('Evaluated every: {:d} iterations'.format(
        checked_iters[1]-checked_iters[0]))
    plt.tight_layout()
    ylabel = '_'.join(ylabel.lower().split())
    fig_file = os.path.join(fig_path, '{:s}_history.eps'.format(ylabel))
    plt.savefig(fig_file)
    plt.savefig(os.path.join(fig_path, '{:s}_history.png'.format(ylabel)))
    plt.clf()
    logging.info('Wrote: {:s}'.format(fig_file))


#####################################################################
#              Utilities to do io for predictions.                  #
#####################################################################
def make_predictions(int_mapped_path, dataset, batcher, model, result_path, rep_size,
                     batch_size, train_size, dev_size, test_size):
    """
    Make predictions on passed data in batches with the model and writing to disk
    batchwise.
    :param int_mapped_path: string; path to the directory with shuffled_data
        and the test and dev json files.
    :param dataset: string; {anyt/ms500k} says which dataset the model must use.
    :param batcher: a model_utils.Batcher class.
    :param model: pytorch model.
    :param result_path: string; path to which results get saved.
    :param rep_size: int; size of the hidden representations to save.
    :param batch_size: int; number of examples per batch.
    :param train_size: int; number of training examples.
    :param dev_size: int; number of dev examples.
    :param test_size: int; number of test examples.
    :return: None.
    """
    # Make predictions on the specified data. This will simply write a file with the
    # model assigned probabilities to the triples in the file specified below.
    if dataset == 'freebase':
        raise NotImplementedError
        # TODO: Place your test time splits here. <size, fname, non-int mapped fname>.
        split_fnames = [(1173075, 'ep-test-llpdepcandidates-im-full', 'ep-test-llpdepcandidates'),
                        (110663, 'ep-dev-llpdepcandidates-im-full', 'ep-dev-llpdepcandidates')]
    else:
        raise ValueError("Unknown dataset: {:s}".format(dataset))
    for num_examples, fname, split_str in split_fnames:
        start = time.time()
        # Point to the int mapped data to make predictions on.
        ex_fname = {
            'pos_ex_fname': os.path.join(int_mapped_path, fname) + '.json'
        }
        logging.info('Making predictions on: {:s}'.format(ex_fname['pos_ex_fname']))
        # Write predictions to disk while also writing the original data.
        # For the sempl data some of the split files dont exist. So skip them silently.
        try:
            raw_ex_file = read_json(codecs.open(os.path.join(int_mapped_path, split_str + '.json'), 'r', 'utf-8'))
        except IOError:
            logging.warning('Raw-file does not exist. Continuing.')
            continue
        batch_preds = mu.batched_predict(
            model=model, batcher=batcher, batch_size=batch_size,
            ex_fnames=ex_fname, num_examples=num_examples)
        # Write learnt reps to h5 file incrementally.
        col_rep_file = tables.open_file(os.path.join(result_path, split_str + '-col_rep.h5'), mode='w')
        col_atom = tables.Float64Atom()
        col_array = col_rep_file.create_earray(col_rep_file.root, 'data', col_atom, (0, rep_size))

        row_rep_file = tables.open_file(os.path.join(result_path, split_str + '-row_rep.h5'), mode='w')
        row_atom = tables.Float64Atom()
        row_array = row_rep_file.create_earray(row_rep_file.root, 'data', row_atom, (0, rep_size))
        # Write the scores for each example to a per-line json file.
        pred_file = codecs.open(os.path.join(result_path, split_str + '-probs.json'), 'w', 'utf-8')
        for batch_doc_ids, (batch_probs, batch_col_rep, batch_row_rep) in batch_preds:
            # Append to the reps.
            col_array.append(batch_col_rep)
            row_array.append(batch_row_rep)
            # Write prob predictions to readable files.
            for i, (doc_id, raw_dict) in enumerate(itertools.izip(batch_doc_ids, raw_ex_file)):
                if raw_dict != {}:
                    assert (doc_id == raw_dict['doc_id'])
                    pred_dict = raw_dict
                    pred_dict['prob'] = float(batch_probs[i])
                    r = json.dumps(pred_dict)
                    pred_file.write(r + '\n')
        logging.info('Prediction time: {:.4f}s'.format(time.time() - start))
        pred_file.close()
        col_rep_file.close()
        row_rep_file.close()
        logging.info('Wrote: {:s}'.format(pred_file.name))
        logging.info('Wrote: {:s}'.format(os.path.join(result_path, split_str + '-col_rep.h5')))
        logging.info('Wrote: {:s}'.format(os.path.join(result_path, split_str + '-row_rep.h5')))
    logging.info('\n')
    # Since I am not doing disambiguation evaluation anymore I am commenting these out.
    # prob_negs(int_mapped_path, batcher, model, result_path, rep_size,
    #           batch_size, train_size, dev_size, test_size)
    # # Turn the predicted scores into numpy arrays.
    # prob_json2numpy(result_path)


def prob_negs(int_mapped_path, batcher, model, result_path, rep_size,
              batch_size, train_size, dev_size, test_size):
    """
    This is a utility function that must be called and only by the
    make_predictions* functions.
    """
    raise NotImplementedError
    # Make predictions on the negative example test, dev and train sets so you
    # can draw prc curves.
    split_fnames = [(132, 'manual-dev-neg-im-full', 'manual-dev-neg'),
                    (dev_size, 'dev-neg-im-full', 'dev-neg'),
                    (test_size, 'test-neg-im-full', 'test-neg'),
                    (test_size, 'test-20-finerneg-im-full', 'test-20-finerneg'),
                    (test_size, 'test-50-finerneg-im-full', 'test-50-finerneg'),
                    (test_size, 'test-80-finerneg-im-full', 'test-80-finerneg'),
                    (test_size, 'test-100-finerneg-im-full', 'test-100-finerneg')]
    # Unusable in multitask models because it needs specification
    # of the number of chains in 1 million.
    # (1000000, 'test-500-ranking-im-full', 'test-500-ranking')]

    for num_examples, fname, split_str in split_fnames:
        start = time.time()
        # Point to the int mapped data to make predictions on.
        ex_fname = {
            'pos_ex_fname': os.path.join(int_mapped_path, fname) + '.json'
        }
        logging.info('Making predictions on: {:s}'.format(ex_fname['pos_ex_fname']))
        batch_preds = mu.batched_predict(
            model=model, batcher=batcher, batch_size=batch_size,
            ex_fnames=ex_fname, num_examples=num_examples)
        # Write predictions to disk while also writing the original data.
        # For the conll data some of the split files dont exist. So skip them silently.
        try:
            raw_ex_file = read_json(codecs.open((os.path.join(int_mapped_path, split_str) + '.json'), 'r', 'utf-8'))
        except IOError:
            logging.warning('Raw-file does not exist. Continuing.')
            continue
        # Write the scores for each example to a per-line json file.
        pred_file = codecs.open(os.path.join(result_path, split_str + '-probs.json'), 'w', 'utf-8')
        # batch_pred_items can include role reps or not. The first item is always probs though.
        for batch_doc_ids, batch_pred_items in batch_preds:
            batch_probs = batch_pred_items[0]
            for i, (doc_id, raw_dict) in enumerate(itertools.izip(batch_doc_ids, raw_ex_file)):
                if raw_dict != {}:
                    assert (doc_id == raw_dict['doc_id'])
                    pred_dict = raw_dict
                    pred_dict['prob'] = float(batch_probs[i])
                    r = json.dumps(pred_dict)
                    pred_file.write(r + '\n')
        logging.info('Prediction time: {:.4f}s'.format(time.time() - start))
        pred_file.close()
        logging.info('Wrote: {:s}'.format(pred_file.name))


def prob_json2numpy(run_path):
    """
    Read the per-line-readable-json file with predicted probs for the positive
    and negative examples, convert the scores to numpy arrays and write them to
    disk.
    :param run_path: string; path to the directory with all run items.
    :return: None.
    """
    splits = ['dev', 'test', 'dev-neg', 'test-neg',
              'manual-dev', 'manual-dev-neg', 'test-20-finerneg', 'test-50-finerneg',
              'test-80-finerneg', 'test-100-finerneg', 'test-sempl18']
    # Build a numpy array of probs.
    for split in splits:
        pos_prob_file = os.path.join(run_path, '{:s}-probs.json'.format(split))
        pos_prob_npfile = os.path.join(run_path, '{:s}-probs.npy'.format(split))
        if os.path.isfile(pos_prob_npfile):
            logging.info('Exists: {:s}'.format(pos_prob_npfile))
            continue
        logging.info('Processing: {:s}'.format(pos_prob_file))
        prob_arr = []
        # For the conll data some of the split files dont exist. So skip them silently.
        try:
            with codecs.open(pos_prob_file, 'r', 'utf-8') as fp:
                count = 0
                for json_line in fp:
                    res_dict = json.loads(json_line.replace('\r\n', '\\r\\n'), encoding='utf-8')
                    prob_arr.append(res_dict['prob'])
                    if count % 100000 == 0 and count != 0:
                        logging.info('Example: {:d}'.format(count))
                    count += 1
        except IOError:
            logging.warning('Raw-file does not exist. Continuing.')
            continue
        # Write out the numpy array of probs.
        prob_arr = np.array(prob_arr)
        with open(pos_prob_npfile, 'w') as fp:
            np.save(fp, prob_arr)
            logging.info('Wrote: {:s}\n'.format(pos_prob_npfile))


if __name__ == '__main__':
    if sys.argv[1] == 'test_plot_hist':
        plot_train_hist([1,2,3,4], checked_iters=[100,200,300,400],
                        fig_path=sys.argv[2], ylabel='test')
    else:
        sys.stderr.write('Unknown argument.')