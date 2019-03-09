"""
Call code from everywhere, read data, initialize model, train model and make
sure training is doing something meaningful.
"""
from __future__ import unicode_literals
from __future__ import print_function
import argparse, os, sys
import logging
import codecs, pprint, json

import numpy as np
import torch
import data_utils
import batchers
import trainer
from frame_models import compus

# Write unicode to stdout.
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)


def train_model(model_name, int_mapped_path, model_hparams, train_hparams,
                run_path, dataset, embedding_path=None, use_toy=False):
    """
    Read the int mapped training and dev data, initialize and train the model.
    :param model_name: string; says which model to use.
    :param int_mapped_path: string; path to the directory with shuffled_data
        and the test and dev json files.
    :param model_hparams: dict; hyperparameters for the model.
    :param train_hparams: dict; hyperparameters for the trainer.
    :param run_path: string; path to which results and model gets saved.
    :param embedding_path: string; optionally the path from which to load
        pre-trained embeddings for the rows and column elements. Unused and never passed. :/
    :param dataset: string; says which dataset it is {freebase/anyt/fbanyt}
        use freebase alone, use anyt alone or use the combination of fb and anyt.
    :param use_toy: boolean; whether you should use a small subset of examples
        to debug train.
    :return: None.
    """
    # Load word2idx maps.
    map_fname = os.path.join(int_mapped_path, 'op2idx-full.json')
    with codecs.open(map_fname, 'r', 'utf-8') as mapf:
        op2idx = json.load(mapf)
    map_fname = os.path.join(int_mapped_path, 'ent2idx-full.json')
    with codecs.open(map_fname, 'r', 'utf-8') as mapf:
        ent2idx = json.load(mapf)
    # Unpack hyperparameter settings.
    # hdim isnt always the lstm hidden dimension its unfortunate that I named it
    # that way but it refers to the dimension of the argument set.
    hdim, dropp = model_hparams['rdim'], model_hparams['dropp']
    argdim = model_hparams['argdim']
    lstm_comp = model_hparams['lstm_comp']
    # Unpack training hparams.
    bsize, epochs, lr = train_hparams['bsize'], train_hparams['epochs'], train_hparams['lr']
    decay_every, decay_by = train_hparams['decay_every'], train_hparams['decay_by']
    es_check_every = train_hparams['es_check_every']
    train_size, dev_size, test_size = train_hparams['train_size'], train_hparams['dev_size'], \
                                      train_hparams['test_size']
    if use_toy:
        train_size, dev_size, test_size = 5000, 5000, 5000
    logging.info('Model hyperparams:')
    logging.info(pprint.pformat(model_hparams))
    logging.info('Train hyperparams:')
    logging.info(pprint.pformat(train_hparams))

    # Save hyperparams to disk.
    run_info = {'model_hparams': model_hparams,
                'train_hparams': train_hparams}
    with codecs.open(os.path.join(run_path, 'run_info.json'), 'w', 'utf-8') as fp:
        json.dump(run_info, fp)

    # Initialize model.
    if model_name in ['latfeatus']:
        model = compus.CompVS(row2idx=op2idx, col2idx=ent2idx, embedding_dim=rdim,
                              hidden_dim=rdim, dropout=dropp, lstm_comp=lstm_comp)
    else:
        logging.error('Unknown model: {:s}'.format(model_name))
        sys.exit(1)
    logging.info(model)

    # Move model to the GPU.
    if torch.cuda.is_available():
        model.cuda()
        logging.info('Running on GPU.')

    # Initialize the trainer.
    batcher_cls = batchers.RCBatcher
    ustrainer = trainer.USTrainer(
        model=model, data_path=int_mapped_path, train_size=train_size,
        dev_size=dev_size, batcher=batcher_cls, batch_size=bsize,
        update_rule='adam', num_epochs=epochs, learning_rate=lr, check_every=es_check_every,
        decay_lr_by=decay_by, decay_lr_every=decay_every, model_path=run_path,
        early_stop=True)

    # Train and save the best model to model_path.
    ustrainer.train()
    # Plot training time stats.
    data_utils.plot_train_hist(ustrainer.loss_history, ustrainer.loss_checked_iters,
                               fig_path=run_path, ylabel='Batch loss')
    data_utils.plot_train_hist(ustrainer.dev_score_history,
                               ustrainer.dev_checked_iters,
                               fig_path=run_path, ylabel='Dev-set Loss')


def run_saved_model(model_name, int_mapped_path, run_path, dataset, embedding_path=None,
                    sentwordemb_path=None, use_toy=False):
    """
    Read the int training and dev data, initialize and run a saved model.
    :param model_name: string; says which model to use.
    :param int_mapped_path: string; path to the directory with shuffled_data
        and the test and dev json files.
    :param run_path: string; path to which results are saved and from where
        model gets read.
    :param dataset: string; {anyt/ms500k} used to select the set of files
        predictions are made on.
    :param embedding_path: string; optionally the path from which to load
        pre-trained embeddings.
    :param sentwordemb_path: string; path to load the pretrained embeddings for sentence
        context words from. Used in models using sentence context.
    :param use_toy: boolean; whether you should use a small subset of examples
        to debug train.
    :return: None.
    """
    # Load word2idx maps.
    map_fname = os.path.join(int_mapped_path, 'op2idx-full.json')
    with codecs.open(map_fname, 'r', 'utf-8') as mapf:
        op2idx = json.load(mapf)
    map_fname = os.path.join(int_mapped_path, 'ent2idx-full.json')
    with codecs.open(map_fname, 'r', 'utf-8') as mapf:
        ent2idx = json.load(mapf)

    # Load the hyperparams from disk.
    with codecs.open(os.path.join(run_path, 'run_info.json'), 'r', 'utf-8') as fp:
        run_info = json.load(fp)
        model_hparams = run_info['model_hparams']
        train_hparams = run_info['train_hparams']
        try:
            side_info = run_info['side_info']
        # Handle this for backward compatibility with the existing trained models.
        except KeyError:
            if model_name in ['ltentvs', 'mementvs', 'shmementvs']:
                side_info = 'types'
            elif model_name in ['rnentdepvs', 'gnentdepvs']:
                side_info = 'deps'
            else:
                side_info = None
    # Load up side info maps.
    if side_info == 'types' and model_name != 'dsentposrivs':
        map_fname = os.path.join(int_mapped_path, 'type2idx-full.json')
        with codecs.open(map_fname, 'r', 'utf-8') as mapf:
            sideinfo2idx = json.load(mapf)
    elif side_info == 'deps' and model_name != 'dsentposrivs':
        map_fname = os.path.join(int_mapped_path, 'deps2idx-full.json')
        with codecs.open(map_fname, 'r', 'utf-8') as mapf:
            sideinfo2idx = json.load(mapf)
    # Position embedding NOT part of speech.
    elif model_name in ['dsentposrivs', 'dsentposvs', 'emdsentposvsdum', 'dsentposvsdum']:
        max_arg_pos = model_hparams['max_arg_pos']
        if model_name in ['dsentposrivs']:
            # Create a side info dict on the fly. Just the argument position embeddings.
            sideinfo2idx = dict([(unicode(i), i) for i in range(max_arg_pos+4)])
            # TODO: HACK so that the batcher code doesnt try to read a 'pos' field from the files.
            side_info = 'types' if dataset == 'ms500k' else 'deps'
        elif model_name in ['dsentposvs', 'emdsentposvsdum', 'dsentposvsdum']:
            sideinfo2idx = dict([(unicode(i), i) for i in range(2*max_arg_pos + 4)])
    else:
        assert (model_name in {'naryus', 'typevs', 'typentvs', 'entvs', 'rnentvs', 'gnentvs',
                               'rnentsentvs', 'gnentsentvs', 'dsentvs', 'dsentvsgro', 'dsentrivs', 'dseventvs',
                               'dsentrivsmt', 'emdsentvs', 'emdsentvssup', 'dsentvssup',
                               'emdsentvsdum', 'dsentvsdum', 'emdsentposvsdum', 'dsentposvsdum'})
        logging.info("Not using any side information.")

    if model_name in ['rnentsentvs', 'gnentsentvs', 'rnentsisentvs']:
        map_fname = os.path.join(int_mapped_path, 'sentword2idx-full.json')
        with codecs.open(map_fname, 'r', 'utf-8') as mapf:
            sentword2idx = json.load(mapf)

    # Unpack hyperparameter settings.
    rdim, dropp = model_hparams['rdim'], model_hparams['dropp']
    # This ensures that this function can be called on the already trained models
    # which did not have this as an model hyperparameter. Those assume 'hidden'.
    lstm_comp = model_hparams['lstm_comp'] if 'lstm_comp' in model_hparams else 'hidden'
    if model_name in ['ltentvs', 'mementvs', 'shmementvs', 'rnentvs', 'gnentvs', 'rnentdepvs', 'gnentdepvs',
                      'rnentsentvs', 'gnentsentvs', 'rnentsisentvs']:
        latdim = model_hparams['latdim']
    if model_name in ['rnentvs', 'gnentvs', 'rnentdepvs', 'gnentdepvs', 'rnentsentvs', 'gnentsentvs',
                      'rnentsisentvs', 'dsentvs', 'dsentvsgro', 'dsentetvs', 'dsentrivs', 'dseventvs', 'dsentrivsmt',
                      'dsentetrivs', 'dsentposrivs', 'emdsentvs', 'emdsentvssup', 'dsentvssup',
                      'emdsentvsdum', 'dsentvsdum', 'dsentposvs', 'emdsentposvsdum', 'dsentposvsdum']:
        try:
            argdim = model_hparams['argdim']
        # Backward compatibility.
        except KeyError:
            argdim = rdim
    if model_name in ['dsentvs', 'dsentvsgro', 'dsentetvs', 'dsentrivs', 'dseventvs', 'dsentrivsmt',
                      'dsentetrivs', 'dsentposrivs', 'emdsentvs', 'emdsentvssup', 'dsentvssup',
                      'emdsentvsdum', 'dsentvsdum', 'dsentposvs', 'emdsentposvsdum', 'dsentposvsdum']:
        outargdim = model_hparams['outargdim']
    if model_name in ['dsentrivs', 'dsentetrivs', 'dsentetrivs', 'dsentposrivs', 'emdsentvs',
                      'emdsentvssup', 'dsentvssup', 'emdsentvsdum', 'dsentvsdum', 'emdsentposvsdum',
                      'dsentposvsdum']:
        preddim = model_hparams['preddim']
    if model_name in ['dsentrivsmt', 'emdsentvssup', 'dsentvssup', 'emdsentvsdum', 'dsentvsdum',
                      'emdsentposvsdum', 'dsentposvsdum']:
        lprop = model_hparams['lprop']
    if model_name in ['dsentetvs', 'dsentposvs', 'emdsentposvsdum', 'dsentposvsdum']:
        si_dim = model_hparams['si_dim']
    if model_name in ['dsentetrivs', 'dsentposrivs']:
        si_dim = model_hparams['si_dim']
        si_role_dim = model_hparams['si_role_dim']
    if model_name in ['gnentvs', 'gnentdepvs', 'gnentsentvs']:
        edgenodedim = model_hparams['edgenodedim']
    if model_name in ['rnentdepvs', 'gnentdepvs']:
        si_role_dim = model_hparams['si_role_dim']
        # The side info dimension is the same as the number of items in the map.
        # padding, start stop included for now.
        si_dim = len(sideinfo2idx)
    if model_name in ['rnentsentvs', 'gnentsentvs']:
        sentword_dim = model_hparams['sentword_dim']
        si_role_dim = model_hparams['si_role_dim']
        si_dim = model_hparams['si_dim']
    if model_name in ['rnentsisentvs']:
        sentword_dim = model_hparams['sentword_dim']
        sentcon_dim = model_hparams['sentcon_dim']
        part_si_dim = len(sideinfo2idx)
        # The sentence context and the column element side info form the full side info.
        si_dim = sentcon_dim + part_si_dim
        si_role_dim = model_hparams['si_role_dim']

    bsize, epochs, lr = train_hparams['bsize'], train_hparams['epochs'], train_hparams['lr']
    train_size, dev_size, test_size = train_hparams['train_size'], train_hparams['dev_size'], \
                                      train_hparams['test_size']

    # Using toy might be im-perfect because this is saying read the first set N
    # examples from the per-epoch train shuffled train file. So you're actually
    # training on more than N train examples in total, different N in each epoch.
    if use_toy:
        train_size, dev_size, test_size = 5000, 5000, 5000
    logging.info('Model hyperparams:')
    logging.info(pprint.pformat(model_hparams))
    logging.info('Train hyperparams:')
    logging.info(pprint.pformat(train_hparams))

    # Initialize model.
    if model_name in ['typevs', 'typentvs', 'entvs']:
        model = compus.CompVS(row2idx=op2idx, col2idx=ent2idx, embedding_dim=rdim,
                              hidden_dim=rdim, dropout=dropp, lstm_comp=lstm_comp)
    else:
        logging.error('Unknown model: {:s}'.format(model_name))
        sys.exit(1)
    logging.info(model)
    model_file = os.path.join(run_path, 'model_best.pt')
    model.load_state_dict(torch.load(model_file))

    # Save embeddings to disk.
    learnt_row_embedding = model.row_embeddings.weight.data.numpy()
    learnt_col_embedding = model.col_embeddings.weight.data.numpy()
    logging.info('Learnt row embeddings shape: {}'.format(learnt_row_embedding.shape))
    logging.info('Learnt col embeddings shape: {}'.format(learnt_col_embedding.shape))
    with open(os.path.join(run_path, 'learnt_row_embeddings.npy'), 'w') as fp:
        np.save(fp, learnt_row_embedding)
        logging.info('Wrote: {}'.format(fp.name))
    with open(os.path.join(run_path, 'learnt_col_embeddings.npy'), 'w') as fp:
        np.save(fp, learnt_col_embedding)
        logging.info('Wrote: {}'.format(fp.name))

    # Move model to the GPU.
    if torch.cuda.is_available():
        model.cuda()
        logging.info('Running on GPU.\n')

    # Say how many examples a prediction was made for.
    ev_train_size = train_size
    ev_dev_size = dev_size
    ev_test_size = test_size
    eval_sizes = {'ev_train_size': ev_train_size, 'ev_dev_size': ev_dev_size,
                  'ev_test_size': ev_test_size}
    run_info = {'model_hparams': model_hparams,
                'train_hparams': train_hparams,
                'eval_hparams': eval_sizes,
                'side_info': side_info}
    with codecs.open(os.path.join(run_path, 'run_info.json'), 'w', 'utf-8') as fp:
        json.dump(run_info, fp)
    # Make predictions on all splits and write them to disk incrementally.
    batcher_cls = batchers.RCBatcher
    data_utils.make_predictions(int_mapped_path=int_mapped_path, batcher=batcher_cls, model=model,
                                dataset=dataset, result_path=run_path, rep_size=rdim, batch_size=bsize,
                                train_size=ev_train_size, dev_size=ev_dev_size, test_size=ev_test_size)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='subcommand',
                                       help='The action to perform.')
    # Train the model.
    train_args = subparsers.add_parser('train_model')
    # Where to get what.
    train_args.add_argument('--model_name', required=True,
                            choices=['latfeatus'],
                            help='The name of the model to train.')
    train_args.add_argument('--dataset', required=True,
                            choices=['freebase', 'anyt', 'fbanyt'],
                            help='The dataset predictions are being made on.')
    train_args.add_argument('--int_mapped_path', required=True,
                            help='Path to the int mapped dataset.')
    train_args.add_argument('--run_path', required=True,
                            help='Path to directory to save all run items to.')
    train_args.add_argument('--log_fname',
                            help='File name for the log file to which logs get'
                                 ' written.')
    train_args.add_argument('--train_size', required=True, type=int,
                            help='Number of training examples.')
    train_args.add_argument('--dev_size', required=True, type=int,
                            help='Number of dev examples.')
    train_args.add_argument('--test_size', required=True, type=int,
                            help='Number of test examples.')
    # Model hyper-parameters.
    train_args.add_argument('--rdim', required=True, type=int,
                            help='Dimension of relation embeddings.')
    train_args.add_argument('--argdim', type=int,
                            help='Dimensions of the argument embeddings.')
    train_args.add_argument('--dropp', required=True, type=float,
                            help='Dropout probability.')
    # For the rnentvs this is useless. But require it anyway for potential future use.
    train_args.add_argument('--lstm_comp', required=True,
                            choices=['max', 'hidden', 'add', 'predrn', 'deepset'],
                            help='Says how the col representation should be generated. Even '
                                 'though its called lstm_comp it can say any kind of composition.'
                                 'Keeping it this way for backward compatibility :(')
    # Training hyper-parameters.
    train_args.add_argument('--bsize', required=True, type=int,
                            help='Batch size.')
    train_args.add_argument('--epochs', required=True, type=int,
                            help='Number of passes to make through the dataset'
                                 'when training.')
    train_args.add_argument('--lr', required=True, type=float,
                            help='Learning rate.')
    train_args.add_argument('--decay_by', required=True, type=float,
                            help='Learning rate decay multiplicative factor.')
    train_args.add_argument('--decay_every', required=True, type=int,
                            help='Decay learning rate every few iterations.')
    train_args.add_argument('--es_check_every', required=True, type=int,
                            help='Early stop checking frequency.')
    train_args.add_argument('--use_toy', choices=['true', 'false'],
                            default='false',
                            help='Whether to use a small subset of examples to '
                                 'debug training.')

    # Run a saved model.
    savedrun_args = subparsers.add_parser('run_saved_model')
    # Where to get what.
    savedrun_args.add_argument('--dataset', required=True,
                               choices=['freebase', 'anyt', 'fbanyt'],
                               help='The dataset predictions are being made on.')
    savedrun_args.add_argument('--int_mapped_path', required=True,
                               help='Path to the int mapped dataset.')
    savedrun_args.add_argument('--model_name', required=True,
                               choices=['latfeatus'],
                               help='The name of the model to train.')
    savedrun_args.add_argument('--run_path', required=True,
                               help='Path to directory with all run items.')
    savedrun_args.add_argument('--log_fname',
                               help='File name for the log file to which logs get '
                                    'written.')
    savedrun_args.add_argument('--use_toy', choices=['true', 'false'],
                               default='false',
                               help='Whether to use a small subset of examples '
                                    'to debug training.')
    cl_args = parser.parse_args()
    # If a log file was passed then write to it.
    try:
        logging.basicConfig(level='INFO', format='%(message)s',
                            filename=cl_args.log_fname)
        # Print the called script and its args to the log.
        logging.info(' '.join(sys.argv))
    # Else just write to stdout.
    except AttributeError:
        logging.basicConfig(level='INFO', format='%(message)s',
                            stream=sys.stdout)
        # Print the called script and its args to the log.
        logging.info(' '.join(sys.argv))

    if cl_args.subcommand == 'train_model':
        model_hparams, train_hparams = {}, {}
        model_hparams['rdim'], model_hparams['dropp'], model_hparams['lstm_comp'] = \
            cl_args.rdim, cl_args.dropp, cl_args.lstm_comp
        model_hparams['argdim'] = cl_args.argdim
        train_hparams['bsize'], train_hparams['epochs'], train_hparams['lr'] = \
            cl_args.bsize, cl_args.epochs, cl_args.lr
        train_hparams['decay_every'] = cl_args.decay_every
        train_hparams['es_check_every'] = cl_args.es_check_every
        train_hparams['decay_by'] = cl_args.decay_by
        # Consider dataset sizes to be train variables as well.
        train_hparams['train_size'], train_hparams['dev_size'], train_hparams['test_size'] = \
            cl_args.train_size, cl_args.dev_size, cl_args.test_size
        use_toy = True if cl_args.use_toy == 'true' else False
        train_model(model_name=cl_args.model_name,
                    int_mapped_path=cl_args.int_mapped_path,
                    run_path=cl_args.run_path,
                    model_hparams=model_hparams,
                    train_hparams=train_hparams,
                    use_toy=use_toy,
                    dataset=cl_args.dataset)
    elif cl_args.subcommand == 'run_saved_model':
        use_toy = True if cl_args.use_toy == 'true' else False
        run_saved_model(model_name=cl_args.model_name,
                        int_mapped_path=cl_args.int_mapped_path,
                        run_path=cl_args.run_path,
                        use_toy=use_toy,
                        dataset=cl_args.dataset)


if __name__ == '__main__':
    main()
