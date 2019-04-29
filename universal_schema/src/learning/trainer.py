"""
Train the passed model given the data and the batcher and save the best to disk.
"""
from __future__ import print_function
import sys, os
import logging
import time, copy

import numpy as np
import torch
import torch.optim as optim

import predict_utils as pu


class GenericTrainer:
    def __init__(self, model, batcher, train_size, dev_size,
                 batch_size, update_rule, num_epochs, learning_rate,
                 check_every, decay_lr_by, decay_lr_every, model_path,
                 early_stop=True, verbose=True):
        """
        A generic trainer class that defines the training procedure. Trainers
        for other models should subclass this and define the data that the models
        being trained consume.
        :param model: pytorch model.
        :param batcher: a model_utils.Batcher class.
        :param train_size: int; number of training examples.
        :param dev_size: int; number of dev examples.
        :param batch_size: int; number of examples per batch.
        :param update_rule: string;
        :param num_epochs: int; number of passes through the training data.
        :param learning_rate: float;
        :param check_every: int; check some metric on the dev set every check_every iterations.
        :param decay_lr_by: float; decay the learning rate exponentially by the following
            factor.
        :param decay_lr_every: int; decay learning rate every few iterations.
        :param model_path: string; directory to which model should get saved.
        :param early_stop: boolean;
        :param verbose: boolean;
        """
        # Book keeping
        self.verbose = verbose
        self.check_every = check_every
        self.num_train = train_size
        self.num_dev = dev_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        if self.num_train > self.batch_size:
            self.num_batches = int(np.ceil(float(self.num_train)/self.batch_size))
        else:
            self.num_batches = 1
        self.model_path = model_path  # Save model and checkpoints.
        self.total_iters = self.num_epochs*self.num_batches

        # Model, batcher and the data.
        self.model = model
        self.batcher = batcher
        self.time_per_batch = 0
        self.time_per_dev_pass = 0
        # Different trainer classes can add this based on the data that the model
        # they are training needs.
        self.train_fnames = []
        self.dev_fnames = {}

        # Optimizer args.
        self.early_stop = early_stop
        self.update_rule = update_rule
        self.learning_rate = learning_rate

        # Initialize optimizer.
        if self.update_rule == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.update_rule == 'adagrad':
            self.optimizer = optim.Adagrad(self.model.parameters(), lr=self.learning_rate)
        else:
            logging.error('Unknown update rule.')
            sys.exit(1)
        # Reduce the learning rate every few iterations.
        self.decay_lr_by = decay_lr_by
        self.decay_lr_every = decay_lr_every
        self.log_every = 5
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, mode='min', factor=0.1, patience=1,
        #     verbose=True)
        self.scheduler = optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer,
                                                          gamma=self.decay_lr_by)

        # Train statistics.
        self.loss_history = []
        self.loss_checked_iters = []
        self.dev_score_history = []
        self.dev_checked_iters = []

    def train(self):
        """
        Make num_epoch passes through the training set and train the model.
        :return:
        """
        # Pick the model with the least loss.
        best_params = self.model.state_dict()
        best_epoch, best_iter = 0, 0
        best_dev_loss = np.inf

        total_time_per_batch = 0
        total_time_per_dev = 0
        train_start = time.time()
        logging.info('num_train: {:d}; num_dev: {:d}'.format(
            self.num_train, self.num_dev))
        logging.info('Training {:d} epochs'.format(self.num_epochs))
        iteration = 0
        for epoch, ex_fnames in zip(range(self.num_epochs), self.train_fnames):
            # Initialize batcher. Shuffle one time before the start of every
            # epoch.
            epoch_batcher = self.batcher(ex_fnames=ex_fnames,
                                         num_examples=self.num_train,
                                         batch_size=self.batch_size)
            # Get the next padded and sorted training batch.
            iters_start = time.time()
            for batch_doc_ids, batch_dict in epoch_batcher.next_batch():
                batch_start = time.time()
                # Clear all gradient buffers.
                self.optimizer.zero_grad()
                # Compute objective.
                objective = self.model.forward(batch_dict=batch_dict)
                # Gradients wrt the parameters.
                objective.backward()
                # Step in the direction of the gradient.
                self.optimizer.step()
                if iteration % self.log_every == 0:
                    # Save the loss.
                    if torch.cuda.is_available():
                        loss = float(objective.data.cpu().numpy())
                    else:
                        loss = float(objective.data.numpy())
                    self.loss_history.append(loss)
                    self.loss_checked_iters.append(iteration)
                    if self.verbose:
                        logging.info('Epoch: {:d}; Iteration: {:d}/{:d}; Loss: {:.4f}'.
                                     format(epoch, iteration, self.total_iters, loss))
                    # The decay_lr_every is guaranteed to be a multiple of self.log_every
                    if iteration > 0 and iteration % self.decay_lr_every == 0:
                        # history = np.array(self.loss_history[(iteration - self.decay_lr_every)/self.log_every:
                        #                                      iteration/self.log_every])
                        # metric = np.mean(history)
                        # if self.verbose:
                        #     logging.info('Epoch: {:d}; Iteration: {:d}/{:d}; Aggregate metric: {:.4f}'.
                        #                  format(epoch, iteration, self.total_iters, metric))
                        self.scheduler.step()
                        logging.info('Decayed learning rates: {}'.
                                     format([g['lr'] for g in self.optimizer.param_groups]))
                elif self.verbose:
                    logging.info('Epoch: {:d}; Iteration: {:d}/{:d}'.
                                 format(epoch, iteration, self.total_iters))
                batch_end = time.time()
                total_time_per_batch += batch_end-batch_start
                # Check every few iterations how you're doing on the dev set.
                if iteration % self.check_every == 0 and iteration != 0 and self.early_stop:
                    if torch.cuda.is_available():
                        loss = float(objective.data.cpu().numpy())
                    else:
                        loss = float(objective.data.numpy())
                    dev_start = time.time()
                    dev_loss = pu.batched_loss(
                        model=self.model, batcher=self.batcher, batch_size=self.batch_size,
                        ex_fnames=self.dev_fnames, num_examples=self.num_dev)
                    dev_end = time.time()
                    total_time_per_dev += dev_end-dev_start
                    self.dev_score_history.append(dev_loss)
                    self.dev_checked_iters.append(iteration)
                    self.loss_history.append(loss)
                    self.loss_checked_iters.append(iteration)
                    if type(dev_loss) == torch.Tensor:
                        dev_loss = float(dev_loss.item())
                    if dev_loss < best_dev_loss:
                        best_dev_loss = dev_loss
                        # Deep copy so you're not just getting a reference.
                        best_params = copy.deepcopy(self.model.state_dict())
                        best_epoch = epoch
                        best_iter = iteration
                        everything = (epoch, iteration, self.total_iters, dev_loss)
                        if self.verbose:
                            logging.info('Current best model; Epoch {:d}; '
                                         'Iteration {:d}/{:d}; Dev loss: {:.4f}'.format(*everything))
                    else:
                        everything = (epoch, iteration, self.total_iters, dev_loss)
                        if self.verbose:
                            logging.info('Epoch {:d}; Iteration {:d}/{:d}; Dev loss: {:.4f}'.format(*everything))
                iteration += 1
            epoch_time = time.time()-iters_start
            logging.info('Epoch {:d} time: {:.4f}s'.format(epoch, epoch_time))
            logging.info('\n')

        # Update model parameters to be best params if asked to early stop; else
        # its just the final model state.
        if self.early_stop:
            logging.info('Best model; Epoch {:d}; Iteration {:d}; Dev loss: {:.4f}'
                             .format(best_epoch, best_iter, best_dev_loss))
            self.model.load_state_dict(best_params)

        # Say how long things took.
        train_time = time.time()-train_start
        logging.info('Training time: {:.4f}s'.format(train_time))
        self.time_per_batch = float(total_time_per_batch)/self.total_iters
        logging.info('Time per batch: {:.4f}s'.format(self.time_per_batch))
        if self.early_stop and self.dev_score_history:
            self.time_per_dev_pass = float(total_time_per_dev) / len(self.dev_score_history)
            logging.info('Time per dev pass: {:4f}s'.format(self.time_per_dev_pass))

        # Save the learnt model.
        # https://stackoverflow.com/a/43819235/3262406
        model_file = os.path.join(self.model_path, 'model_best.pt')
        torch.save(self.model.state_dict(), model_file)
        logging.info('Wrote: {:s}'.format(model_file))


class USTrainer(GenericTrainer):
    def __init__(self, model, data_path, batcher, train_size, dev_size,
                 batch_size, update_rule, num_epochs, learning_rate,
                 check_every, decay_lr_by, decay_lr_every, model_path, early_stop=True,
                 verbose=True):
        """
        Trainer for the universal schema style models. Uses everything from the
        generic trainer but needs specification of the data as positive and
        negative example files that the batcher knows how to read.
        :param data_path: string; directory with all the int mapped data.
        """
        GenericTrainer.__init__(self, model, batcher, train_size, dev_size,
                                batch_size, update_rule, num_epochs, learning_rate,
                                check_every, decay_lr_by, decay_lr_every, model_path,
                                early_stop, verbose)
        # Expect the presence of a directory with as many shuffled copies of the
        # dataset as there are epochs and a negative examples file.
        self.train_fnames = []
        for i in range(self.num_epochs):
            ex_fname = {
                'pos_ex_fname': os.path.join(data_path, 'shuffled_data', 'train-im-full-{:d}.json'.format(i)),
                'neg_ex_fname': os.path.join(data_path, 'shuffled_data', 'train-neg-im-full-{:d}.json'.format(i))
            }
            self.train_fnames.append(ex_fname)
        self.dev_fnames = {
            'pos_ex_fname': os.path.join(data_path, 'dev-im-full.json'),
            'neg_ex_fname': os.path.join(data_path, 'dev-neg-im-full.json')
        }
