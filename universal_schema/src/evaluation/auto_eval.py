"""
Evaluate the performance of verb schema models on two tasks.
- Evaluate while treating the probability to classify a row-col pair as pos or neg
- Evaluate to check if the neg row-col pair gets placed above the pos row-col pair.
But for the sempl18 dataset the task is real because the annotations are manual
and all correct.
"""
from __future__ import unicode_literals
import os, sys, argparse
import numpy as np
from sklearn import metrics

# Use mpl on remote.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run_sempl_binclf_eval(run_path):
    """
    Compute binary classification metrics for the semantic plausibility dataset.
    :param run_path: string; directory with all model run items, contains the
        model assigned probabilities for the sempl18 dataset.
    :return: None.
    """
    probs_fname = os.path.join(run_path, 'test-sempl18-probs.npy')
    pos_neg_probs = np.load(probs_fname)
    # Make gold labels.
    y_true = np.zeros(pos_neg_probs.shape[0])
    y_true[:1448] = 1
    # Set a threshold arbitrarily for now.
    thresh = 0.5
    y_pred = np.copy(pos_neg_probs)
    y_pred[y_pred >= thresh] = 1
    y_pred[y_pred < thresh] = 0
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)
    accuracy = metrics.accuracy_score(y_true, y_pred)
    sys.stdout.write("Thresh: {:.4f}\n".format(thresh))
    sys.stdout.write("precision: {:0.4f}; recall: {:0.4f}; f1: {:0.4f} accuracy: {:0.4f}\n".
                     format(precision, recall, f1, accuracy))
    precisions, recalls, thresholds = metrics.precision_recall_curve(
        y_true=y_true, probas_pred=pos_neg_probs, pos_label=1)
    # Compute area under the precision-recall-curve for each split.
    auc = metrics.auc(recalls, precisions)
    sys.stdout.write('Area under precision-recall curve: {:0.4f}\n'.format(auc))
    # Plot for each split into the same plot.
    plt.plot(recalls, precisions, '-', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall: sempl18.')
    plt.tight_layout()
    fig_file = os.path.join(run_path, 'sempl18-precison_recall_curve.eps')
    plt.savefig(fig_file)
    plt.savefig(os.path.join(run_path, 'sempl18-precison_recall_curve.png'))
    plt.clf()
    print('Wrote: {:s}'.format(fig_file))


def run_pseudo_eval(run_path):
    """
    Run an evaluation on two tasks.
    - Calculate how the corresponding pos-neg row-col pairs are ranked wrt each other.

    - Next use the prob predicted by the model for each row-col pair as a threshold
    to say if the row-col pair is positive or negative. Plot the precision-recall
    curve for this task. This in some sense just evaluating that the training worked.

    Both the tasks are somewhat pseudo.
    :param run_path: string; path to the directory with all run items.
    :return: None.
    """
    # Calculate accuracy scores for the pseudo-disambiguation task.
    splits = ['dev', 'manual-dev']
    sys.stdout.write('Pseudo-disambiguation accuracies.\n')
    for split in splits:
        sys.stdout.write('Split: {:s}\n'.format(split))
        # For the conll data some of the split files dont exist. So skip them silently.
        try:
            neg_scores = np.load(os.path.join(run_path, '{:s}-neg-probs.npy'.format(split)))
            pos_scores = np.load(os.path.join(run_path, '{:s}-probs.npy'.format(split)))
        except IOError:
            sys.stderr.write('File does not exist. Continuing.\n')
            continue
        assert (neg_scores.shape[0] == pos_scores.shape[0])
        sys.stdout.write('Sizes: pos_scores: {}; neg_scores: {}\n'.
                         format(pos_scores.shape, neg_scores.shape))
        # You want the prob scores given to the positive examples to be higher
        # than that given to the negative examples. The two arrays are assumed
        # to be aligned.
        split_accuracy = np.mean(pos_scores > neg_scores)
        sys.stdout.write('Accuracy: {:0.4f}\n'.format(split_accuracy))

    # Calculate accuracy scores for the finer negative examples.
    testnegs = ['20-finerneg', '50-finerneg', '80-finerneg', '100-finerneg', 'neg']
    for neg in testnegs:
        sys.stdout.write('Split: {:s}\n'.format(neg))
        # For the conll data some of the split files dont exist. So skip them silently.
        try:
            neg_scores = np.load(os.path.join(run_path, 'test-{:s}-probs.npy'.format(neg)))
            pos_scores = np.load(os.path.join(run_path, 'test-probs.npy'))
        except IOError:
            sys.stderr.write('File does not exist. Continuing.\n')
            continue
        assert (neg_scores.shape[0] == pos_scores.shape[0])
        sys.stdout.write('Sizes: pos_scores: {}; neg_scores: {}\n'.
                         format(pos_scores.shape, neg_scores.shape))
        # You want the prob scores given to the positive examples to be higher
        # than that given to the negative examples. The two arrays are assumed
        # to be aligned.
        split_accuracy = np.mean(pos_scores > neg_scores)
        sys.stdout.write('Accuracy: {:0.4f}\n'.format(split_accuracy))

    # Plot the precision-recall curves for the threshold classification.
    splits = ['dev', 'manual-dev', 'test']
    for split in splits:
        # For the conll data some of the split files dont exist. So skip them silently.
        try:
            neg_y_scores = np.load(os.path.join(run_path, '{:s}-neg-probs.npy'.format(split)))
            pos_y_scores = np.load(os.path.join(run_path, '{:s}-probs.npy'.format(split)))
        except IOError:
            sys.stderr.write('File does not exist. Continuing.\n')
            continue
        assert (neg_y_scores.shape[0] == pos_y_scores.shape[0])
        neg_y_true = np.zeros(neg_y_scores.shape[0])
        pos_y_true = np.ones(neg_y_scores.shape[0])
        y_scores = np.hstack([neg_y_scores, pos_y_scores])
        y_true = np.hstack([neg_y_true, pos_y_true])
        # Compute precision and recall for every unique value of the score. This is
        # how sklearn does it but I think its pretty wasteful because I dont need
        # so many points to threshold on. I should be able to specify a set of
        # thresholds and get a set of precisions and recalls at those thresholds.
        precisions, recalls, thresholds = metrics.precision_recall_curve(
            y_true=y_true, probas_pred=y_scores, pos_label=1)
        # Compute area under the precision-recall-curve for each split.
        auc = metrics.auc(recalls, precisions)
        sys.stdout.write('Area under precision-recall curve: {:0.4f}\n'.format(auc))
        # Plot for each split into the same plot.
        sys.stdout.write('Plotting P-R Curve for split: {:s}\n'.format(split))
        plt.plot(recalls, precisions, '-', linewidth=2)

    # Label the plot at the end.
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall per Split.')
    plt.legend(splits)
    plt.tight_layout()
    fig_file = os.path.join(run_path, 'precison_recall_curve.eps')
    plt.savefig(fig_file)
    plt.savefig(os.path.join(run_path, 'precison_recall_curve.png'))
    plt.clf()
    print('Wrote: {:s}'.format(fig_file))


def plot_finerneg_accuracies(save_path, suffix):
    """
    This is a helper function that just plots all the accuracies for the different
    finer negatives across model variants. The numbers here come from running the above
    function for all variants and entering them in here manually.
    Do this better later :P
    :param save_path: string; says where you want the figures saved.
    :param suffix: string; says which composition type the results are for.
    :return: None.
    """
    # For long title text wrapping: https://stackoverflow.com/a/10634897/3262406
    from textwrap import wrap
    # Use latex while rendering text.
    plt.rc('text', usetex=True)
    # lists going from [20-finer negatives-test, 50-finer negatives-test,
    # 80-finer negatives-test, 100-finer negatives-test, random negatives-test]
    # Some point in the future this dict could be a json from disk.
    results_lstm = {'00-typevs': [0.6624, 0.7132, 0.7529, 0.7906, 0.783],
                    '01-typentvs': [0.7201, 0.7631, 0.8079, 0.8524, 0.8798],
                    '02-ltentvs': [0.6906, 0.7308, 0.7743, 0.8183, 0.8522],
                    '03-mementvs': [0.7199, 0.7611, 0.8008, 0.8412, 0.8747],
                    '04-shmementvs': [0.7046, 0.744, 0.7822, 0.8229, 0.8664]}
    results_add = {'00-typevs': [0.6547, 0.6988, 0.7307, 0.763, 0.7421],
                   '01-typentvs': [0.7828, 0.8202, 0.8507, 0.8788, 0.8766],
                   '02-ltentvs': [0.7502, 0.788, 0.8193, 0.8502, 0.8597],
                   '03-mementvs': [0.7708, 0.8093, 0.8392, 0.8688, 0.8615],
                   '04-shmementvs': [0.7358, 0.772, 0.8011, 0.83, 0.8496]}
    if suffix == 'add':
        results = results_add
        title_str = 'Test accuracy for varying levels of negative arguments for ' \
                    'model variants with additive composition of argument tuple.'
    elif suffix == 'lstm':
        results = results_lstm
        title_str = 'Test accuracy for varying levels of negative arguments for ' \
                    'model variants with LSTM composition of argument tuple.'
    else:
        sys.stderr.write('Unknown suffix: {:s}\n'.format(suffix))
        sys.exit(1)
    readable_model_names = {'00-typevs': 'types of arguments',
                            '01-typentvs': 'types followed by surface forms',
                            '02-ltentvs': 'matrix per type + entity',
                            '03-mementvs': 'matrix per type + memory per type + attention',
                            '04-shmementvs': 'matrix per type + shared memory + attention'}
    # Plot for each model in order.
    xvals = np.array(range(len(results['00-typevs'])))+1
    xtickstrs = ['20', '50', '80', '100', '100-random']
    legend_strs = []
    model_variants = results.keys()
    model_variants.sort()
    for model_str in model_variants:
        plt.plot(xvals, results[model_str], '-', linewidth=2, marker='o')
        legend_strs.append(readable_model_names[model_str])
    # Label the plot at the end.
    plt.ylim([0.6, 1])
    plt.xticks(xvals, xtickstrs)
    plt.xlabel('Percentage of negative arguments')
    plt.ylabel('Test accuracy')
    plt.title("\n".join(wrap(title_str)))
    plt.legend(legend_strs, loc=2)
    # plt.tight_layout()
    eps = os.path.join(save_path, 'finerneg_test_accuracies-{:s}.eps'.format(suffix))
    plt.savefig(eps)
    sys.stdout.write('Wrote: {}\n'.format(eps))
    png = os.path.join(save_path, 'finerneg_test_accuracies-{:s}.png'.format(suffix))
    plt.savefig(png)
    sys.stdout.write('Wrote: {}\n'.format(png))
    plt.clf()


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='subcommand',
                                       help='The action to perform.')

    # Binary clf eval for the sempl18 dataset.
    sempl_eval = subparsers.add_parser('sempl_eval')
    sempl_eval.add_argument('--run_path', required=True,
                            help='Path to directory with all run items.')

    # Run evaluations on the pseudo-disambiguation and clf threshold tasks.
    pseudo_eval = subparsers.add_parser('pseudo_eval')
    pseudo_eval.add_argument('--run_path', required=True,
                             help='Path to directory with all run items.')

    # Plot accuracies across models. This runs on local because it uses tex
    # rendering of text.
    # TODO: In future move this to a plotting source file or something. --low-pri
    plot_finerneg = subparsers.add_parser('plot_finerneg')
    plot_finerneg.add_argument('--save_path', required=True,
                               help='Path to directory to which items should get '
                                    'saved.')
    plot_finerneg.add_argument('--suffix', required=True,
                               choices=['lstm', 'add'],
                               help='Type of composition to plot for.')
    cl_args = parser.parse_args()

    if cl_args.subcommand == 'pseudo_eval':
        run_pseudo_eval(run_path=cl_args.run_path)
    elif cl_args.subcommand == 'sempl_eval':
        run_sempl_binclf_eval(run_path=cl_args.run_path)
    elif cl_args.subcommand == 'plot_finerneg':
        plot_finerneg_accuracies(save_path=cl_args.save_path, suffix=cl_args.suffix)
    else:
        sys.stderr.write('Unknown subcommand.\n')


if __name__ == '__main__':
    main()
