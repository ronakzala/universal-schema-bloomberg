"""
Miscellaneous utilities to read and work with the json files and such.
Stuff multiple functions use.
"""
from __future__ import unicode_literals
from __future__ import print_function
import sys
import os
import errno
import re
import codecs, json, random
import copy

import pp_settings as pps


class DirIterator:
    def __init__(self, in_path, doi_file, dataset='ms500k', out_path=None,
                 args=None, max_count=None, safe_dois=False):
        """
        Generator over the file names. Typically consumed by the map_unordered
        executable which map_unordered would run.
        :param in_path: string; the directory with the jsons for each paper.
        :param doi_file: string; the list/file object with dois for the papers
            in in_path.
        :param dataset: string; says which dataset this is. Changes out in-files
            are processed.
        :param out_path: string; the directory to which the output jsons should
            get written.
        :param args: tuple; the set of arguments to be returned with each in_file
            and out_file name. This could be the set of arguments which the
            callable consuming the arguments might need. This needs to be fixed
            however.
        :param max_count: int; how many items to yield.
        :param safe_dois: boolean; If the dois passed are safe filenames or not.
        :returns:
            tuple:
                (in_paper,): a path to a file to open and do things with.
                (in_paper, out_paper): paths to input file and file to write
                    processed content to.
                (in_paper, args): set of arguments to the function
                    doing the processing.
                (in_paper, out_paper, args): set of arguments to the function
                    doing the processing.
        """
        self.dataset = dataset
        self.in_path = in_path
        self._return_outpath = False if out_path == None else True
        self.out_path = out_path
        self.doi_file = doi_file
        self.optional_args = args
        self.max_count = max_count
        self.safe_dois = safe_dois

    def __iter__(self):
        count = 0
        for doi in self.doi_file:
            if self.max_count:
                if count >= self.max_count:
                    raise StopIteration
            doi = doi.strip()
            if doi[0] == '#':
                continue
            if self.dataset == 'ms500k':
                if self.safe_dois:
                    in_paper = os.path.join(self.in_path, doi.strip())
                else:
                    in_paper = os.path.join(self.in_path, re.sub('/', '-', doi) + '.json')
            elif self.dataset == 'anyt800k':
                # This is a .comm file.
                in_paper = os.path.join(self.in_path, doi.strip())
            # Only yield an output path if out_path is not None.
            if self._return_outpath:
                if self.safe_dois:
                    out_paper = os.path.join(self.out_path, doi.strip())
                else:
                    out_paper = os.path.join(self.out_path, re.sub('/', '-', doi) + '.json')
                if self.optional_args:
                    yield (in_paper, out_paper) + self.optional_args
                else:
                    yield (in_paper, out_paper)
            else:
                if self.optional_args:
                    yield (in_paper,) + self.optional_args
                else:
                    yield in_paper
            count += 1


class ReusableFileStream:
    """
    Iterator over the lines of a file. Allowing for the file to be traversed in a loop
    multiple times.
    """
    def __init__(self, fname):
        """
        :param fname: string; file to iterate over.
        """
        self.fname = fname

    def __iter__(self):
        # "Rewind" the input file at the start of the loop
        self.in_file = codecs.open(self.fname, 'r', 'utf-8')
        return self.next()

    def next(self):
        # In each loop iteration return one example.
        for line in self.in_file:
            yield line.strip()


def write_json(data_dict, out_fname, save_space=True, sort_keys=True):
    """
    Write data to out_fname.
    :param data_dict: dict; any data to write out.
    :param out_fname: string; path to the file to write the data to.
    :param save_space: bool; if saving space dont pretty print json.
    :param sort_keys: bool; sort keys or not.
    :return: 1 if success else None.
    """
    try:
        out_file = codecs.open(out_fname, u'w', u'utf-8')
        try:
            # Write out unicode and pretty print it if you're not saving space.
            if save_space == True:
                json.dump(data_dict, out_file, ensure_ascii=False,
                          sort_keys=sort_keys)
            else:
                json.dump(data_dict, out_file, ensure_ascii=False, indent=2,
                          sort_keys=sort_keys)
            return 1
        except OverflowError or TypeError as e:
            sys.stderr.write('IO ERROR: Error writing json file: {}\n'.
                             format(e.args))
            return None
    except IOError as ioe:
        sys.stderr.write('IO ERROR: {:d}: {:s}: {:s}\n'.
                         format(ioe.errno, ioe.strerror, out_fname))
        return None


def read_perline_json(json_file):
    """
    Read per line JSON and yield.
    :param json_file: Just a open file. file-like with a next method.
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
            yield {}


def read_rawms_json(paper_path):
    """
    Read data needed from the raw material science JSON file specfied and
    return.
    :param paper_path: string; path to the paper to read from.
    :return:
        paper_doi: string
        segments: list(string); Whole sentence segments for readability.
        segments_toks: list(list(str)); Tokenized sentence segments.
        segments_labs: list(list(str)); Labels of tokenized sentence segments.
    """
    with codecs.open(paper_path, u'r', u'utf-8') as fp:
        paper_dict = json.load(fp, encoding=u'utf-8')

    segments = []
    segments_toks = []
    segments_labs = []
    paper_doi = paper_dict['doi']
    for action in paper_dict['actions']:
        segments.append(action['segment'])
        segments_toks.append(action['segment_toks'])
        segments_labs.append(action['segment_labs'])

    return paper_doi, segments, segments_toks, segments_labs


def get_rand_indices(maxnum, rand_seed=0.4186):
    """
    Return a permutation of the [0:maxnum-1] range.
    :return:
    """
    indices = range(maxnum)
    # Get random permutation of the indices but control for randomness.
    # https://stackoverflow.com/a/19307027/3262406
    random.shuffle(indices, lambda: rand_seed)
    return indices


def print_sorted_dict(d, out_file):
    for k in sorted(d, key=d.get, reverse=True):
        try:
            out_file.write("{}, {}\n".format(k, d[k]))
        except UnicodeError:
            pass


# Utilities for filtering data from the ANYT and conll data.
def check_verb_count(verb, op_counts):
    """
    Check if the op passed is valid.
    :param verb: string
    :param op_counts: giant dict with counts of all the operations. To filter with.
    :return: bool; True if all checks passed, else False.
    """
    count = op_counts[verb]
    if count >= pps.ANYT_MIN_OP_COUNT:
        return True
    else:
        return False


def string_is_number(arg):
    """
    Check if the argument is a number.
    :param arg: string
    :return:
        ret_str: is a designated placeholder if arg is a number, else its
            the passed arg.
    """
    # If its a number return without examining counts.
    try:
        float(arg)
        return pps.NUM_TAG
    except ValueError:
        return arg


def check_entity_count(arg, ent_counts, ignore_missing=False):
    """
    Check if the entity passed is valid.
    :param arg: string
    :param ent_counts: giant dict with counts of all the non-op entities.
    :param ignore_missing: bool; whether things not in the ent_counts dict should
        be ignored.
    :return:
        passed; bool; True if all checks passed, else False.
    """
    if arg == pps.NUM_TAG:
        return True
    # Added this when generating the emdsentvs data because of the fix there
    # to ignore non-contiguous nn spans.
    if ignore_missing:
        count = ent_counts.get(arg, 0)
    else:
        count = ent_counts[arg]
    # Throw if occurs few times.
    if count >= pps.MIN_ENT_COUNT:
        return True
    else:
        return False


def check_anyt_verb(verb):
    """
    Check if the op passed is valid.
    :param verb: string
    :return: bool; True if all checks passed, else False.
    """
    # Throw if it occurs very often, few times or is too short.
    if len(verb) >= pps.ANYT_MIN_OP_LEN:
        # Throw if any character in the verb matches a character in the
        # character set to be ignored.
        if pps.ANYT_RE_IC.search(verb) == None:
            return True
    return False


def check_anyt_arg(arg, exclude_args=False):
    """
    Check if the entity passed is valid.
    :param arg: string
    :param exclude_args: bool; says whether to exclude hand picked args which
        look noisey.
    :return:
        passed; bool; True if all checks passed, else False.
    """
    # If its a number then let it pass.
    if arg == pps.NUM_TAG:
        return True
    # Throw it away if its among a pre-specified set of arguments.
    if (arg.lower() in pps.EXCLUDE_ARGS) and exclude_args:
        return False
    # Throw if any character in the arg matches a character in the
    # character set to be ignored.
    if pps.ANYT_RE_IC.search(arg) == None:
        return True
    return False


def swap_content_in_context(verb_idxs, arg_idxs, tokens, window_size=pps.ANYT_SCON_WINDOW):
    """
    Given the verb, the argument indices and a list of the sentences tokens,
    return a string which is the verb-arg pairs sentence context.
    Just a util function in creating the rnentvs examples.
    :param verb_idxs: list(int); length one always. Says which token is the verb.
    :param arg_idxs: list(int); Says which tokens are the argument.
    :param tokens: list(string); tokens of the sentence.
    :param window_size: int; the number of tokens of context to consider on either
        side of the predicate-argument pair.
    :return: sentcon: string; "_" seperated tokens of the sentence with the
        verb and the argument swapped out with "<predicate_if>" "<argument_if>"
        also hoping that those markers wont be vocab elements.
            sentcon_w: string; "_" seperated tokens of the sentence with the
        verb and the argument swapped out with "<predicate_if>" "<argument_if>"
        and windowed around the pred-arg pair.
    """
    context_tokens = []
    for i in range(len(tokens)):
        # Make sure the tag is inserted only once.
        if i in verb_idxs:
            if not (pps.PRED_TAG in set(context_tokens)):
                context_tokens.append(pps.PRED_TAG)
            else:
                continue
        elif i in arg_idxs:
            if not (pps.ARG_TAG in set(context_tokens)):
                context_tokens.append(pps.ARG_TAG)
            else:
                continue
        else:
            # At this point, also filter out noisy looking context tokens.
            # If the filtered token is not None then use it.
            filtered_token = filter_anyt_scon_token(tokens[i])
            if filtered_token:
                context_tokens.append(filtered_token)
            else:
                continue
    # Get context tokens only from at most a scon window number of tokens on the
    # sides of the pred-arg pairs.
    span_idxs = [context_tokens.index(pps.PRED_TAG), context_tokens.index(pps.ARG_TAG)]
    span_idxs.sort()
    min_idx, max_idx = span_idxs[0], span_idxs[-1]
    context_tokens_window = context_tokens[:]
    # If the sentence context is less than the window then move on.
    if (min_idx - window_size < 0) and (max_idx + window_size + 1 > len(context_tokens)-1):
        pass
    # More tokens after the pred-arg pair than sentcon window.
    elif (min_idx - window_size < 0) and (max_idx + window_size + 1 <= len(context_tokens)-1):
        context_tokens_window = context_tokens[:max_idx + window_size + 1]
    # More tokens before the pred-arg pair than the sentcon window.
    elif (min_idx - window_size > 0) and (max_idx + window_size + 1 <= len(context_tokens)-1):
        context_tokens_window = context_tokens[min_idx-window_size:]
    else:
        context_tokens_window = context_tokens[min_idx - window_size:max_idx + window_size + 1]

    sentcon = '_'.join(context_tokens).lower()
    sentcon_w = '_'.join(context_tokens_window).lower()
    return sentcon, sentcon_w


def filter_anyt_scon_token(token):
    """
    Filter for the sentence context tokens.
    :param token: string.
    :return: string; if the token passed then return it else return None.
    """
    # If its a number return the number tag.
    try:
        float(token)
        return pps.NUM_TAG
    except ValueError:
        pass
    # If its a left or right bracket throw it away.
    if pps.SCON_LRB.search(token) or pps.SCON_RRB.search(token):
        return None
    # Throw if any character in the token matches a character in the
    # character set to be ignored.
    if pps.SCON_RE_IC.search(token):
        return None
    return token


def replace_in_string(original_str, replacement_str, start_idx, end_idx):
    """
    Place replacement_str in original_str starting at start_idx and ending at end_idx.
    :param original_str: string;
    :param replacement_str: string;
    :param start_idx: int;
    :param end_idx: int;
    :return: replaced str, new_start_idx, new_end_idx
    """
    ori_str_list = list(original_str)
    new_str_list = ori_str_list[:start_idx] + list(replacement_str) + ori_str_list[end_idx:]
    return ''.join(new_str_list), start_idx, start_idx+len(replacement_str)


def create_dir(dir_name):
    """
    Create the directory whose name is passed.
    :param dir_name: String saying the name of directory to create.
    :return: None.
    """
    # Create output directory if it doesnt exist.
    try:
        os.makedirs(dir_name)
        print(u'Created {} for paper json files'.format(dir_name))
    except OSError as ose:
        # For the case of *file* by name of out_dir existing
        if (not os.path.isdir(dir_name)) and (ose.errno == errno.EEXIST):
            sys.stderr.write('IO ERROR: Could not create output directory\n')
            sys.exit(1)
        # If its something else you don't know; report it and exit.
        if ose.errno != errno.EEXIST:
            sys.stderr.write('OS ERROR: {:d}: {:s}: {:s}\n'.format(ose.errno,
                                                                   ose.strerror,
                                                                   dir_name))
            sys.exit(1)


# Utilities for the MSALL data.
def fix_bio_labels(sent_labs):
    """
    Fix stray I-labels in the predicted entities.
    :param sent_labs: list(string)
    :return: sent_labs_fixed: list(string); with stray I-label tags fixed.
    """
    # Dont change the sent_labs in-place. Make a copy.
    sent_labs_fixed = copy.deepcopy(sent_labs)
    inlab = False
    for i, lab in enumerate(sent_labs_fixed):
        if lab[0] == 'O':
            inlab = False
        if lab[0] == 'B':
            inlab = True
        if lab[0] == 'I' and inlab == True:
            continue
        elif lab[0] == 'I' and inlab == False:
            sent_labs_fixed[i] = 'B' + lab[1:]
            inlab = True
    return sent_labs_fixed


def parse_bio_labels(sent_toks, sent_labs):
    """
    Parse the bio entity tagged sentence and return spans of entities in the
    sentence.
    :param sent_toks: list(string); the tokenized sentence.
    :param sent_labs: list(string); BIO labels for every token in the sentence.
    :return:
        sent_spans: list(string); the parsed sentence with entity spans.
        sent_span_labs: list(string); the labels for every sentence span.
    """
    sent_spans = []
    sent_span_labs = []
    for tok, lab in zip(sent_toks, sent_labs):
        if lab == 'O':
            sent_spans.append(tok)
            sent_span_labs.append(lab)
        # You saw a beginning, so add ent to the sent and place label into
        # labels.
        elif lab[0] == 'B':
            cur_lab = lab[2:]
            sent_spans.append(tok)
            sent_span_labs.append(cur_lab)
        # You see a "in", if its the same current label concat it with the B ent
        # token in the sentence and skip any addition to labs.
        elif lab[0] == 'I' and lab[2:] == cur_lab:
            sent_spans[-1] = (sent_spans[-1] + ' ' + tok).strip()
    return sent_spans, sent_span_labs


def check_operation(op):
    """
    Check if the op passed is valid.
    :param op: string
    :return: bool; True if all checks passed, else False.
    """
    # Throw if its too short. # Throw if its all-caps. # Throw if it matches a defined pattern
    if len(op) >= pps.MIN_OP_LEN and (not op.isupper()) and (pps.RE_D.search(op) == None):
        return True
    else:
        return False


def check_argument(ent):
    """
    Check if the entity passed is valid.
    :param ent: string
    :return: bool; True if all checks passed, else False.
    """
    # If its a single character that occurs in the set of invalid characters.
    if ent in pps.IC:
        return False
    else:
        return True
