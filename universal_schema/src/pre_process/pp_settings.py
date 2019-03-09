# -*- coding: utf-8 -*-
"""
Constants used throughout.
"""
from __future__ import unicode_literals
import re


## Stuff for the old materials science dataset. At the end of May 2018.
# Labels you care about.
label_set = {'amt_unit', 'amt_misc', 'cnd_unit', 'cnd_misc', 'material',
             'target', 'descriptor', 'prop_unit', 'prop_type',
             'synth_aprt', 'brand', 'intrmed', 'prop_misc',
             'cnd_type', 'aprt_unit', 'aprt_des'}
# Minimum count of operation for it to be considered valid.
MIN_OP_COUNT = 100
# Minimum length of operation string for it to be considered.
MIN_OP_LEN = 3
# Match patterns in side op. For now its just numbers
RE_D = re.compile('\d')
# Minimum count of non-operation entities for it to be considered valid.
MIN_ENT_COUNT = 10
# Check if there are single char entities.
IC = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
IC.update(['"', '!', '#', '$', '%', '&', '*', '+', '-', '.', '^', '_', '`', '\'',
           '|', '~', ':', ';', '(', ')', '[', ']', '\\', '/'])

## Stuff for the 235 materials science anns. Obtained at the end of July 2018.
# Labels for entity types which are direct one hop relations to the operation.
# Defined in: https://github.com/olivettigroup/paper-annotation/blob/master/brat/annotation.conf
msanna_oplabel = 'Operation'
maanns_arg_label_set = {# Types admitted by material relations to operations.
                        'Material', 'Nonrecipe-Material', 'Unspecified-Material',
                        # Types admitted by Condition_of.
                        'Condition-Unit', 'Condition-Misc',
                        # Types admitted by Apparatus_of.
                        'Synthesis-Apparatus', 'Characterization-Apparatus'}
msanns_null = 'null'
MSANN_SCON_WINDOW = 5


## Stuff for the annotated nyt dataset.
# Dependency edge labels to ignore.
SEC_TYPE = 'Passage'
VB_POS = 'VB'
DEP_TOOL = 'Stanford CoreNLP col-CC'
EXCLUDE_SDEPS = {'abbrev', 'advcl', 'advmod', 'amod', 'appos', 'attr', 'aux', 'auxpass',
                 'cc', 'complm', 'conj', 'cop', 'det', 'dep', 'discourse', 'expl', 'infmod',
                 'goeswith', 'mark', 'measure', 'mwe', 'nn', 'num', 'number', 'npadvmod',
                 'number', 'parataxis', 'pcomp', 'poss', 'possessive', 'preconj',
                 'predet', 'prt', 'punct', 'quantmod', 'rel', 'ref', 'vmod'}
INCLUDE_SDEPS = {'acomp', 'agent', 'ccomp: obj', 'csubj', 'csubjpass',
                 'dobj', 'iobj', 'neg', 'nsubj', 'nsubjpass', 'partmod', 'pobj',
                 'prep', 'prepc', 'purpcl', 'rcmod', 'tmod', 'xcomp',
                 'xsubj'}
# The dependency relations from the verb for saying if a token is an argument.
INCLUDE_DEPS09 = {'SBJ', 'OBJ', 'ADV', 'TMP', 'OPRD', 'LOC', 'DIR', 'MNR', 'PRP',
                  'LGS', 'EXT', 'DTV'}
# Constants for the Lang and Lapata set of rules.
LLR2_EXCLUDE_DEPS09 = {'IM', 'PRT', 'COORD', 'P', 'SUB'}
LLR4_EXCLUDE_DEPS09 = {'ADV', 'AMOD', 'APPO', 'BNF', 'CONJ', 'COORD', 'DIR', 'DTV', 'EXT',
                       'EXTR', 'HMOD', 'IOBJ', 'LGS', 'LOC', 'MNR', 'NMOD', 'OBJ', 'OPRD', 'POSTHON',
                       'PRD', 'PRN', 'PRP', 'PRT', 'PUT', 'SBJ', 'SUB', 'SUFFIX', 'TMP', 'VOC'}

SDEPS_MAP = {'csubj': 'subj',
             'csubjpass': 'subj',
             'nsubj': 'subj',
             'nsubjpass': 'subj',
             'xsubj': 'subj',
             'dobj': 'obj',
             'iobj': 'iobj',
             'rcmod': 'mod',
             'acomp': 'comp',
             'ccomp': 'comp',
             'xcomp': 'comp',
             'neg': 'otherdep',  # Including because the prior work evaluated this role.
             'prep': 'prep',
             'prepc': 'prep',
             'root': 'root',
             # Pretty much semantic arguments.
             'purpcl': 'purpcl',
             'agent': 'agent',
             'tmod': 'tmod'}
DEPS09_MAP = {'SBJ': 'subj',
              'OBJ': 'obj',
              'DTV': 'iobj',
              'ADV': 'mod',
              'OPRD': 'comp',
              'ROOT': 'root',
              # Pretty much semantic arguments. But mapping some to deps to allow dep based
              # model to be trained.
              'LOC': 'prep',
              'DIR': 'prep',
              'MNR': 'prep',
              'EXT': 'prep',
              'PRP': 'purpcl',
              'LGS': 'agent',
              'TMP': 'tmod'}
# Everything not getting mapped to one of the above deps is mapped to 'other'.
OTHER_DEP = 'otherdep'

# Arguments to exclude. These were things which occurred numerous times in the
# anyt data. Don't look like content words but keeping them now because the
# conll2009en data contains them.
EXCLUDE_ARGS = {'is', 'do', 'go', 'be', 'much', 'a', 'want', 'has', 'give', 'with',
                'become', 'other', 'as', 'by', 'given', 'each', 'got', 'able', 'getting',
                'get', 'did', 'none', 'at', 'not', 'coming', 'less', 'took', 'such',
                'than', 'no', 'no.', 'an', 'q.', 'p.', 't.', 'c', 'r.', ''}
# Sentence lengths to be considered.
ANYT_MAX_SENT_LEN = 90
ANYT_MIN_SENT_LEN = 3
# Minimum count of operation for it to be considered valid.
ANYT_MIN_OP_COUNT = 100
# Minimum length of operation string for it to be considered.
ANYT_MIN_OP_LEN = 2
# Match patterns in side op. For now its just numbers
ANYTRE_D = re.compile('\d')
# Minimum count of non-operation entities for it to be considered valid.
ANYT_MIN_ENT_COUNT = 10
# Check if there are single char entities.
ANYT_RE_IC = re.compile('[\@\!\#\*\+\^\_\`\|\~\;\(\)\[\]\\\/]')
# For getting the sentence context dont go beyond the window length number of tokens
# centered on the predicate-argument pair.
ANYT_SCON_WINDOW = 5
# Constants for the sentence context filtering.
NUM_TAG = '<numberptag>'
# Ignore sentence context tokens if they have the following characters in them
SCON_RE_IC = re.compile('[0-9\%\&\@\!\#\*\+\^\_\`\|\~\:\;\(\)\[\]\{\}\\\/\.]')
SCON_LRB = re.compile('\-lrb\-')
SCON_RRB = re.compile('\-rrb\-')

PRED_TAG = '<predicateptag>'
ARG_TAG = '<argumentptag>'

# Constants to parse the conll 2009 file (same as 2008 content).
# Column indices for specific items from the dataset paper.
# https://www.aclweb.org/anthology/W/W09/W09-1201.pdf
ID_08 = 1-1
FORM_08 = 2-1
FILLPRED_08 = 13-1
PRED_08 = 14-1
LEMMA_08 = 3-1
DEPGOV_08 = 9-1
DEPREL_08 = 11-1
PDEPGOV_08 = 10-1
PDEPREL_08 = 12-1
APRED_08 = 15-1
POS_08 = 5-1
# Constants for the MALT Parser output that Jeff handed me.
ID_MP = 1-1
DEPGOV_MP = 9-1
DEPREL_MP = 10-1
POS_MP = 4-1
# Ignore any roles starting with R or C. These are the discontinuous or referent
# arguments: (Slide 25 below)
# http://courses.washington.edu/ling571/ling571_WIN2016/slides/ling571_class12_sem_srl_flat.pdf
EXCLUDE_ROLES_RE = re.compile('^[R|C]\-')
EXCLUDE_ROLES_SET = {'ARGM-MOD', 'ARGM-ADV', 'ARGM-DIS', 'ARGM-NEG', 'ARGM-PRD',
                     'ARGM-ADJ', 'ARGM-LVB', 'ARGM-REC'}
# Constants for the conll formatted ontonotes file from 2012.
TOK_ID_12 = 3-1
FORM_12 = 4-1
LEMMA_12 = 13-1
PROPB_FS_ID_12 = 14-1
APRED_12 = 18-1
PDEPHEAD_12 = 7-1
PDEPREL_12 = 8-1
DEPHEAD_12 = 10-1
DEPREL_12 = 11-1

# Constants for sampling the ANYT dataset for training examples.
MAX_TO_EXAMINE = 200000
