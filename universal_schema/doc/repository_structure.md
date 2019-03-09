This describes how the repository is structured and what specific files are for. While I try to be consistent to an overall design here, there are times where I include hacks to get quick results. Some of these hacks might always be floating around although they're often highlighted in the comments.

        experiments
        ├── bin
        │   ├── evaluation
        │   │   ├── run_auto_eval.sh
        │   │   ├── run_cluster_eval.sh
        │   │   └── run_man_eval.sh
        │   ├── explore_data
        │   │   ├── run_event_stats.sh
        │   │   └── run_plot_op_pos.sh
        │   ├── learning
        │   │   ├── run_main_depembs.sh
        │   │   ├── run_main_frame.sh
        │   │   ├── run_main_mt.sh
        │   │   ├── run_main_script.sh
        │   │   └── run_nn_preproc.sh
        │   └── pre_process
        │       ├── run_pre_proc_anyt.sh
        │       ├── run_pre_proc_conll.sh
        │       ├── run_pre_proc_msall.sh
        │       └── run_pre_proc_sempl.sh
        ├── config
        │   └── models_config
        │       ├── anyt
        │       ├── conll2009en
        │       └── ms500k
        ├── doc
        │   └── repository_structure.md
        ├── ipynb
        ├── README.md
        ├── scripts
        └── src
            ├── evaluation
            │   ├── auto_eval.py
            │   ├── cluster_assign.py
            │   ├── cluster_eval.py
            │   ├── edge_eval.py
            │   ├── man_eval.py
            │   └── script_eval.py
            ├── explore_data
            │   ├── data_utils.py
            │   ├── event_stats.py
            │   ├── explore_anyt.py
            │   ├── explore_parsedwiki.py
            │   └── plot_op_pos.py
            ├── __init__.py
            ├── learning
            │   ├── batchers.py
            │   ├── data_utils.py
            │   ├── dep_embeddings.py
            │   ├── frame_models
            │   │   ├── compus.py
            │   │   ├── compvschema.py
            │   │   ├── ltcompus.py
            │   │   ├── narycompus.py
            │   │   ├── relnet.py
            │   │   └── rncompus.py
            │   ├── le_settings.py
            │   ├── main_frame.py
            │   ├── main_multitask.py
            │   ├── main_script.py
            │   ├── models_common
            │   │   ├── loss.py
            │   │   ├── model_utils.py
            │   ├── multitask_models
            │   │   └── vs_rnnlm_mt.py
            │   ├── nn_preproc.py
            │   ├── predict_utils.py
            │   ├── script_models
            │   │   ├── ngram_lm.py
            │   │   ├── rnn_lm.py
            │   └── trainer.py
            ├── pre_process
            │   ├── count_ents.py
            │   ├── data_utils.py
            │   ├── finereval_neg.py
            │   ├── pp_settings.py
            │   ├── pre_proc_anyt.py
            │   ├── pre_proc_conll.py
            │   ├── pre_proc_msall.py
            │   ├── pre_proc_msann.py
            │   └── pre_proc_sempl.py
            └── tests
                ├── mu_batcher_tests.py
                └── pp_msall_tests.py

The following tries to describe what the different parts of the repo are for with more detailed descriptions for some source files.

`ipynb/`: Things I want to visualize and interactively develop.

`scripts/`: Miscellaneous sets of scripts that fit nowhere else. Most of these were mainly written up at different times to glue things or get quick results. `scripts/split_merge_nw.sh` is critically important though.

`scripts/split_merge_nw.sh`: This contains well documented bash functions which merge the train/dev/test splits created in conll2009en, conll2012wsj and semantic\_plausibility\_naacl2018 dataset into the larger Annotated NYT corpus for training. It also makes merges between splits of these different datasets for development and evaluation of the clustering post the learning stage. Look at the function documentation.

`src/explore_data`: This contains a miscellaneous set of source files which collect some summary statistics or makes exploratory plots for different datasets. Summary statistics are generally collected and printed by the pre-processing functions but this is the set of functions which did not fit in there and were better done as stand alones functions.

`src/tests`: These mainly test the buggiest parts of the code. This is mainly creating some test cases for the batching code (`src/learning/batchers.py`) and for the model classes (`src/learning/*_models`) and printing results out to stdout. I most often just look at these and make sure things look fine. No really fancy testing here.

`src/pre_process`: This contains source files to pre-process each of the different datasets I use: the large annotated nyt corpus, the large materials science corpus and each of the target task datasets, materials science annotations, the semantic plausibility dataset and the conll2009en and conll2012wsj data. Aside from pre-processing this will also create files which a trained model is asked to make predictions on for evaluation. Some of the heuristic baselines are also implemented here.

`src/learning`: This contains all the models I currently have as pytorch model classes. Source files contain functions and classes for: mapping text data to integers, model classes, batcher classes, a generic trainer class to train any model and scripts to instantiate models and make predictions with the trained model.

`src/evaluation`: Source files for: perform an automatic evaluation of a trigger-argument_set affinity model with a disambiguation task, make cluster assignments for learned representations, evaluate the generated clusters with cluster evaluation measures, evaluate edge assignments post learning and inference, generating output files for manual evaluation of various learned representations, evaluate script models under language modeling style evaluation.

`config/`: Training hyper-parameters and model hyper-parameters for every dataset for every model variant are fixed here. The config files are essentially bash variables which get sourced into the bin file which launches a model (`bin/learning/run_main_*.sh`) with these parameters as command line arguments to a python source file.

`bin/`: This is a set of bash scripts which execute the python code with the right set of command line arguments, redirect the outputs to log files and do any necessary setting up and cleaning up of files that the python code expects. All the bash scripts take a smaller set of command line arguments themselves, hopefully reading the source of the scripts will tell you what they are.
