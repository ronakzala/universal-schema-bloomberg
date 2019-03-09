### Running this code:

TODO. In general the workflow and the comments in bin and src files should inform you of this. If not, get in touch with me.

### Working directories:

This project lives on blake: `/iesl/canvas/smysore/material_science_framex`.

`datasets_raw`: This contains unprocessed data in its original form (eg. conll tsv files) with the directories being symbolic links to the original raw data elsewhere else on blake. Many of the pre-processing functions read these raw unprocessed files.

`datasets_proc`: This contains data splits for every dataset I'm working with for every model variant I am experimenting with. Pre-processing mostly writes things out to some sub-directory in this directory. All processed data files are text files are are one json string per line. This data has also undergone filtering steps to discard infrequent tokens and such.

`experiments`: Is the main code directory corresponding to this repository.

`logs`: All the code except the training and parts of the eval code write their logs here. Often log files are names after the source file and function generating them.

`model_runs`: This contains sub-directories for every dataset and every model variant that I train. The trained models, their outputs and all sorts of evaluations of the models live here.

`results`: This contains some results. But this is mostly not in use now. Most results are in the model_runs and data stats are in the logs.

### General workflow:
(This is just an example workflow, based on what you want to do with which dataset this can vary considerably. Get in touch if you need to.)

1. `cd /iesl/canvas/smysore/material_science_framex/`
1. `export CUR_PROJ_DIR="/iesl/canvas/smysore/material_science_framex"`
1. `cd ./experiments/bin`
1. Create json-per-line files from the raw dataset with an appropriate pre-proc function.
1. Create the train, test, eval splits for the dataset.
1. Create a further processed dataset which does some count based filtering and such. 

        ./pre_process/run_pre_proc_msall.sh make_rowcol <EXPERIMENT>

1. Create readable negative examples to help with evaluation:

        ./models/run_nn_preproc.sh readable_neg <EXPERIMENT>

1. Map all strings to integers:

        ./models/run_nn_preproc.sh int_map <EXPERIMENT>

1. Train the model:

        ./models/run_main.sh train_model <EXPERIMENT>

1. Make predictions:

        ./models/run_main.sh run_saved <EXPERIMENT> <RUN_DIR>

1. Print nearest neighbours and high scoring row-col pairs for manual evaluation:

        ./evaluation/run_man_eval.sh print_scores <EXPERIMENT> <RUN_DIR>
        ./evaluation/run_man_eval.sh nearest_ents <EXPERIMENT> <RUN_DIR>
        ./evaluation/run_man_eval.sh nearest_rows <EXPERIMENT> <RUN_DIR>
        ./evaluation/run_man_eval.sh nearest_cols <EXPERIMENT> <RUN_DIR>
        ./evaluation/run_man_eval.sh nearest_ops <EXPERIMENT> <RUN_DIR>

1. Run automatic evaluations:

        ./run_auto_eval.sh pseudo_eval <RUN_DIR>
        

