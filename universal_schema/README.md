### Running this code:

TODO. In general the workflow and the comments in bin and src files should inform you of this. If not, get in touch with me.

### Working directories:

The code expects a directory structure described below.

`datasets_raw`: This contains unprocessed data in its original form. Any pre-processing code will work with this in general and write the results to the `datasets_proc` directory for being used by the models.

`datasets_proc`: This contains data splits for every dataset and every model variant. Pre-processing mostly writes things out to some sub-directory in this directory. All processed data files are text files are are one json string per line (This can be different for you and you will appropriately need to change the code in `src/learning/nn_preproc.py` and `src/learning/batchers.py`). This data has also undergone filtering steps to discard infrequent tokens/entities/relations and such. For example for the `latfeatus` model trained on `freebase` triples there should be directory `datasets_proc/freebase/latfeatus/` with the appriproate data for training the model.

`experiments`: The main code directory corresponding to this repository.

`logs`: All the code except the training and parts of the eval code write their logs here. Often log files are names after the source file and function generating them.

`model_runs`: This contains sub-directories for every dataset and every model variant that you train. The trained models, their outputs and all sorts of evaluations of the models live here.

`results`: This contains some results. But this is mostly not in use now. Most results are in the model_runs and data stats are in the logs.

### Contents of the repository:

            ├── bin
            │   ├── evaluation
            │   │   └── run_man_eval.sh
            │   ├── learning
            │   │   ├── run_main_frame.sh
            │   │   └── run_nn_preproc.sh
            │   └── pre_process
            ├── config
            │   └── models_config
            │       ├── anyt
            │       ├── fbanyt
            │       └── freebase
            │           └── latfeatus-fixed.conf
            ├── ipynb
            ├── README.md
            ├── scripts
            └── src
                ├── evaluation
                │   ├── auto_eval.py
                │   └── man_eval.py
                ├── __init__.py
                ├── learning
                │   ├── batchers.py
                │   ├── data_utils.py
                │   ├── frame_models
                │   │   ├── compus.py
                │   │   ├── compvschema.py
                │   │   ├── __init__.py
                │   ├── __init__.py
                │   ├── le_settings.py
                │   ├── main_frame.py
                │   ├── models_common
                │   │   ├── __init__.py
                │   │   ├── loss.py
                │   │   ├── model_utils.py
                │   ├── nn_preproc.py
                │   ├── predict_utils.py
                │   └── trainer.py
                ├── pre_process
                │   ├── data_utils.py
                │   ├── finereval_neg.py
                │   ├── __init__.py
                │   ├── pp_settings.py
                └── tests
                    ├── __init__.py
                    └── mu_batcher_tests.py
                    
The following tries to describe what the different parts of the repo are for with more detailed descriptions for some source files.

`ipynb/`: Things you may want to visualize and interactively develop.

`scripts/`: Miscellaneous sets of scripts that fit nowhere else. Most of these were mainly written up at different times to glue things or get quick results.

`src/tests`: These mainly test the buggiest parts of the code. This is mainly creating some test cases for the batching code (`src/learning/batchers.py`) and for the model classes (`src/learning/*_models`) and printing results out to stdout. I most often just look at these and make sure things look fine. No really fancy testing here.

`src/pre_process`: This contains source files to pre-process each of the different datasets you may use. For now this contains some code which may be useful in future but nothing important here for now.

`src/learning`: This contains all the models as pytorch model classes. Source files contain functions and classes for: mapping text data to integers (`nn_preproc.py`), model classes (mostly everything), batcher classes (`batchers.py`), a generic trainer class to train any model (`trainer.py`) and scripts to instantiate models and make predictions with the trained model (`main_frame.py`).

`src/evaluation`: Source files for: perform an automatic evaluation of a trigger-argument_set affinity model with a disambiguation task (`auto_eval.py`) if it is to be used in the future and generating output files for manual evaluation of various learned representations (`man_eval.py`).

`config/`: Training hyper-parameters and model hyper-parameters for every dataset for every model variant are specified here. The config files are essentially bash variables which get sourced into the bin file which launches a model (`bin/learning/run_main_*.sh`) with these parameters as command line arguments to a python source file.

`bin/`: This is a set of bash scripts which execute the python code with the right set of command line arguments, redirect the outputs to log files and do any necessary setting up and cleaning up of files that the python code expects. All the bash scripts take a smaller set of command line arguments themselves, reading the source of the scripts should tell you what they are.


### General workflow:
This expects a directory structure as defined above. This top level directory is your working directory and you should set `CUR_PROJ_DIR` to the top level directory. This should be pointing to universal_schema.

1. `export CUR_PROJ_DIR="/iesl/canvas/smysore/material_science_framex"` # This is just an example
1. Create a set of relation and entity pair triples in a suitable format(dev.txt, train.txt, test.txt).
1. Conver the creates set of relation and entity pair to json compatible format(dev.json, train.json, test.json) 

        ./bin/learning/run_convert2json.sh 
1. Create readable negative examples that you need for training the US model (here, `latfeatus` on the `freebase` data):

        ./bin/learning/run_nn_preproc.sh -a readable_neg -e latfeatus -d freebase

1. Map all the data to integers:

        ./bin/learning/run_nn_preproc.sh -a int_map -e latfeatus -d freebase

1. Train the model:

        ./bin/learning/run_main_frame.sh -a train_model -d freebase -e latfeatus -s <meaningful_suffix>

1. Make predictions on test data if you have that set up, save learned embeddings etc:

        ./bin/learning/run_main_frame.sh -a run_saved -d freebase -e latfeatus -s <same meaningful_suffix as above> -r <model_run directory name>


