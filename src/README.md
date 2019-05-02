# Prerequisites
* Pytorch
* h5py (Python's hdf5 Library)

# Baseline Model
Unizip the files present in the data directory (The --datafile argument should point to a specific file in the data directory). In the example below, the model is run on the 106th Congress.
## Running the Model
To create a new model, run the following command:
```
python baseline_model.py --datafile ../data/106.hdf5 --classifier nn_embed_m_nocv --eta 0.1 --nepochs 10 --dp 10 --congress 106 --runeval True
```
To test an existing model, run the above command but add the ``` --modelpath <path to saved model> ``` parameter.
# Text Model
Unzip the glove_text_files.zip or text_feature_files.zip in the data directory, depending on whether you want to use the glove embedding model or the bag-of-words model respectively. The ``` --modeltype glove ``` or ``` --modeltype bag ``` parameter should be set correctly, as in the below command:
## Running the model
```
python text_model.py --datafile ../data/106.hdf5 --textfile ../data/106_text_bag.hdf5 --classifier nn_embed_m_nocv --eta 0.1 --nepochs 10 --dp 10 --modeltype bag --congress 106 --runeval True
```
To test an existing model, run the above command but add the ``` --modelpath <path to saved model> ``` parameter.
# Universal Schema Model
This model uses the same zip files as the baseline model, but additionally needs the pol_to_pairs.json file and learned embedding matrices from the universal schema model.
## Running the model
```
python us_model.py --datafile ../data/106_no_eval.hdf5 --classifier nn_embed_m_nocv --eta 0.1 --nepochs 10 --dp 10 --congress 106 --lognum 1
```
