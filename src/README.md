# Baseline Model
## Prerequisites
* Pytorch
* h5py (Python's hdf5 Library)

## Running the Model
Unizip the files present in the data directory. In the example below, the model is run on the 106th Congress.

```
python baseline_model.py --datafile ../data/106.hdf5 --classifier nn_embed_m_nocv --eta 0.1 --nepochs 10 --dp 10
```
