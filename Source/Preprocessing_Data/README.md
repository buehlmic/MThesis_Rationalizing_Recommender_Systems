# Preprocessing Data

This folder contains all necessary scripts to prepare the data sets. 

Because the necessary data relies on the [Amazon product data set](http://jmcauley.ucsd.edu/data/amazon/) from Julian McAuley, we will not make the data sets public. Please contact us if you need access to the data. 

## Dependencies

1. NLTK + the additional 'punkt' package from the [NLTK Downloader](http://www.nltk.org/data.html)

## How to Run the Code

The python scripts should be run in the following order:

1. *splitting_data_set.py*: This script splits the data sets into categories.
2. *prepare_FLIC_dataset.py*: This script constructs the data set for training the FLIC model.
3. *prepare_model_dataset.py*: This script constructs the data sets for the RRS and the sentence embedding models.

For all scripts, a help string is available (e.g. run `python prepare_FLIC_dataset.py -h`)
