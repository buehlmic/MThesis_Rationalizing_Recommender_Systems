# Rationalizing Recommender Systems

This is the repository for the code written for my Master's Thesis. It trains and evaluates an interpretable Recommender System model as described in *MThesis_Michael_Buehler.pdf*. 

All code is stored in the folder *Source/*. The folder *Data/Preprocessed_Data/* should store the data sets (see below). The data from the experiments described in the written part can be found in the folder *Results*.


## Installation instructions

### Getting the Data

The code comes together with all necessary data sets to train the model on a small data set called **Wine**.  The data sets are stored in the folder *Data/Preprocessed_Data/Wine/*.

The data sets for training the categories **Kindle** and **Magazine Subscriptions** can be downloaded [here](https://polybox.ethz.ch/index.php/s/w2qGeO8Eh9KLdps). Store the downloaded folders in *Data/Preprocessed_Data/*.

### Dependencies

Please make sure that the following libraries are successfully installed on your system for running the code in 'Source/Model/':

1. Theano >= 0.9
2. Python 2.7
3. Numpy
4. Pytorch >= 0.1.12 (to run the sentence embedding model)


To run the code to prepare the data sets (optional) you additionally need:

5. NLTK + the 'punkt' package from the [NLTK Downloader](http://www.nltk.org/data.html)

### Running the Code

1) Add the folder *Source/* to your python path. (E.g. run `export PYTHONPATH=$PYTHONPATH:/path/to/RRS/Source` in your terminal.) 

2) Go to the folder *Source/Model/FLIC/* and then run the command `make`. This will construct the file *_flm.so* which is needed for the FLIC model.

3) Go to the folder *Source/Model/FLIC/*. Then you can train the FLIC model by running `python flic.py [OPTIONS]`. We recommend that you have a look at the available options  by running `python flic.py -h`.

4) If you want to train the Sentence Embeddings (optional), go to the folder *Source/Model/Sentence_Embedding/*. Then run `python sentence_embeddings.py [OPTIONS]`. We recommend that you have a look at the avaliable options by running `python sentence_embeddings.py -h`.

5) If you want to train the Recommender System, go to the folder *Source/Model/RRS/*. There you can train and evaluate the data set **Wine** by running `sh run_Wine.sh`. Please have a look at the available options by running `python rrs.py -h`.  
After you have downloaded the data sets **Kindle** and **Magazine Subscriptions** (see above), you can train and evaluate the models by running `sh run_Kindle.sh` or `sh run_Magazine.sh`.
