# Training the FLIC model

## Contributions

The following files were written by Sebastian Tschiatschek (Sebastian.Tschiatschek@microsoft.com):
  1. fast_train.py
  2. _flm.cpp
  3. Makefile
 
 
The following file was written by Michael Bühler (buehler_michael@bluewin.ch):
  1. flic.py
  
  
## Training the FLIC model


To train the FLIC model, type the following in your terminal:

1) `make` to compile the file _flm.cpp.

2) `python flic.py -d 100 -n 100` to train the FLIC model on the data of the category **Wine** with d=100 hidden (attraction) dimensions and n=100 steps.

3) `python flic.py -h` to see the available options for training the model.

For a qualitatively good model, we recommend having the number of steps at least 100 (-n 100).
