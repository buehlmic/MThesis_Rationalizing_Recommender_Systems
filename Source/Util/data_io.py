# coding: utf-8
'''
    A collection of classes and functions to load and store data.
    
    @author: Michael Bühler (buehler_michael@bluewin.ch)
'''

import numpy as np
from itertools import repeat
from Util.data_helper import load, store, data_subset, ProcessText
from Util.nn.basic import EmbeddingLayer
from random import shuffle
from math import ceil
import pandas as pd

class CrossValidationLinks:
    """ Class for loading the link_data in k subsamples (k-fold cross-validation)."""

    def __init__(self, args, k = 4, perc_validation_data = 0.5):
        """
        :type args: Argument Parser
        :param args: A class containing the parameters which were given to the program as command line arguments.

        :type k: Int
        :param k: The number of disjoint data link samples.
        :type perc_validation_data: float
        :param perc_validation_data: The percentage of the links in the smaller subsample, which is used as the
        validation data. The remaining '1-perc_validation_data' percent of the link data will be returned as the test data.

        :return type: A tuple of pairs
        :return return: D_train, D_C_train, D_valid, D_C_valid, D_test, D_C_test
        """

        self.k = k

        # Load datasets from memory
        D = list(load(args.linkdata_pos))
        D_C = list(load(args.linkdata_neg))
        self.len_D = len(D)
        self.len_D_C = len(D_C)

        # Permute datasets
        np.random.seed(1234)
        self.D = np.random.permutation(D)
        self.D_C = np.random.permutation(D_C)

        # Indices for datasets D and D_C
        self.len_D_valid = int((self.len_D/float(k))*perc_validation_data)
        self.len_D_test = int(self.len_D/float(k) - self.len_D_valid)
        self.len_D_C_valid = int((self.len_D_C/float(k))*perc_validation_data)
        self.len_D_C_test = int(self.len_D_C/float(k) - self.len_D_C_valid)


    def next_sample(self):
        start_D_valid =  0
        start_D_C_valid = 0

        for i in range(self.k):
            end_D_valid = start_D_valid + self.len_D_valid
            end_D_test = end_D_valid + self.len_D_test
            end_D_C_valid = start_D_C_valid + self.len_D_C_valid
            end_D_C_test = end_D_C_valid + self.len_D_C_test

            yield np.concatenate((self.D[:start_D_valid], self.D[end_D_test:])), \
                  self.D[start_D_valid:end_D_valid], \
                  self.D[end_D_valid:end_D_test], \
                  np.concatenate((self.D_C[:start_D_C_valid], self.D_C[end_D_C_test:])), \
                  self.D_C[start_D_C_valid:end_D_C_valid], \
                  self.D_C[end_D_C_valid:end_D_C_test]

            start_D_valid = end_D_test
            start_D_C_valid = end_D_C_test


class CrossValidationProducts:

    def __init__(self, args, k = 4, perc_validation_data = 0.5):
        """
        :type args: Argument Parser
        :param args: A class containing the parameters which were given to the program as command line arguments.

        :type k: Int
        :param k: The number of disjoint data link samples.
        :type perc_validation_data: float
        :param perc_validation_data: The percentage of the links from the training part of the subsample, which is used as the
        validation data.

        :return type: A tuple of pairs
        :return return: D_train, D_C_train, D_valid, D_C_valid, D_test, D_C_test
        """

        self.k = k
        self.perc_validation_data = perc_validation_data

        # Load datasets from memory
        self.D = list(load(args.linkdata_pos))
        self.D_C = list(load(args.linkdata_neg))

        # Get list of products
        prods = {prod for prod,_ in self.D}
        prods = list(prods)

        # Permute product
        np.random.seed(1234)
        prods = np.random.permutation(prods)

        # Split products for the cross-validation runs
        num, rem = divmod(len(prods), k)
        self.prods = [set(prods[i * num + min(i, rem):(i + 1) * num + min(i + 1, rem)]) for i in range(k)]

    def next_sample(self):
        for i in range(self.k):

            # Split off test data
            D_test = filter(lambda x: x[1] in self.prods[i], self.D)
            D_C_test = filter(lambda x: x[1] in self.prods[i], self.D_C)
            D_data = filter(lambda x: x[1] not in self.prods[i], self.D)
            D_C_data = filter(lambda x: x[1] not in self.prods[i], self.D_C)

            # Split validation from training data
            len_D_val = int(ceil(self.perc_validation_data*len(D_data)))
            len_D_C_val = int(ceil(self.perc_validation_data*len(D_C_data)))
            shuffle(D_data)
            shuffle(D_C_data)

            D_train = D_data[len_D_val:]
            D_val = D_data[:len_D_val]
            D_C_train = D_C_data[len_D_C_val:]
            D_C_val = D_C_data[:len_D_C_val]
            yield D_train, D_val, D_test, D_C_train, D_C_val, D_C_test


def load_reviews(args):
    """Loads the reviews data set.

    :type args: Argument Parser
    :param args: A class containing the parameters which were given to the program as command line arguments.

    """

    reviews = load(args.reviews)
    return reviews


def load_flic_data(args, flic):
    flic.load_model(args.flic)
    flic._w_att = flic._w_att / (np.linalg.norm(flic._w_att, axis=1, keepdims=True))

def load_link_data(args, perc_validation_data = 0.2, perc_test_data = None):
    """Loads the link data from the paths given by the command line parameters of the program. The training data
    is split into training, validation and test data.

    The following Data gets loaded: D, D_C, prod_to_sent.

    :type args: Argument Parser
    :param args: A class containing the parameters which were given to the program as command line arguments.

    :type perc_validation_data: float
    :param perc_validation_data: The percentage of the link data which should be in the validation set.

    :type perc_test_data: float
    :param perc_test_data: The percentage of the link data which should be in the test set.

    :type return value: A sextuple. Each item is a list of string-pairs.
                        If 'perc_test_data' == None, then only a quadruple is returned.

    :return return value: The datasets D, D_C, D_valid, D_C_valid, D_test, D_C_test.
                          If 'perc_test_data' == None, then only D, D_C, D_valid, D_C_valid is returned.
                          Each of these datasets stores a list of pairs. The first string in each pair denotes the object
                          ID of the reference product and the second string denotes the object ID of the target product.

    """

    # Load datasets from memory
    D = list(load(args.linkdata_pos))
    D_C = list(load(args.linkdata_neg)) 
    len_D = len(D)
    len_DC = len(D_C)

    # Permute datasets
    np.random.seed(1234)
    D = np.random.permutation(D)
    D_C = np.random.permutation(D_C)

    # Split the data in training, validation and test set.
    end_D_valid = int(len_D*perc_validation_data)
    end_DC_valid = int(len_DC*perc_validation_data)
    end_D_test = int(end_D_valid + len_D*perc_test_data)
    end_DC_test = int(end_DC_valid + len_DC*perc_test_data)
    return D[end_D_test:], D_C[end_DC_test:], \
           D[:end_D_valid], D_C[:end_DC_valid], \
           D[end_D_valid:end_D_test], D_C[end_DC_valid:end_DC_test]

def get_strings_from_samples(reviews, samples):
    """Getting (string-)sentences from boolean sample vectors.

    :type reviews: List of strings
    :param reviews: The review sentences (belonging to one fixed product)
    :type samples: np.array, ndim=2, dtype=boolean
    :param samples: Each row of samples encodes one sample. If the i-th entry of the j-th row is 1, then the i-th sentence
    is part of the j-th sample

    :type return value: list of list of strings
    :return return value: Each item of the outer list contains a list of sentences (all sentences in a sample).
    """
    sents_strings = [[reviews[x] for x in np.nonzero(sample)[0]] for sample in samples]
    return sents_strings

def training_sample(D, D_C):
    data = map(lambda d: (d[0],d[1],1), D) + map(lambda d: (d[0],d[1],0), D_C)
    len_data = len(data)
    permutation = np.random.permutation(len_data)
    for i in xrange(len_data):
        yield data[permutation[i]]
        #yield data[i]

def _store_dataset(args, flic_dim, sent_emb_dim, ace_error, prob_pos, prob_neg, accuracy, num_samples, dataset='train'):
    data = np.vstack((accuracy, prob_pos, prob_neg, ace_error))
    DF = pd.DataFrame(data, index=['accuracy', 'prob_pos', 'prob_neg', 'ace_error']).T
    DF['dataset'] = dataset
    if args.category:
        DF['category'] = args.category
    else:
        DF['category'] = ""
        print("Warning: Category could not be added.")
    DF['model_type'] = args.model_type
    DF['context'] = args.context
    DF['test_set_num_prods'] = args.testset_has_new_products
    DF['flic_dim'] = flic_dim
    DF['sent_emb_dim'] = sent_emb_dim
    DF['num_target_sentences'] = args.num_target_sentences
    DF['num_samples_per_link'] = num_samples
    DF['regularizer'] = args.regularizer
    DF['gen_num_layers'] = args.gen_num_hidden_layers
    DF['gen_num_units'] = args.gen_num_hidden_units
    DF['enc_num_layers'] = args.enc_num_hidden_layers
    DF['enc_num_units'] = args.enc_num_hidden_units
    DF['gen_dropout'] = args.dropout_generator
    DF['enc_dropout'] = args.dropout_encoder

    return DF

def store_results_in_DF(args, flic_dim, sent_emb_dim,
                        error_train, accuracy_train, num_samp_train,
                        error_valid, accuracy_valid, num_samp_valid,
                        error_test, accuracy_test, num_samp_test):

    ace_error_train, prob_pos_train, prob_neg_train = zip(*error_train)
    ace_error_valid, prob_pos_valid, prob_neg_valid = zip(*error_valid)
    ace_error_test, prob_pos_test, prob_neg_test = zip(*error_test)

    DF_train = _store_dataset(args, flic_dim, sent_emb_dim, ace_error_train, prob_pos_train, prob_neg_train,
                        accuracy_train, num_samp_train, dataset='train')

    DF_valid = _store_dataset(args, flic_dim, sent_emb_dim, ace_error_valid, prob_pos_valid, prob_neg_valid,
                        accuracy_valid, num_samp_valid, dataset='valid')

    DF_test = _store_dataset(args, flic_dim, sent_emb_dim, ace_error_test, prob_pos_test, prob_neg_test,
                        accuracy_test, num_samp_test, dataset='test')

    DF = pd.concat((DF_train, DF_valid, DF_test))

    store(DF, args.training_output_path + 'DataFrame.pkl')
    return DF
