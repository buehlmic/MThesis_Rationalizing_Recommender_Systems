# coding: utf-8
'''
    Collection of useful functions.
    
    @author: Michael Bühler (buehler_michael@bluewin.ch)

'''

import os
import cPickle as pickle
import re
from nltk.tokenize import sent_tokenize, word_tokenize

def pa(ar):
    """Print the numpy array 'ar' to the command line. The string is formatted such that he can be directly
    copy-pasted into matlab.

    :type ar: 2D numpy array
    :param ar: An array for which a 2D matlab representation should be built.

    """
    rem_new_lines = re.sub(r"\s", ' ', str(ar))
    print(re.sub(r"\],?\s? ?\[", '];\n [', rem_new_lines))

def store(obj, path):
    """Pickle and store an object at a given path. If the path doesn't exist, it gets constructed.

    :type obj: object
    :param obj: An arbitrary python object

    :type path: str
    :param path: The path where 'obj' should be stored (e.g. '/path/to/obj.pkl').

    """

    if os.path.dirname(path) and not os.path.isdir(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    f = open(path, 'wb')
    pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    f.close()

def load(path):
    """Load a pickled python object.

    :type path: str
    :param path: The path were the pickled python object is stored.

    :type return value: object
    :param return value: The python object stored at 'path'.
    """

    f = open(path, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj

def data_subset(data, p):
    """Return the p-th percentage of 'data'. Which items of 'data' get returned is determined
    by the iterator which is called on 'data'.

    :type data: set
    :param data: A set of arbitrary items_

    :type return value: The same as data
    :return return value: The p-th percentage of 'data' (i.e. about p*len(data) elements are returned).
    """

    count = 0
    num_items = len(data)
    new_data = set()
    for item in data:
        if count > num_items*p:
            break
        new_data.add(item)
        count += 1
    return new_data

class ProcessText:
    """Class for processing reviews, sentences and words. The idea of this class is also to give an interface for
    parsing sentences into words. The output of the function can for example be used as an input for a word embedding
    layer.

    """

    def __init__(self):
        pass

    def process_review(self, text):
        # Solve problems with missing spaces after punctuation marks.
        text = text.replace('.', ' . ').replace('?', ' ? ').replace('!', ' ! ').replace(',', ' , ').replace('/', ' / ')
        text = text.replace('*', '').replace(':','').replace('~','').replace(' _', ' ').replace('_ ', ' ')
        return text

    def process_sentence(self, sent):
        return sent.replace('/', ' ')

    def process_word_list(self, word_list):
        word_list = [re.sub(r"[^\w!?,.]+","", word) for word in word_list]
        return [word.lower() for word in word_list if word and not word.isdigit()]

    def parse_sentence(self, sent):
        return self.process_word_list(word_tokenize(self.process_sentence(sent)))
