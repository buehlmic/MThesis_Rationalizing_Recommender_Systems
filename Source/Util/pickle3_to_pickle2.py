# coding: utf-8
'''
    Script to transform pickle objects from format 3 to format 2. This is needed because python 2 is not able to read
    pickle objects with format 3.

    @author: Michael Bühler (buehler_michael@bluewin.ch)

'''


import pickle
import pandas
import argparse

path = ''

def parse_arguments():
    parser = argparse.ArgumentParser(description="This scripts transforms a pickle object from format 3 to format 2.")
    parser.add_argument("-p","--pickle_file", help="The path to the pickle file that should be transformed.")

    args = parser.parse_args()
    if args.pickle_file:
        global path
        path = args.pickle_file


def main():

    # Parses command line arguments
    parse_arguments()

    pandas_object = pandas.read_pickle(path)
    pickle.dump(pandas_object, open(path, 'wb'), protocol=2)


if __name__ == '__main__':
    main()
