# -*- coding: utf-8 -*-
'''
    A script to construct the data set for training the FLIC model. Additionally, it constructs link data for the model
    (i.e. the data which describes for each product its connection (links) to other products in the same category.
    
    @author: Michael Bühler (buehler_michael@bluewin.ch)

'''

import pandas
import pickle
import os
import argparse
import random

raw_data_path = '../../Data/Raw_Data/'
preprocessed_data_path = '../../Data/Processed_Data/'
category_list = ['Wine']
min_num_links = 5               # Minimum number of links each product needs to have to other products in the dataset.
min_num_sets_per_product = 10   # Minimum number of sets each product needs to be in (in the training data).
min_num_products_in_set = 3     # Minimum number of products in each set of the training data.
min_num_reviews_per_product = 5 # Minimum number of reviews each product needs to have.

def parse_arguments():
    parser = argparse.ArgumentParser(description="This scripts constructs the FLIC training data. Additionally, it "
                                                 "constructs the link data for the model (i.e. the data which describes "
                                                 "for each product its connection to other products.)")
    parser.add_argument("-r", "--raw_data_path", help="The path to the raw data."
                                                      "(Default: '../../Data/Raw_Data/')")
    parser.add_argument("-p", "--preprocessed_data_path", help="The path where the resulting data (link and FLIC data) "
                                                            "should be stored. (Default: '../../Data/Processed_Data/')")
    parser.add_argument("-c", "--category_list", nargs='+', help= "A list of categories for which the data sets should be "
                                                       "constructed. (Default: Wine).")
    parser.add_argument("-l" ,"--min_num_links", help="Minimum number of links each product needs to have to other "
                                                       "products in the dataset. (Default: 5)", type=int)
    parser.add_argument("-s" ,"--min_num_sets_per_product", help="Minimum number of sets each product needs to be in "
                                                                  "(in the training dataset). (Default: 10)", type=int)
    parser.add_argument("-m" ,"--min_num_products_in_set", help="Minimum number of products in each set of the "
                                                                "training data. (Default: 3)", type=int)
    parser.add_argument("-reviews", "--min_num_reviews_per_product", help="Minimum number of reviews each product "
                                                                      "needs to have (Default: 5)", type=int)

    args = parser.parse_args()
    if args.raw_data_path:
        global raw_data_path
        raw_data_path = args.raw_data_path
    if args.preprocessed_data_path:
        global preprocessed_data_path
        preprocessed_data_path = args.preprocessed_data_path
    if args.category_list:
        global category_list
        category_list = args.category_list
    if args.min_num_links:
        global min_num_links
        min_num_links = args.min_num_links
    if args.min_num_sets_per_product:
        global min_num_sets_per_product
        min_num_sets_per_product = args.min_num_sets_per_product
    if args.min_num_products_in_set:
        global min_num_products_in_set
        min_num_products_in_set = args.min_num_products_in_set
    if args.min_num_reviews_per_product:
        global min_num_reviews_per_product
        min_num_reviews_per_product = args.min_num_reviews_per_product


def count_links(product):
    """ Calculates, how many outgoing link this particular product has. For this, it sums up the number of elements in
        the list ['also_bought', 'also_viewed', 'bought_together', 'buy_after_viewing'].

    Input: product: One row of a metadata Dataframe

    Return Value: An integer denoting how many outgoing links this product has in the metadata dataset."""

    def length(l):
        return len(l) if l else 0

    return product[['also_bought', 'also_viewed', 'bought_together', 'buy_after_viewing']].map(length).sum()

def merge_link_lists(metadata):

    # Merges one product (i.e. one row of metadata)
    def merge(ser):
        tmp = ser[['also_bought', 'also_viewed', 'bought_together', 'buy_after_viewing']].map(lambda x: x if x else [])
        merged = [tmp['also_bought'], tmp['also_viewed'],
                  tmp['bought_together'], tmp['buy_after_viewing']]
        res = [item for l in merged for item in l]
        return set(res)

    return metadata.apply(merge, axis='columns').to_dict()

def remove_sparse_products(metadata):
    objects_to_handle = set()

    # The inverted index stores links pointing the other way around than in
    # metadata. This is needed to update products, which loose a link because
    # the target product of the link gets deleted.
    inv_index = dict((obj_id, []) for obj_id in metadata)

    for obj_id, links in metadata.items():
        objects_to_handle.add(obj_id)

        # Removes broken links
        metadata[obj_id] = set([l for l in links if l in metadata])

        # Fills inverted index with back links.
        for link in metadata[obj_id]:
            inv_index[link].append(obj_id)

    # Recursively removes elements with < min_num_links links
    while objects_to_handle:
        obj_id = objects_to_handle.pop()
        if (obj_id in metadata and len(metadata[obj_id]) < min_num_links):
            del metadata[obj_id]
            for link in inv_index[obj_id]:
                if link in metadata:
                    metadata[link].discard(obj_id)
                    objects_to_handle.add(link)

    return metadata

def sample_sets(objects_to_links):
    dataset = set()
    for obj_id in objects_to_links:
        num_sets = 0
        counter = 0

        # For each product, 'min_num_sets_per_product' sets get sampled and added to 'dataset'.
        while num_sets < min_num_sets_per_product:
            length = random.randint(min(min_num_links, min_num_products_in_set), len(objects_to_links[obj_id]))
            sampled_set = set(random.sample(objects_to_links[obj_id], length))
            sampled_set.add(obj_id)
            if sampled_set not in dataset:
                num_sets += 1
                dataset.add(frozenset(sampled_set))

            # For the rare case that it is not possible to sample new sets anymore (because all possible sets are
            # already in 'dataset'), we allow early stopping.
            counter += 1
            if counter == 100:
                break
    return dataset

def construct_FLIC_data(category):
    metadata = pandas.read_pickle(raw_data_path + '/' + category + '/metadata.pkl')

    # Removes products with not enough reviews
    metadata = metadata[metadata['num_Reviews'] > 5]

    # Merges the 4 link lists (['also_bought', 'also_viewed', 'bought_together', 'buy_after_viewing']).
    # The result is a dict from object ID's to a set of links.
    objects_to_links = merge_link_lists(metadata)

    # Removes products with less than 'min_num_links' outgoing links. The set of links associated with each product
    # gets updated accordingly as well.
    objects_to_links = remove_sparse_products(objects_to_links)

    # For each product, it samples a few training sets constructed from the outgoing link list.
    random.seed(0)
    dataset = sample_sets(objects_to_links)
    return (objects_to_links, dataset)

def main():

    # Parses command line arguments and sets global variables.
    parse_arguments()

    # Constructs the data sets for each category.
    # 'object_to_links' is a dict from object_ids to outgoing links. Here, it is guaranteed that all link exists
    # in the category (i.e. no outgoing links to products from other categories). Furthermore each product has at least
    # 5 outgoing links.
    # 'dataset' is a set consisting of sets of products, which can be used for training FLIC.
    for category in category_list:

        (object_to_links, dataset) = construct_FLIC_data(category)

        # Construct the folder to store the data sets if it doesn't exist yet.
        path = preprocessed_data_path + '/' + category + '/'
        if not os.path.isdir(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
            open(os.path.dirname(path) + '/.gitignore', 'a').close()

        # Pickles and stores 'object_to_links' and 'dataset'.
        pickle.dump(object_to_links, open(path + 'link_data.pkl', 'wb'))
        pickle.dump(dataset, open(path + 'FLIC_sets.pkl', 'wb'))
        print("Category '{}' done!".format(category))


if __name__ == '__main__':
    main()
