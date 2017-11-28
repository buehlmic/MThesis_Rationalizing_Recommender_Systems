# -*- coding: utf-8 -*-
'''
    This scripts splits the products and reviews from the data sets 'metadata.json.gz' and 'kcore_5.json.gz' up into
    product categories and stores the resulting data to the disk.
    
    @author: Michael Bühler (buehler_michael@bluewin.ch)

'''

import json
import gzip
import pandas as pd
import pickle
import sys
import os
import argparse


# Path variables
data_path = '../../Data/'
raw_data_path = data_path + 'Raw_Data/'

# How many products and reviews should be loaded? (>=2**50 => All products/reviews loaded for sure)
max_num_products = 2**50
max_num_reviews = 2**50

def parse_arguments():
    parser = argparse.ArgumentParser(description="This scripts splits the products and reviews from the data sets "
                                                 "'metadata.json.gz' and 'kcore_5.json.gz' up into "
                                                 "product categories and stores the resulting data to the disk.")
    parser.add_argument("-np", "--num_products", help="How many products should be read in from the metadata file? "
                                                      "(Default: All products)", type=int)
    parser.add_argument("-nr" ,"--num_reviews", help="How many reviews should be read in? "
                                                     "(Default: All Reviews)", type=int)
    parser.add_argument("-p", "--path", help="The path storing metadata.json.gz and kcore_5.json.gz. "
                                             "(Default: ../../Data/Raw_Data/")
    args = parser.parse_args()
    if args.num_products:
        global max_num_products
        max_num_products = args.num_products
    if args.num_reviews:
        global max_num_reviews
        max_num_reviews = args.num_reviews
    if args.path:
        global raw_data_path
        raw_data_path = args.path+'/'


# Generates JSON data (returns 1 data point per call to this function).
def parse(path):
    file = gzip.open(path, 'r')
    for line in file:
        yield eval(line)

# Updates 'category_dict' with the information about 'product'.
def fill_product_lists(product, category_products, products_to_category):
    if ('categories' not in product or 'asin' not in product or 'related' not in product): # Product not useful
        return

    # Fill a new list with information about the product to append to it the correct lists in category_lists.
    prod_info = list()
    taglist = list(['title'])
    for tag in taglist:
        if tag in product:
            prod_info.append(product[tag])
        else:
            prod_info.append(None)
    prod_info.append(0) # num_reviews (will be filled in later)
    related_taglist = list(['also_bought', 'also_viewed', 'bought_together', 'buy_after_viewing'])
    for tag in related_taglist:
        if tag in product['related']:
            prod_info.append(product['related'][tag])
        else:
            prod_info.append(None)

    # Enter the information about the product in all lists with the right categories.
    for categs in product['categories']:
        category = categs[0]

        if category == '':
            continue

        # If we haven't seen the category yet, we add a new empty dict for this category.
        if category not in category_products:
            category_products[category] = dict()

        # Adds the constructed list of all relevant product information to the category products.
        category_products[category][product['asin']] = prod_info

        # products_to_category stores the information to which categories a product belongs
        if product['asin'] not in products_to_category:
            products_to_category[product['asin']] = set([category])
        else:
            products_to_category[product['asin']].add(category)

def fill_review_lists(review, category_products, products_to_category, category_to_reviews):
    if 'asin' not in review or review['asin'] not in products_to_category or \
                               'reviewText' not in review:
        return

    review_info = list()
    taglist = list(['asin', 'overall', 'reviewText', 'summary', 'helpful'])
    for tag in taglist:
        if tag in review:
            review_info.append(review[tag])
        else:
            review_info.append(None)

    for category in products_to_category[review['asin']]:
        category_products[category][review['asin']][1] += 1
        category_to_reviews[category].append(review_info)

def main():
    num_products_metadata = 0               # How many products (metadata) are already processed?
    num_reviews = 0                         # How many reviews are already processed?

    # Parsing command line arguments
    parse_arguments()

    #-------------------------------------------------------------------------------------------------------------------
    # Meta Data Processing
    #-------------------------------------------------------------------------------------------------------------------

    # One product after another, its metadata gets processed.
    category_products = dict()  # A dictionary mapping categories to lists to store the metadata of the products
    products_to_category = dict()
    for product in parse(raw_data_path+'metadata.json.gz'):
        fill_product_lists(product, category_products, products_to_category)
        num_products_metadata += 1
        if num_products_metadata >= max_num_products:
            break


    #-------------------------------------------------------------------------------------------------------------------
    # Review Processing
    #-------------------------------------------------------------------------------------------------------------------

    category_to_reviews = {category : list() for category in category_products.keys()}
    for review in parse(raw_data_path+'kcore_5.json.gz'):
        fill_review_lists(review, category_products, products_to_category, category_to_reviews)
        num_reviews += 1
        if num_reviews >= max_num_reviews:
            break


    #-------------------------------------------------------------------------------------------------------------------
    # Stores the metadata and the reviews in Pandas DataFrames.
    #-------------------------------------------------------------------------------------------------------------------

    # Copies the Meta Data to Pandas Dataframes. 'metadata' is a dictionary, which maps categories to the products
    # belonging to this category (stored in a Pandas Dataframe).
    metadata = dict()
    for (category, products) in category_products.items():
        metadata[category] = pd.DataFrame.from_dict(products, 'index')
        metadata[category].columns = ['title', 'num_Reviews', 'also_bought', 'also_viewed',
                                      'bought_together', 'buy_after_viewing' ]
        metadata[category].index.name = 'object_ID'
        metadata[category] = metadata[category].sort_index(ascending=True)

    # Copies the Review Data to Pandas Dataframes. 'review_data' is a dictionary, which maps categories to the reviews
    # belonging to this category (stored in a Pandas Dataframe).
    review_data = dict()
    for (category, products) in category_to_reviews.items():
        review_data[category] = pd.DataFrame(products, columns = ['object_ID', 'overall', 'reviewText',
                                                                  'summary', 'helpful']).set_index('object_ID')
        # Randomly permuting the rows
        if review_data[category].shape[0] > 1:
            review_data[category] = review_data[category].sample(frac=1)

    #-------------------------------------------------------------------------------------------------------------------
    # Stores the Pandas DataFrames on the disk.
    #-------------------------------------------------------------------------------------------------------------------

    for category in category_products:
        path_to_store = raw_data_path + category +'/'

        # Creates the directory if it doesn't exist yet
        if not os.path.isdir(path_to_store):
            os.makedirs(path_to_store)
            open(path_to_store +'.gitignore', 'a').close()      # This will allow git to commit the new folder
                                                                # even without other files in it.

        # Removes duplicates in the metadata and the reviews and stores the resulting Pandas DataFrames on disk.
        metadata_duplicates_removed = metadata[category][~metadata[category].index.duplicated(keep='first')]
        metadata_duplicates_removed.to_pickle(path_to_store + 'metadata.pkl')
        review_data_duplicates_removed = review_data[category][~(review_data[category].index.duplicated(keep='first') &
                                                       review_data[category]['reviewText'].duplicated(keep='first'))]
        review_data_duplicates_removed.to_pickle(path_to_store + 'reviews.pkl')


    # Stores the files for the products of all categories

    # Creates the directory if it doesn't exist yet
    path_to_store = raw_data_path + 'All_Categories/'
    if not os.path.isdir(path_to_store):
        os.makedirs(path_to_store)
        open(path_to_store + '.gitignore', 'a').close()

    # Removes duplicates in the metadata and the reviews and stores the resulting Pandas DataFrames on disk.
    metadata_total = pd.concat(metadata.values())
    metadata_total_duplicates_removed = metadata_total[~metadata_total.index.duplicated(keep='first')]
    metadata_total_duplicates_removed.to_pickle(path_to_store + 'metadata.pkl');

    reviews_total = pd.concat(review_data)
    reviews_total_duplicates_removed = reviews_total[~(reviews_total.index.duplicated(keep='first') &
                                                       reviews_total['reviewText'].duplicated(keep='first'))]
    if reviews_total_duplicates_removed.shape[0] > 1:
        reviews_total_duplicates_removed = reviews_total_duplicates_removed.sample(frac=1.0)
    reviews_total_duplicates_removed.to_pickle(path_to_store + 'reviews.pkl');

if __name__ == '__main__':
    main()
