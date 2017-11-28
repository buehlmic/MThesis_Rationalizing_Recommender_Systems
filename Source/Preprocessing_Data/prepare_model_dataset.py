# -*- coding: utf-8 -*-
'''
    A script to construct the data set used for training the model. This data set consists of 3 parts:
    1) A dictionary mapping product ID's to a set of review sentences (input to the LSTM).
    2) A set of links from reference products to other products. This is the Dataset D in our model.
    3) A set of randomly generated link between products. This is the Dataset D^C in our model.) 
    
    @author: Michael Bühler (buehler_michael@bluewin.ch)

'''

import pandas
import pickle
import os
import argparse
import random
import functools
import collections
from nltk.tokenize import sent_tokenize
from Util.data_helper import ProcessText


raw_data_path = '../../Data/Raw_Data/'
preprocessed_data_path = '../../Data/Processed_Data/'
category_list = ['Wine']
num_reviews_per_product = 5

def parse_arguments():
    parser = argparse.ArgumentParser(description="This scripts constructs the data sets needed to train the model. In "
                                                 "particular, it constructs a dictionary mapping product ID's to a set of"
                                                 "sentences and 2 datasets D and D_C consisting of links from reference"
                                                 "products to other products.")
    parser.add_argument("-r", "--raw_data_path", help="The path to the raw data."
                                                      "(Default: '../../Data/Raw_Data/')")
    parser.add_argument("-p", "--preprocessed_data_path", help="The path where the link data is stored. The resulting "
                                                               "datasets will also be stored in this path."
                                                               "(Default: '../../Data/Processed_Data/')")
    parser.add_argument("-c", "--category_list", nargs='+', help= "A list of categories for which the data sets should be "
                                                       "constructed. (Default: Wine).")
    parser.add_argument("-nr" ,"--num_reviews_per_product", help="The number of reviews that should be selected for each"
                                                                 "product (model input) (Default: 5)", type=int)

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
    if args.num_reviews_per_product:
        global num_reviews_per_product
        num_reviews_per_product = args.num_reviews_per_product


# This function chooses the best 'num_reviews_per_product' reviews from a list of reviews.
def choose_x_reviews(review_list):
    # Helper function to decide if review a or review b is better.
    def comp(review_a, review_b):

        # looking at the 'helpful' values of the reviews
        la = review_a['helpful'][0]
        lb = review_b['helpful'][0]
        ua = review_a['helpful'][1]
        ub = review_b['helpful'][1]
        if la > 0 and lb > 0:
            # The 'helpfulness score' is calculated as the helpful-ratio plus a small bias helping often rated reviews.
            score_a = la / float(ua) + la * 0.002
            score_b = lb / float(ub) + lb * 0.002
            if score_a > score_b:
                return -1
            elif score_a < score_b:
                return 1
        elif la > 0 and la / float(ua) > 0.5:  # The 'helpfulness score' of product a is > 0.5
            return -1
        elif lb > 0 and lb / float(ub) > 0.5:  # The 'helpfulness score' of product b is > 0.5
            return 1

        # If the code reaches this point, the helpfulness score couldn't decide which product is the better review.
        # Hence we decide according to the length of the reviews.
        len_a = len(review_a['reviewText'])
        len_b = len(review_b['reviewText'])
        if len_a > len_b:
            return -1
        elif len_a < len_b:
            return 1
        else:
            return 0

    # Returns the best 'num_reviews_per_product' reviews from the 'review_list'.
    sorted_review_list =  sorted(review_list, key=functools.cmp_to_key(comp))
    if len(sorted_review_list) >= num_reviews_per_product:
        return sorted_review_list[:num_reviews_per_product]
    else:
        return sorted_review_list

# This function chooses processes 'review_data' such that each product gets a list of good reviews.
def extract_reviews(review_data):
    def obj_id_to_reviews(df):
        return df[['helpful', 'overall', 'reviewText']].to_dict('records')

    # Constructs a DataFrame where each row is indexed by an object_ID and stores a list of reviews.
    df = review_data.groupby('object_ID').apply(obj_id_to_reviews)

    # Constructs a dict mapping object ID's to a list of its best reviews.
    product_to_reviews = {obj_id: choose_x_reviews(obj_data) for obj_id, obj_data in df.iteritems()}

    # Removes the metadata ('helpful', 'overall' and 'summary') and only keeps the content of the review (i.e. the text).
    return {obj_id: list(map(lambda x : x['reviewText'], obj_data)) for obj_id, obj_data in product_to_reviews.items()}

# A tokenizer, which tokenizes a list of user reviews into a list of sentences.
def tokenize_sentences(reviews):
    process_text = ProcessText()
    sentences = []

    for review in reviews:
        # Solves problems with missing spaces after punctuation marks.
        sentences.extend(sent_tokenize(process_text.process_review(review)))

    # Returns a list where sentences with less than 4 chars are omitted.
    return list(filter(lambda x: len(x) >= 4,sentences))

# This function constructs a dict from object ID's to a set of sentences extracted from user reviews.
def construct_review_data(link_data, category):
    review_data = pandas.read_pickle(raw_data_path + '/' + category + '/reviews.pkl')

    # Extracts the most insightfull reviews for each product. The number of extracted reviews per product is given by
    # the minimum of 'num_reviews_per_product' and the number of reviews of the considered product in review_data.
    # The function returns a dictionary from product ID's to a list of reviews.
    product_to_reviews = extract_reviews(review_data)



    # Tokenizes the reviews such that each product gets represented by a list of sentences.
    # This function returns a dict mapping product ID's to a list of sentences.
    product_to_sentences = {obj_id: tokenize_sentences(obj_data) for obj_id, obj_data in product_to_reviews.items()}
    return product_to_sentences

    """ The reamining 2 lines of code in this function are only needed if the model is not trained with the pretrained
    sentence embeddings.
    # Our model assumes that the sentences associated to each product are independent of each other. Hence we return
    # a list of sentences. Each sentence is represented by a list of words.
    process_text = ProcessText()
    return {obj_id : map(process_text.parse_sentence, obj_data) for obj_id, obj_data in product_to_sentences.items()}
    """


def construct_D_and_D_C(link_data):
    D = {(ref_prod, prod) for ref_prod in link_data for prod in link_data[ref_prod]}

    # Counts how many times each product appears as a reference product.
    counts = collections.Counter()
    for (ref_prod, prod) in D:
        counts[ref_prod] += 1

    # Constructs D_C.
    D_C = []
    sample_from = set(link_data)
    for ref_prod in counts:
        counter = 0
        target_products = set()

        # We don't want to sample 'ref_prod', so we remove it.
        sample_from.remove(ref_prod)

        # We sample new links as long as we don't have sampled enough links for the reference product 'ref_prod'. We
        # want that 'ref_prod' has the same number of outgoing links in D_C as it has in D.
        # The condition 'counter < 100' makes sure that we break out when we can not sample more negative links.
        while len(target_products) < counts[ref_prod] and counter < 100:
            newly_sampled = set(random.sample(sample_from, counts[ref_prod] - len(target_products)))
            # Adds the newly sampled products to target_products if these sampled products are not yet in target_products
            # and if the resulting link (ref_prod, prod) is not in D (we only want to sample negative links).
            target_products |= (newly_sampled - link_data[ref_prod])
            counter += 1

        # We need to add the removed product 'ref_prod' again.
        sample_from.add(ref_prod)

        # We add the sampled links to the dataset D_C.
        D_C.extend({(ref_prod, prod) for prod in target_products})

    return (D, set(D_C))

def main():

    # Parse command line arguments and sets global variables.
    parse_arguments()

    # Construct the data sets for each category.
    for category in category_list:
        link_data = pickle.load(open(preprocessed_data_path + '/' + category + '/link_data.pkl', 'rb'))

        # Construct a dict mapping products to a set of sentences (input to the LSTM of our model).
        product_to_sentences = construct_review_data(link_data, category)

        # Construct the data sets D and D^C.
        # The function returns 2 datasets D and D_C. Each dataset is a set of pairs, where each pair denotes a link
        # between a reference product and another product.
        (D, D_C) = construct_D_and_D_C(link_data)

        # Pickle and store 'product_to_sentences', 'D' and 'D_C'. Note that the correct path to the files already
        # exists (otherwise we could not have loaden 'link_data'.
        pickle.dump(product_to_sentences, open(preprocessed_data_path + '/' + category + '/reviews.pkl', 'wb'))
        pickle.dump(D, open(preprocessed_data_path + '/' + category + '/D.pkl', 'wb'))
        pickle.dump(D_C, open(preprocessed_data_path + '/' + category + '/D_C.pkl', 'wb'))
        print("Category '{}' done!".format(category))

if __name__ == '__main__':
    main()


