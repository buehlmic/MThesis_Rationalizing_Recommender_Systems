import sys
import argparse
import cPickle as pickle
import torch

def embed_reviews(args, reviews):
    """
    :type args: argparse.ArgumentParser
    :param args: The command line argument parser
    :type reviews: Dict. The keys are strings and the values list of strings.
    :param reviews: The keys are object ID's and the values are a list of sentences (the reviews belonging to the product
    with the ID given by the key).

    :type: Dict. The keys are strings and the values a pair of np.array and a list of strings.
    :return: The keys are object ID's. The first entry of the value-pair contains the sentence embeddings (each row encodes
    1 sentence). The second entry are the list of sentences (the same as in the reviews-dict).
    """

    print("Loading embedding model.")
    model = torch.load(args.infersent_path, map_location=lambda storage, loc: storage)
    model.set_glove_path(args.gloVe_path)
    print("Building truncated vocabulary.")
    sents = [sent for product in reviews.values() for sent in product]
    model.build_vocab([sent for product in reviews.values() for sent in product], tokenize=True)
    print("Encoding sentences.")
    embedded_reviews = {}
    for product_id, sentences in reviews.iteritems():
        embedded_reviews[product_id] = (model.encode(sentences, tokenize=True), sentences)
        print("Product {} done.".format(product_id))
    return embedded_reviews, model.enc_lstm_dim*2


def load_arguments():
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("-p", "--preprocessed_data_path",
            type = str,
            default = '../../../Data/Processed_Data/',
            help="The directory where the reviews are stored. The resulting dataset will also be stored "
                 "in this path. (Default: '../../../Data/Processed_Data/')"
    )
    argparser.add_argument("-g", "--gloVe_path",
            type = str,
            default = '../../../Data/Word_Embedding/glove.840B.300d.txt',
            help="The GloVe word embedding file (Default: '../../../Data/Word_Embedding/glove.840B.300d.txt')"
    )
    argparser.add_argument("-i", "--infersent_path",
            type = str,
            default = 'infersent256.pkl',
            help="The path to the model which stores the Infersent Model"
                 " (Default: 'infersent256.pkl')"
    )

    argparser.add_argument("-c", "--category_list",
            nargs = '+',
            type = str,
            default = ['Wine'],
            help="A list of categories for which the data sets should be "
                 "constructed. (Default: Wine).")

    args = argparser.parse_args()
    return args


def main():

    # Reads command line arguments.
    args = load_arguments()

    # Construct the data sets for each category.
    for category in args.category_list:
        reviews = pickle.load(open(args.preprocessed_data_path + '/' + category + '/reviews.pkl', 'rb'))

        # Construct a dict mapping products to a set of sentences (input to the LSTM of our model).
        embedded_reviews, sent_emb_dim = embed_reviews(args, reviews)

        # Pickle and store 'product_to_sentences', 'D' and 'D_C'. Note that the correct path to the files already
        # exists (otherwise we could not have loaden 'link_data'.
        pickle.dump(embedded_reviews, open(args.preprocessed_data_path + '/' + category + '/reviews_embedded_' +
                                           str(sent_emb_dim) + '.pkl', 'wb'))
        print("Category '{}' done!".format(category))

if __name__ == '__main__':
    main()

