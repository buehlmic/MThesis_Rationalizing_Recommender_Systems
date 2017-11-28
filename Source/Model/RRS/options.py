# coding: utf-8

"""
  File to handle the input arguments for 'rss.py'.

"""

import sys
import argparse
import warnings
import os

def load_arguments():
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--evaluate_model",
        default = False,
        action = 'store_true',
        help = "If this argument is present, the model will not get trained, but get evaluated on the test set. "
               "Don't forget to add the arguments 'load_model', 'category' and 'perc_test_data' to some reasonable values."
    )
    argparser.add_argument("--testset_has_new_products",
        default = False,
        action = 'store_true',
        help = "If the testset should consist of new products or not."
    )
    argparser.add_argument("--linkdata_pos",
        type = str,
        default = "",
        help = "The path to training data."
    )
    argparser.add_argument("--linkdata_neg",
        type = str,
        default = "",
        help = "The path to positive part of the training data (Dataset D)."
    )
    argparser.add_argument("--flic",
        type = str,
        default = "",
        help = "The path to negative part of the training data (Dataset D_C)."
    )
    argparser.add_argument("--reviews",
        type = str,
        default = "",
        help = "The path to the dictionary which maps product ID's to a set of sentences."
    )
    argparser.add_argument("--training_output_path",
        type = str,
        default = "",
        help = "The path where the training information should be outputted (Only valid when the flag "
               "'--evaluate_model' is set as well)."
    )
    argparser.add_argument("--num_samples",
        type = int,
        default = 1,
        help = "The number of sets which should be sampled from the generator in each iteration."
    )
    argparser.add_argument("--max_epochs",
        type = int,
        default = 5000,
        help = "The maximum number of epochs"
    )
    argparser.add_argument("--num_cross_validation",
        type = int,
        default = 4,
        help = "The number k in cross-validation (number of passes)."
    )
    argparser.add_argument("--enc_num_hidden_layers",
        type = int,
        default = 1,
        help = "The number of hidden layers in the Deep Sets Architecture."
    )
    argparser.add_argument("--enc_num_hidden_units",
        type = int,
        default = 4,
        help = "The number of hidden units in each permutation equivariant layer of Deep Sets."
    )
    argparser.add_argument("--gen_num_hidden_layers",
        type = int,
        default = 2,
        help = "The number of hidden layers in the Feed Forward Architecture of the Generator."
    )
    argparser.add_argument("--gen_num_hidden_units",
        type = int,
        default = 4,
        help = "The number of hidden units in each layer of the Feed Forward Architecture of the Generator."
    )
    argparser.add_argument("--num_target_sentences",
        type = float,
        default = 2,
        help = "The average (target) number of sentences that should be sampled in each set of the Generator."
    )

    argparser.add_argument("--regularizer",
        type = float,
        default = 0.01,
        help = "The multiplicative regularizing factor for the cost of the set size. Recommended values are in"
               "the range of [0.01, 0.001]."
    )
    argparser.add_argument("--l2_enc_reg",
        type = float,
        #default = 1e-6,
        default = 0,
        help = "L2 regularization weight for the parameters in the encoder. (Not implemented)"
    )
    argparser.add_argument("--l2_gen_reg",
        type = float,
        #default = 1e-6,
        default = 0,
        help = "L2 regularization weight for the parameters in the generator. (Not implemented)"
    )
    argparser.add_argument("--learning_rate_generator",
        type = float,
        default = 0.0001,
        help = "learning rate"
    )
    argparser.add_argument("--learning_rate_encoder",
        type = float,
        default = 0.001,
        help = "learning rate"
    )
    argparser.add_argument("--beta",
        type = float,
        default = 0.01
    )
    argparser.add_argument("--learning",
        type = str,
        default = "adam",
        help = "learning method"
    )
    argparser.add_argument("--dropout_generator",
            type = float,
            default = 0.00,
            help = "Dropout probability for the generator. (Not yet implemented)."
    )
    argparser.add_argument("--dropout_encoder",
            type = float,
            default = 0.00,
            help = "Dropout probability for the generator. (Not yet implemented)."
    )

    argparser.add_argument("--save_model",
            type = str,
            default = "Models/",
            help = "Directory to save model parameters."
    )

    argparser.add_argument("--load_model",
            type = str,
            default = "",
            help = "Path to load model parameters."
    )
    argparser.add_argument("--perc_validation_data",
            type = float,
            default = 0.2,
            help = "The percentage of links from the training data which should be used for validation."
    )
    argparser.add_argument("--perc_test_data",
            type = float,
            default = 0.2,
            help = "The percentage of links from the training data which should be used as test data."
    )
    argparser.add_argument("--num_iters_between_validation",
        type = int,
        default = 10000,
        help = "The number of iterations before the validation set should be evaluated."
    )
    argparser.add_argument("--measure_timing",
        default = False,
        action = 'store_true',
        help = "If this argument is present, the runtime of the generator / sampling / encoder is measured "
               "during execution and printed out after every epoch."
    )
    argparser.add_argument("--sample_all_sentences",
        default=False,
        action='store_true',
        help="If this argument is present, the program always uses all sentences for the Encoder."
    )
    argparser.add_argument("--adaptive_lrs",
        default=False,
        action='store_true',
        help="If this argument is present, the program uses an adaptive learning rate. (I.e. it looks at the "
             "validation error to decide when the learning rate should be made smaller.) Additionally, the program"
             "stops when the learning rate is under the threshold 1e-6."
    )
    argparser.add_argument("--model_type",
            type = str,
            default = "DPP",
            help = "How to sample sentence sets from the Generator. Possible options are 'DPP' (default), "
                   "'Unif_Sampling', 'Independent', 'Logreg_Baseline', 'Logreg_Baseline_1_layer, "
                   "'Logreg_SentEmbs'. 'Logreg_SentEmbs_1_layer'"
    )
    argparser.add_argument("--category",
            type = str,
            default = None,
            help = "The category, which is trained. This is needed for the output in a Pandas Dataframe. "
                   "This flag needs only be set if the flag --evaluate_model is given as well."
    )
    argparser.add_argument("--context",
           type = str,
            default = "train_context",
            help = "If to train the contex (i.e. transformation of the context before sampling or rather to train the "
                   "sentence embeddings (i.e. transformation of the sentence embeddings before sampling and no "
                   "context transformation). Possible options are 'train_context', 'train_sent_embs' and "
                   "'no_context. The latter is like 'train_sent_embs', but uses constant 1-vectors instead of FLIC-vectors"
                   "for the context in the generator."
    )
    argparser.add_argument("--print_most_frequent_sentences",
                           default=False,
                           action='store_true',
                           help="If this argument is present, the model will not get trained, but get evaluated on a small"
                                "test set. For each link it will rank all sampled sentences according to their relative "
                                "frequency in the sampled sets."
                           )

    args = argparser.parse_args()

    # Exceptions
    if not args.linkdata_pos:
        raise(TypeError, "No path given!")
    if not args.linkdata_neg:
        raise(TypeError, "No path given!")
    if not args.reviews:
        raise(TypeError, "No path given!")
    if not args.flic:
        raise(TypeError, "No path given!")
    if args.evaluate_model and not args.training_output_path:
        warnings.warn("No directory to store the output of the training during cross-validation is given!")
    if  args.model_type != 'DPP' and args.model_type != 'Independent' and args.model_type != 'Unif_Sampling' and \
        args.model_type != 'Logreg_Baseline' and args.model_type != 'Logreg_SentEmbs' and \
        args.model_type != 'Logreg_Baseline_1_layer' and args.model_type != 'Logreg_SentEmbs_1_layer':
        raise(TypeError, "Model_type must be one of {DPP, Independent, Unif_Sampling, Logreg_Baseline, Logreg_SentEmbs,"
                         "Logreg_Baseline_1_layer, Logreg_SentEmbs_1_layer}.")
    if args.evaluate_model and not args.category:
        warnings.warn("You did not specify a category, eventhough you gave the --evaluate_model flag.")
    if args.context != 'train_context' and args.context != 'no_context' and args.context != 'train_sent_embs':
        raise(TypeError, "Context must be one of {train_context, no_context, train_sent_embs}.")


    # Create directory to store model
    if args.save_model:
        if not os.path.isdir(args.save_model):
            os.makedirs(args.save_model)
            open(args.save_model + '.gitignore', 'a').close()

    return args
