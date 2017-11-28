import options
from Model.FLIC.flic import FLIC
from Util.data_io import CrossValidationProducts, CrossValidationLinks, load_link_data, load_reviews, \
                         load_flic_data, get_strings_from_samples, store_results_in_DF
from Util.nn.basic import EmbeddingLayer
from Util.data_helper import store, load
from Util.user_signals import GracefullExit
import theano
import numpy as np
from multiprocessing import current_process, Process, Manager
from Util.no_daemon_pool import NoDaemonPool
import random
import os
import sys
import time

def _train_and_evaluate(D, D_valid, D_test, D_C, D_C_valid, D_C_test, flic, prod_to_sent, args, sent_emb_dim, k):
    """ A function to train and evaluate 1 model.

    This function should only be called by main(.).

    :param D, D_valid, D_test: Training, Validation and Test Set from the D dataset.
    :param D_C, D_C_valid, D_C_test: Training, Validation and Test Set from the D_C dataset.
    :type flic: FLIC
    :param flic: The FLIC model trained on the product data set
    :type prod_to_sent: dict: string -> Set of sentences
    :param prod_to_sent: A dict from product ID's to a set of (sentence, sent_embedding) pair.
    :type args: Argument parser
    :param args: Arguments from the command line.
    :type sent_emb_dim: int
    :param sent_emb_dim: The dimension of the sentence embedding vectors.
    :type k: int
    :param k: The ID of the process (with respect to the cross-validation loop)

    :return values: Errors, accuracies, and the sets of sampled sentences from the training, validation and test set.
    """

    # File to print the training output
    if args.training_output_path:
        path = args.training_output_path
        try:
            if not os.path.isdir(path):
                os.makedirs(path)
                open(path + '.gitignore', 'a').close()
        except OSError as e:
            pass
    else:
        path = ""
    file = open(path + 'train_output_cv' + str(k) + '.txt', 'wb+')
    sys.stdout = file

    # Print process ID (useful to kill the process)
    print("I am process {}.".format(os.getpid()))
    print(args)

    # Choose Model type
    if args.model_type == 'DPP':
        from DPP_Model.model_dpp import Model
    elif args.model_type == 'Independent':
        from Independent_Model.model_indep import Model
    elif args.model_type == 'Unif_Sampling':
        from UnifSample_Model.model_unif import Model
    elif args.model_type == 'Logreg_Baseline' or args.model_type == 'Logreg_Baseline_1_layer':
        from LogReg_Baseline.model_logreg_baseline import Model
    elif args.model_type == 'Logreg_SentEmbs' or args.model_type == 'Logreg_SentEmbs_1_layer':
        from LogReg_SentEmbs.model_Logreg_sentEmbs import Model
    else:
        raise ValueError("Model Type {} not known.".format(str(args.model_type)))
    print("We use the following model type: {}".format(args.model_type))

    # Train model
    if not args.load_model:
        best_val_error = train_model(args, sent_emb_dim, flic, prod_to_sent, D, D_C, D_valid, D_C_valid)
        print('best_val_error = {}'.format(best_val_error))
        model = Model(args, sent_emb_dim, flic.dim_att)
        model.load(args.save_model + '/best_valerr_' + str("{0:3f}.pkl".format(best_val_error)))
    # Load model
    else:
        model = Model(args, sent_emb_dim, flic.dim_att)

    def average_sample_size(sampled_sents):
        return np.mean([len(samp) for samples in map(lambda x: x[2], sampled_sents) for samp in samples])

    def std_sample_size(sampled_sents, average_sample_size):
        return np.sqrt(np.mean([(len(samp)-average_sample_size)**2 for samples in map(lambda x: x[2], sampled_sents) for samp in samples]))


    print("Evaluating model...")

    # Training data
    samples_train, accuracy_train, error_train = model.testset_evaluation(D, D_C, prod_to_sent, flic,
                                                                          num_samples=10)

    # Validation data
    samples_valid, accuracy_valid, error_valid = model.testset_evaluation(D_valid, D_C_valid, prod_to_sent, flic,
                                                                          num_samples=10)

    # Test data
    samples_test, accuracy_test, error_test = model.testset_evaluation(D_test, D_C_test, prod_to_sent, flic,
                                                                       num_samples=10)

    print("Average sample size on the training set = {}".format(average_sample_size(samples_train)))
    print("Standard deviation on the training set = {}".format(std_sample_size(samples_train,
                                                                               average_sample_size(samples_train))))
    print("Average sample size on the validation set = {}".format(average_sample_size(samples_valid)))
    print("Standard deviation on the validation set = {}".format(std_sample_size(samples_valid,
                                                                                 average_sample_size(samples_valid))))
    print("Average sample size on the test set = {}".format(average_sample_size(samples_test)))
    print("Standard deviation on the validation set = {}".format(std_sample_size(samples_test,
                                                                                 average_sample_size(samples_test))))

    # Reset stdout to terminal output
    sys.stdout = sys.__stdout__
    file.close()

    return [error_train, samples_train, accuracy_train,
            error_valid, samples_valid, accuracy_valid,
            error_test, samples_test, accuracy_test]


def main():

    # Setting theano environment flags
    theano.config.floatX = 'float64'
    theano.config.optimizer = 'fast_run'
    theano.config.exception_verbosity = 'high'
    args = options.load_arguments()
    print(args)

    # Loading data
    print("Loading Review data.")
    prod_to_sent = load_reviews(args)
    flic = FLIC()
    load_flic_data(args, flic)
    sent_emb_dim = np.shape(prod_to_sent.values()[0][0])[1]

    # Evaluate model using cross validation
    if args.evaluate_model:

        # If training gets cancelled using the SIGINT exception, the model will still be evaluated on the testsets.
        signal_handler = GracefullExit(parent=True)

        # The following lists store the errors, accuracies and num_samples_average of the k processes (from the k-fold
        # cross validation).
        error_train_l = []
        error_valid_l = []
        error_test_l = []

        accuracy_train_l = []
        accuracy_valid_l = []
        accuracy_test_l = []

        num_samp_train_l = []
        num_samp_valid_l = []
        num_samp_test_l = []

        k = args.num_cross_validation  # The number of runs in the crossvalidation loop (k-fold cross validation)
        # The products in the testset is disjoint from the products in the training and validation setes.
        if args.testset_has_new_products:
            cross_validation = CrossValidationProducts(k=k, args=args, perc_validation_data= 0.15)
        else:
            cross_validation = CrossValidationLinks(k=k, args=args, perc_validation_data=0.5)

        # Parallelize cross validation runs
        pool = NoDaemonPool(processes=k)
        rand_generator = random.Random()
        rand_generator.jumpahead(random.randint(1, 10000000))

        # A function to store the return values of the k processes (from the k-fold cross validation).
        def store_results(return_value):
            error_train, samples_train, accuracy_train, \
            error_valid, samples_valid, accuracy_valid, \
            error_test, samples_test, accuracy_test = return_value

            def average_sample_size(sampled_sents):
                return np.mean([len(samp) for samples in map(lambda x: x[2], sampled_sents) for samp in samples])

            error_train_l.append(error_train)
            num_samp_train_l.append(average_sample_size(samples_train))
            accuracy_train_l.append(accuracy_train)

            error_valid_l.append(error_valid)
            num_samp_valid_l.append(average_sample_size(samples_valid))
            accuracy_valid_l.append(accuracy_valid)

            error_test_l.append(error_test)
            num_samp_test_l.append(average_sample_size(samples_test))
            accuracy_test_l.append(accuracy_test)

        print("Starting Cross-Validation. I am process {}.".format(os.getpid()))
        k_0 = 0

        # Cross validation loop
        for D, D_valid, D_test, D_C, D_C_valid, D_C_test in cross_validation.next_sample():
            rand_generator.jumpahead(random.randint(1, 10000000))
            pool.apply_async(_train_and_evaluate,
                             (D, D_valid, D_test, D_C, D_C_valid, D_C_test,
                                   flic, prod_to_sent, args, sent_emb_dim, k_0),
                             callback=store_results)
            k_0 += 1

        pool.close()
        pool.join()
        sys.stdout = sys.__stdout__

        # Store the results in a Pandas DataFrame.
        store_results_in_DF(args, flic._dim_att, sent_emb_dim,
                            error_train_l, accuracy_train_l, num_samp_train_l,
                            error_valid_l, accuracy_valid_l, num_samp_valid_l,
                            error_test_l, accuracy_test_l, num_samp_test_l)

        # Console output for fast interpretation.
        print("Train error = {}".format(error_train_l))
        print("Mean = {}".format(np.array(error_train_l).mean(axis=0)))
        print("Std = {}".format(np.array(error_train_l).std(axis=0)))
        print("Average sample size = {}".format(np.mean(num_samp_train_l)))
        print("Accuracy = {}\n".format(accuracy_train_l))

        print("Validation error = {}".format(error_valid_l))
        print("Mean = {}".format(np.array(error_valid_l).mean(axis=0)))
        print("Std = {}".format(np.array(error_valid_l).std(axis=0)))
        print("Average sample size = {}".format(np.mean(num_samp_valid_l)))
        print("Accuracy = {}\n".format(accuracy_valid_l))

        print("Test error = {}".format(error_test_l))
        print("Mean = {}".format(np.array(error_test_l).mean(axis=0)))
        print("Std = {}".format(np.array(error_test_l).std(axis=0)))
        print("Average sample size = {}".format(np.mean(num_samp_test_l)))
        print("Accuracy = {}\n".format(accuracy_test_l))

    # Train the data
    else:
        # Loads the data
        print("Load training Data.")
        D, D_C, D_valid, D_C_valid, D_test, D_C_test = load_link_data(args, perc_validation_data=args.perc_validation_data,
                                                                      perc_test_data= args.perc_test_data)
        best_val_error = train_model(args, sent_emb_dim, flic, prod_to_sent, D, D_C, D_valid, D_C_valid)

        print("Best validation error = {}".format(best_val_error))

def train_model(args, sent_emb_dim, flic, prod_to_sent,
                D, D_C, D_valid, D_C_valid):
    """This function is a wrapper for model.train() to make sure that not too much memory is leaked inside of Theano.
    This is achieved by reconstructing the computation graph after 10 passes through the data."""

    # Manager() is a class for handling shared memory between this process and the childrens it constructs.
    manager = Manager()
    result_list = manager.list([None, 100, 0, "10_epochs"])

    # We reload the model after 10 epochs to make the amount of leaked memory negligible.
    while result_list[3] == "10_epochs":
        p = Process(target=_train_10_epochs,
                    args=(args, sent_emb_dim, flic, prod_to_sent, D, D_C, D_valid, D_C_valid, result_list))
        p.start()
        p.join()

    # Return best validation error
    return result_list[1]

def _train_10_epochs(args, sent_emb_dim, flic, prod_to_sent,
                   D, D_C, D_valid, D_C_valid, result_list):
    """ Function to train 10 passes through the data.
    This function should only be called by train_model(.)! """

    # Unpack some values we need for training
    load_model, best_val_error_old, num_epochs, end_string = result_list

    # Train Model

    # Choose Model type
    if args.model_type == 'DPP':
        from DPP_Model.model_dpp import Model
    elif args.model_type == 'Independent':
        from Independent_Model.model_indep import Model
    elif args.model_type == 'Unif_Sampling':
        from UnifSample_Model.model_unif import Model
    elif args.model_type == 'Logreg_Baseline' or args.model_type == 'Logreg_Baseline_1_layer':
        from LogReg_Baseline.model_logreg_baseline import Model
    elif args.model_type == 'Logreg_SentEmbs' or args.model_type == 'Logreg_SentEmbs_1_layer':
        from LogReg_SentEmbs.model_Logreg_sentEmbs import Model
    else:
        raise ValueError("Model Type {} not known.".format(str(args.model_type)))

    if load_model:
        model = Model(args, sent_emb_dim, flic.dim_att,
                      load_model=load_model,
                      epochs_done=num_epochs)
        valid_cost, end_string = model.train(D, D_C, D_valid, D_C_valid, flic,
                                                 prod_to_sent)

    else:
        model = Model(args, sent_emb_dim, flic.dim_att,
                      load_model=None,
                      epochs_done=num_epochs)
        valid_cost, end_string = model.train(D, D_C, D_valid, D_C_valid, flic,
                                                prod_to_sent)

    # Collect return values
    result_list[0] = args.save_model + '/after_10_epochs_' + str("{0:3f}.pkl".format(valid_cost))
    result_list[1] = valid_cost
    result_list[2] = num_epochs + 10
    result_list[3] = end_string


if __name__ == '__main__':
    main()
