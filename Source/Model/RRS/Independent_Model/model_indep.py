from generator_indep import Generator
from encoder import Encoder
from Layers.dpp import DPP
from Layers.input_layer import InputLayer
from Model.FLIC.flic import FLIC
from Util.data_helper import store, load
from adaptive_learning_rate import adaptive_learning_rate
from Util.data_io import training_sample, get_strings_from_samples
from Util.nn.basic import EmbeddingLayer, apply_dropout
from Util.nn import create_optimization_updates
from Util.user_signals import GracefullExit
import numpy as np
import os
import sys
import theano.tensor as T
import theano
from theano.compile.nanguardmode import NanGuardMode
from timeit import default_timer as timer

class Model:
    def __init__(self, args, sent_emb_dim, flic_dim, load_model=None, epochs_done=0):
        """Initializes the model and constructs the Theano Computation Graph."""
        self.args = args
        self.sig_handler = GracefullExit()
        self.best_val_error = sys.float_info.max

        if self.args.sample_all_sentences:
            print("We sample all items from the generator in each iteration.")
        else:
            print("We sample {} sets from the generator in each iteration.".format(args.num_samples))

        self.flic_dim = flic_dim
        self.sent_emb_dim = sent_emb_dim

        self.dropout_encoder = theano.shared(
            np.float64(args.dropout_encoder).astype(theano.config.floatX)
        )
        self.dropout_generator = theano.shared(
            np.float64(args.dropout_generator).astype(theano.config.floatX)
        )

        # Generator and Encoder Layers
        if args.dropout_generator:
            self.generator = Generator(args, self.dropout_generator, self.sent_emb_dim, flic_dim)
        else:
            self.generator = Generator(args, None, self.sent_emb_dim, flic_dim)
        if args.dropout_encoder:
            self.encoder = Encoder(args, self.dropout_encoder, self.sent_emb_dim)
        else:
            self.encoder = Encoder(args, None, self.sent_emb_dim)

        #---------------------------------------------------------------------------------------------------------------
        # Construct computation graph
        #---------------------------------------------------------------------------------------------------------------
        print("Constructing computation graph.")

        # (Input) Tensors
        sent_embs_t = T.matrix('sent_embs', dtype=theano.config.floatX)
        context_t = T.vector('context', dtype=theano.config.floatX)
        sample_sentences_padded_t = T.tensor3('sample_sent_embeddings', dtype=theano.config.floatX)
        item_counts_t = T.ivector('item_counts')
        y_t = T.scalar('y', dtype=theano.config.floatX)
        samples_t = T.imatrix('samples')        # Sentence embedding
        max_num_sents_t = T.iscalar('max_num_sents')

        transformed_context_t = self.generator.transform_context(context_t, normalize_embeddings=True)

        if not self.args.sample_all_sentences:
            # Construct L to sample from the DPP.

            L_t = self.generator.get_L(sent_embs_t, transformed_context_t)

            self.get_L_t = theano.function(inputs = [sent_embs_t, context_t],
                                                        outputs = [L_t,transformed_context_t],
                                                        #mode='DebugMode',
                                                        #profile=True,
                                                        allow_input_downcast=True,
                                                        #mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True),
                                                        #on_unused_input='warn'
                                                        )


            # The samples will be passed into the Deep Set Layer to calculate the final cost.

            # Encoder cost & updates
            padded_sents_t, sents_count_t = self.generator.get_padded_embeddings_from_samples_t(samples_t, sent_embs_t, max_num_sents_t)
            regularizer_cost_t, exp_set_size_t = self.generator.regularizer_cost(L_t)

            probs_t, costs_encoder_t, \
            probs_mean_t, costs_encoder_mean_t, rand_updates = self.encoder.cost(padded_sents_t, sents_count_t,
                                                                                 transformed_context_t, y_t)

            # Generator cost & updates
            lprob_t, cost_enc_t, cost_generator_t = self.generator.cost(L_t, samples_t, costs_encoder_t, sents_count_t)

        else:
            probs_mean_t, costs_encoder_mean_t = self.encoder.cost_all_sentences(sent_embs_t, transformed_context_t, y_t)
            cost_generator_t = costs_encoder_mean_t

        # Updates of the Generator and Encoder Parameters
        updates_e, self.lr_e, gnorm_e, self.params_opt_e = create_optimization_updates(
            cost = costs_encoder_mean_t,
            params = self.encoder.get_params(),
            method = self.args.learning,
            lr = self.args.learning_rate_encoder
        )[:4]

        updates_g, self.lr_g, gnorm_g, self.params_opt_g = create_optimization_updates(
                               cost = cost_generator_t,
                               params = self.generator.get_params(),
                               method = self.args.learning,
                               lr = self.args.learning_rate_generator
        )[:4]

        if self.args.adaptive_lrs:
            self.adaptive_learning_rate = adaptive_learning_rate(lr_1=self.lr_e,
                                                             lr_2=self.lr_g)
        else:
            self.adaptive_learning_rate = adaptive_learning_rate()

        if not self.args.sample_all_sentences:
            # Compile training graph
            self.train_model_t = theano.function(inputs = [sent_embs_t, samples_t, max_num_sents_t,
                                                         context_t, y_t],
                                               outputs = [probs_mean_t, costs_encoder_mean_t],
                                               updates = updates_e.items() + updates_g.items() + rand_updates,
                                               allow_input_downcast=True,
                                               #mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=False),
                                               #mode='DebugMode',
                                               on_unused_input='warn'
                                               )


            # Compile graph for validation data
            self.validate_t = theano.function(inputs = [sent_embs_t, samples_t, max_num_sents_t,
                                                      context_t, y_t],
                                            outputs = [probs_mean_t, costs_encoder_mean_t],
                                            updates = rand_updates,
                                            allow_input_downcast=True,
                                            on_unused_input='warn'
                                            )

        else:
            # Compile train graph
            self.train_model_t = theano.function(inputs= [sent_embs_t, context_t, y_t],
                                               outputs= [probs_mean_t, costs_encoder_mean_t],
                                               updates= updates_g.items() + updates_e.items(),
                                               allow_input_downcast=True,
                                               # mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True),
                                               on_unused_input='warn'
                                               )
            # Compile graph for validation data
            self.validate_t = theano.function(inputs= [sent_embs_t, context_t, y_t],
                                            outputs= [probs_mean_t, costs_encoder_mean_t],
                                            updates= [],
                                            allow_input_downcast=True,
                                            on_unused_input='warn'
                                            )

        # Load pretrained model
        if load_model:
            self.load(load_model)
        elif self.args.load_model:
            self.load(args.load_model)
        self.epochs_done = epochs_done


    def validation_cost(self, D_valid, D_C_valid, prod_to_sent, flic, num_samples = 10):
        probs_pos = 0
        probs_neg = 0
        probs_pos_count = 0
        probs_neg_count = 0
        tot_cost = 0
        num_errors = 0.0

        self.dropout_encoder.set_value(0.0)
        self.dropout_generator.set_value(0.0)

        for context, x, y in training_sample(D_valid, D_C_valid):

            sent_embs = prod_to_sent[x][0]

            if not self.args.sample_all_sentences:
                L, trans_sent_embs = self.get_L_t(sent_embs, flic[context])

                samples, max_sents_count = self.generator.sample(L, num_samples)
                mean_prob, mean_cost = self.validate_t(sent_embs, samples, max_sents_count, flic[context], y)
            else:
                mean_prob, mean_cost = self.validate_t(sent_embs, flic[context], y)

            if abs(mean_prob - y) > 0.5:
                num_errors += 1

            tot_cost +=  mean_cost
            if y == 1:
                probs_pos += mean_prob
                probs_pos_count += 1
            else:
                probs_neg += mean_prob
                probs_neg_count += 1

        self.dropout_generator.set_value(self.args.dropout_generator)
        self.dropout_encoder.set_value(self.args.dropout_encoder)

        count = probs_pos_count + probs_neg_count

        if probs_pos_count > 0 and probs_neg_count > 0:
            return 1 - num_errors / count, tot_cost / count, probs_pos / probs_pos_count, probs_neg / probs_neg_count
        else:
            return 0, 0, 0, 0

    def testset_evaluation(self, D_test, D_C_test, prod_to_sent, flic, num_samples = 20, random_sampling=False):

        self.dropout_encoder.set_value(0.0)
        self.dropout_generator.set_value(0.0)

        probs_pos = 0
        probs_neg = 0
        probs_pos_count = 0
        probs_neg_count = 0
        tot_cost = 0
        sampled_sentences = []
        costs = []
        num_errors = 0.0

        for context, x, y in training_sample(D_test, D_C_test):

            sent_embs = prod_to_sent[x][0]

            if not self.args.sample_all_sentences:
                L, trans_sent_embs = self.get_L_t(sent_embs, flic[context])

                if random_sampling:
                    samples, max_sents_count = self.generator.sample(L, num_samples, random_sampling=True)
                else:
                    samples, max_sents_count = self.generator.sample(L, num_samples)
                mean_prob, mean_cost = self.validate_t(sent_embs, samples, max_sents_count, flic[context], y)

                sampled_sentences.append((context, x, get_strings_from_samples(prod_to_sent[x][1], samples)))
                costs.append(mean_cost)
            else:
                mean_prob, mean_cost = self.validate_t(sent_embs, flic[context], y)

            tot_cost +=  mean_cost
            if y == 1:
                probs_pos += mean_prob
                probs_pos_count += 1
            else:
                probs_neg += mean_prob
                probs_neg_count += 1

            if abs(mean_prob - y) > 0.5:
                num_errors += 1


        self.dropout_generator.set_value(self.args.dropout_generator)
        self.dropout_encoder.set_value(self.args.dropout_encoder)

        return sampled_sentences, 1 - num_errors / (probs_pos_count + probs_neg_count),  \
               (tot_cost / (probs_pos_count + probs_neg_count), probs_pos / probs_pos_count, probs_neg / probs_neg_count)

    def _print_scores(self, valid_scores, train_scores):

        if train_scores:
            train_cost, train_pos_score, train_neg_score = train_scores
            print("Average cost on the training set: {}".format(train_cost))
            print("Average probabilities of true links: {}".format(train_pos_score))
            print("Average probabilities of false links: {}".format(train_neg_score))

        if valid_scores:
            valid_cost, valid_pos_score, valid_neg_score = valid_scores
            print("Average cost on the validation set: {}".format(valid_cost))
            print("Average probabilities of true links: {}".format(valid_pos_score))
            print("Average probabilities of false links: {}".format(valid_neg_score))

    def train(self, D, D_C, D_valid, D_C_valid, flic, prod_to_sent):

        #---------------------------------------------------------------------------------------------------------------
        # Looping through the data (i.e. the 'Training-For-Loop').
        #---------------------------------------------------------------------------------------------------------------

        # For time measurements
        t_sampling = 0.0
        t_generator = 0.0
        t_encoder = 0.0
        t_gen_training = 0.0
        t_enc_training = 0.0


        print("Start training.")
        print("Num links in training set = {}".format(len(D) + len(D_C)))
        for epoch in range(self.args.max_epochs):
            epoch += self.epochs_done
            iteration = 0
            cost_epoch = 0
            cost_tmp = 0
            probs_pos = 0; probs_neg = 0; probs_pos_count = 0; probs_neg_count = 0
            probs_pos_tmp = 0; probs_neg_tmp = 0; probs_pos_count_tmp = 0; probs_neg_count_tmp = 0

            for context, x, y in training_sample(D, D_C):

                #-------------------------------------------------------------------------------------------------------
                # Train model
                #-------------------------------------------------------------------------------------------------------

                t_0 = timer()

                sent_embs = prod_to_sent[x][0]

                if not self.args.sample_all_sentences:
                    L, trans_context = self.get_L_t(sent_embs, flic[context])
                    t_1 = timer()
                    samples, max_set_size = self.generator.sample(L, self.args.num_samples)

                    p = self.generator.scale_sent_embs.get_value()

                    t_2 = timer()
                    mean_prob, mean_cost = self.train_model_t(sent_embs, samples,
                                                            max_set_size, flic[context], y)
                else:
                    t_1 = timer()
                    t_2 = timer()
                    mean_prob, mean_cost = self.train_model_t(sent_embs, flic[context], y)

                t_3 = timer()

                t_sampling += (t_2 - t_1)
                t_generator += (t_1 - t_0)
                t_encoder += (t_3 - t_2)

                #-------------------------------------------------------------------------------------------------------
                # Output of the training and validation scores (only sometimes)
                #-------------------------------------------------------------------------------------------------------

                cost_epoch += mean_cost
                cost_tmp += mean_cost
                if y == 1:
                    probs_pos += mean_prob
                    probs_pos_tmp += mean_prob
                    probs_pos_count += 1
                    probs_pos_count_tmp += 1
                else:
                    probs_neg += mean_prob
                    probs_neg_tmp += mean_prob
                    probs_neg_count += 1
                    probs_neg_count_tmp += 1

                iteration += 1

                # All 1000 datapoints
                if iteration % 1000 == 0:
                    print("Cost on the training set after {} iterations: {}".format(iteration, cost_tmp /
                                                                            (probs_pos_count_tmp + probs_neg_count_tmp)))
                """
                # All xxx datapoints
                if iteration % self.args.num_iters_between_validation == 0:
                    print("\nResults after {} iterations:".format(self.args.num_iters_between_validation))

                    # Calculate the cost on the validation set
                    if D_valid is not None and D_C_valid is not None:
                        accuracy, valid_cost, valid_pos_score, valid_neg_score = self.validation_cost(D_valid, D_C_valid,
                                                                                            prod_to_sent, flic)
                        if valid_cost < self.best_val_error and self.args.save_model:
                            self.best_val_error = min(self.best_val_error, valid_cost)
                            self.store(self.args.save_model + '/best_valerr_' + str("{0:3f}.pkl".format(valid_cost)))

                        # Decreasing the learning rate and reloading the best model if the validation cost
                        # is not decreasing anymore.
                        if self.args.adaptive_lrs:
                            lrs_adapted = self.adaptive_learning_rate.adapt_learning_rate(valid_cost, self.lr_e, self.lr_g)
                        if self.args.save_model and self.args.adaptive_lrs and lrs_adapted:
                            lr_e_value = self.lr_e.get_value()
                            lr_g_value = self.lr_g.get_value()
                            self.load(self.args.save_model + '/best_valerr_' + str("{0:3f}.pkl".format(self.best_val_error)))
                            self.lr_e.set_value(lr_e_value)
                            self.lr_g.set_value(lr_g_value)
                            print("Best model loaded.")


                        # If the learning rates are very small, we stop our training.
                        if self.lr_g.get_value() < 1e-5 and self.lr_e.get_value() < 1e-5:
                            return (self.best_val_error, "END")

                        self._print_scores((valid_cost, valid_pos_score, valid_neg_score),
                                           (cost_tmp / (probs_pos_count_tmp + probs_neg_count_tmp),
                                            probs_pos_tmp / probs_pos_count_tmp,
                                            probs_neg_tmp / probs_neg_count_tmp))
                    else:
                        self._print_scores(None, (cost_tmp / (probs_pos_count_tmp + probs_neg_count_tmp),
                                                  probs_pos_tmp / probs_pos_count_tmp,
                                                  probs_neg_tmp / probs_neg_count_tmp))
                    print("")
                    probs_neg_tmp = 0
                    probs_pos_tmp = 0
                    probs_pos_count_tmp = 0
                    probs_neg_count_tmp = 0
                    cost_tmp = 0
                """

                if iteration % 10000 == 0 and self.args.save_model:
                    self.store(self.args.save_model, epoch, iteration)
                    print("Model saved. Epoch = {}, Iteration = {}".format(epoch, iteration))

                if self.sig_handler.exit == True:
                    if self.best_val_error == sys.float_info.max and self.args.save_model:
                        self.best_val_error = -1
                        self.store(self.args.save_model + '/best_valerr_' + str("{0:3f}.pkl".format(self.best_val_error)))
                    return (self.best_val_error, "SIGNAL")


            # ----------------------------------------------------------------------------------------------------------
            # Output per epoch
            # ----------------------------------------------------------------------------------------------------------

            if self.args.save_model:
                self.store(self.args.save_model, epoch, 0)
                print("\nEpoch {} done. Model saved! ".format(epoch))
            else:
                print("\nEpoch {} done.".format(epoch))

            # Timing output
            if self.args.measure_timing:
                print("Time Sampling = {}".format(t_sampling / iteration))
                print("Time Generator = {}".format(t_generator / iteration))
                print("Time Encoder = {}".format(t_encoder / iteration))


            print("Calculating the average cost in this epoch.")
            if D_valid is not None and D_C_valid is not None:
                accuracy, valid_cost, valid_pos_score, valid_neg_score = self.validation_cost(D_valid, D_C_valid, prod_to_sent, flic)
                self._print_scores((valid_cost, valid_pos_score, valid_neg_score),
                                   (cost_epoch / (probs_pos_count + probs_neg_count),
                                   probs_pos/probs_pos_count, probs_neg/probs_neg_count))
                if valid_cost < self.best_val_error and self.args.save_model:
                    self.best_val_error = min(self.best_val_error, valid_cost)
                    self.store(self.args.save_model + '/best_valerr_' + str("{0:3f}.pkl".format(valid_cost)))


                # Decreasing the learning rate and reloading the best model if the validation cost
                # is not decreasing anymore.
                if self.args.adaptive_lrs:
                    lrs_adapted = self.adaptive_learning_rate.adapt_learning_rate(valid_cost, self.lr_e, self.lr_g)
                if self.args.save_model and self.args.adaptive_lrs and lrs_adapted:
                    lr_e_value = self.lr_e.get_value()
                    lr_g_value = self.lr_g.get_value()
                    self.load(self.args.save_model + '/best_valerr_' + str("{0:3f}.pkl".format(self.best_val_error)))
                    self.lr_e.set_value(lr_e_value)
                    self.lr_g.set_value(lr_g_value)
                    print("Best model loaded.")

                # If the learning rates are very small, we stop our training.
                if self.lr_g.get_value() < 1e-5 and self.lr_e.get_value() < 1e-5:
                    return (self.best_val_error, "END")
            else:
                self._print_scores(None, (cost_epoch / (probs_pos_count + probs_neg_count),
                                                probs_pos/probs_pos_count,
                                                probs_neg/probs_neg_count))
            print("")

            # Resets all temporary variables.
            cost_epoch = 0
            if epoch+1 >= self.args.max_epochs:
                return (self.best_val_error, "END")
            if (epoch+1) % 10 == 0:
                self.epochs_done += 10
                self.store(self.args.save_model + '/after_10_epochs_' + str("{0:3f}.pkl".format(valid_cost)))
                return (valid_cost, "10_epochs")
            probs_neg = 0
            probs_pos = 0
            probs_pos_count = 0
            probs_neg_count = 0
        return (self.best_val_error, "END")


    def store(self, path, epoch = None, iter = None):
        """ Stores the model with the filename 'path'.

        :type path: String
        :param path: The filename under which the model should be saved. If 'epoch' is not None and 'iter' is not None,
        then the model will be saved in the folder given by 'path' and the filename will be constructed from 'epoch' and 'iter'.

        """
        if epoch is not None and iter is not None:
            path = path + '/wedim_' + str(self.sent_emb_dim) + '_fdim_' + str(self.flic_dim) + \
                   '_epoch_' + str(epoch) + '_iter_' + str(iter) + '.pkl'
        obj = [(self.sent_emb_dim, self.flic_dim, self.args.gen_num_hidden_layers, self.args.gen_num_hidden_units,
                self.args.enc_num_hidden_layers, self.args.enc_num_hidden_units)]
        obj.append(self.generator.get_params())
        obj.append(self.encoder.get_params())
        obj.append(self.params_opt_g)
        obj.append(self.params_opt_e)
        obj.append(self.adaptive_learning_rate.val_error)
        obj.append(self.best_val_error)

        store(obj, path)

    def load(self, file):
        print("Loading trained model from file.")
        params = load(file)

        # Test if the parameters have the right format
        sent_emb_dim, flic_dim, gen_num_layers, gen_num_units, enc_num_layers, enc_num_units = params[0]
        if sent_emb_dim != self.sent_emb_dim or flic_dim != flic_dim or gen_num_layers != self.args.gen_num_hidden_layers or \
            gen_num_units != self.args.gen_num_hidden_units or enc_num_layers != self.args.enc_num_hidden_layers or \
            enc_num_units != self.args.enc_num_hidden_units:
            raise(ValueError, "The dimension of the loaded model are not consistent "
                              "with the dimensions of the command line arguments.")

        params_gen = [theano.shared(np.float64(x.get_value()).astype(theano.config.floatX)) for x in params[1]]
        params_enc = [theano.shared(np.float64(x.get_value()).astype(theano.config.floatX)) for x in params[2]]
        self.generator.set_params(params_gen)
        self.encoder.set_params(params_enc)

        params_opt_g = params[3]
        params_opt_e = params[4]
        for model_param, stored_param in zip(self.params_opt_g,params_opt_g):
            model_param.set_value(stored_param.get_value())

        for model_param, stored_param in zip(self.params_opt_e,params_opt_e):
            model_param.set_value(stored_param.get_value())

        self.adaptive_learning_rate.val_error = params[5]
        self.best_val_error = params[6]