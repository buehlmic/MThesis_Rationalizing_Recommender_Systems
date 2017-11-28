import theano.tensor as T
import theano
import theano.typed_list
from Layers.deep_sets import DeepSetLayer
from Util.nn.basic import Layer, Dropout, apply_dropout
from Util.nn.initialization import sigmoid, tanh, ReLU


class Encoder:
    def __init__(self, args, dropout, num_in):
        self.args = args
        self.num_in = num_in

        self.deep_sets = DeepSetLayer(num_in = num_in,
                                  num_hidden = args.enc_num_hidden_units,
                                  context_size=num_in,
                                  num_perm_equiv_layers = args.enc_num_hidden_layers,
                                  activation=sigmoid
                                  )

    def _encode_sentences(self, sentences, context):
        """ Apply the Deep Set Layer

        :type sentences: T.matrix, shape = (num_sents_in_set, sent_emb_dim)
        :param sentences: Each row of the matrix contains the embedding vector of one sentence.
        :type context: T.vector
        :param context: The context vector

        :type return value: T.scalar
        :param return value: The probability that the link represented by sentences and context is in D.

        """
        prob = self.deep_sets.forward(sentences, context)
        return prob

    def _cost_one_sample(self, sentences, num_sents, context, y):
        """ Calculates the probability (of the link in D) and the cost of one sample

        :type sentences: T.matrix, shape = (>=num_sents, sent_emb_dim)
        :param sentences: Each row of the matrix contains the embedding vector of one sentence.
        :type num_sents: T.iscalar
        :param num_sents: The number of sentences in the set
        :type regularizer_cost: T.scalar
        :param regularizer_cost: The cost penalty from the difference of the expected set size and 'num_target_sentences'.
        :type context: T.vector
        :param context: The context vector
        :type y: Boolean
        :param y: 1 (link in D) or 0 (link not in D)

        :type return value: pair (T.scalar, T.scalar)
        :return: A pair of (probability that the link is in D, cost)
        """

        prob = self._encode_sentences(sentences[:num_sents,:], context)
        return prob,-(y*T.log(prob) + (1-y)*T.log(1-prob))

    def cost(self, sampled_sents, sents_count, context, y = None):
        """Calculates the cost of the encoder

        :type sampled_sents: T.tensor, shape = (num_samples, max(sents_count), sent_emb_dim)
        :param sampled_sents: The tensor, where each slice contains one sampled set. Each such set has the sentence
        embeddings in the rows.
        :type sents_count: T.vector, shape = (num_samples,)
        :param sents_count: The sizes of the sampled sets
        :type context: T.vector
        :param context: The context vector
        :type y: Boolean or None
        :param y: 1 (link in D) or 0 (link not in D) or None (if predicting)
        :type regularizer_cost: float
        :param regularizer_cost: The cost penalty from the difference of the expected set size and 'num_target_sentences'.

        :type return value: A tuple (T.vector, T.vector, T.scalar, T.scalar, ?)
        :type return value: A tuple of (vector of probabilities, vector of costs, mean_probabilites, mean_costs,
        random_update_object). Each element in the vector represents one set.

        """

        if y:
            y0 = y
        else:
            y0 = 0

        # A scan over all sampled sets.
        (probs, costs), updates = theano.scan(self._cost_one_sample,
                           sequences=[sampled_sents, sents_count],
                           non_sequences = [context, y0],
                           outputs_info = None)

        if y:
            return probs, costs, T.mean(probs), T.mean(costs), updates
        else:
            return T.mean(probs), probs, updates

    """def cost_all_sentences(self, sampled_sents, context, y = None):
        prob = self._encode_sentences(sampled_sents, context)
        if y:
            cost = -(y*T.log(prob) + (1-y)*T.log(1-prob))
            return prob, cost
        else:
            return prob
    """

    def get_params(self):
        return self.deep_sets.params

    def set_params(self, param_list):
        self.deep_sets.params = param_list


