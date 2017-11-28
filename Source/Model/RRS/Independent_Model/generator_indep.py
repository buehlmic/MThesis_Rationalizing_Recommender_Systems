from Util.data_helper import load
from Util.nn.basic import LSTM, apply_dropout, Layer
from Util.nn.extended_layers import ExtLSTM
from Util.nn.initialization import tanh, ReLU, sigmoid
import theano.tensor as T
import theano
import numpy as np
from Layers.dpp import DPP
from theano.tensor.nlinalg import Det


class Generator:
    def __init__(self, args, dropout, dim_sent_embedding, dim_flic_embedding):
        self.args = args
        self.dpp = DPP()
        self.dim_sent_emb = dim_sent_embedding
        self.dim_flic = dim_flic_embedding
        self.dropout = dropout

        if not self.args.sample_all_sentences:
            self.scale_sent_embs = theano.shared(
                #(1.0/100)*
                (np.float64(args.num_target_sentences)).astype(theano.config.floatX), 'scale_sent_embs'
            )

        num_layers = args.gen_num_hidden_layers
        num_units = args.gen_num_hidden_units
        activation = tanh
        self.ff_layers = []
        for i in range(num_layers):
            if i == 0 and i == num_layers - 1:
                self.ff_layers.append(Layer(self.dim_flic, self.dim_sent_emb, activation=activation))
            elif i == 0 and i != num_layers - 1:
                self.ff_layers.append(Layer(self.dim_flic, num_units, activation=activation))
            elif i != 0 and i == num_layers -1 :
                self.ff_layers.append(Layer(num_units, self.dim_sent_emb, activation=activation))
            else:
                self.ff_layers.append(Layer(num_units, num_units, activation=activation))


    def transform_context(self, context, normalize_embeddings=True):
        for layer in self.ff_layers:
            context = layer.forward(context)
            if self.dropout:
                context = apply_dropout(context, self.dropout)

        #sent_embs_normalized = sent_embs / T.sqrt((sent_embs**2).sum(axis=1)).reshape((sent_embs.shape[0],1))
        if normalize_embeddings:
            context = context / context.norm(2)
        return context

    def get_L(self, sent_embs, context):
        L = self.dpp.build_L_independent(sent_embs, context, activation=tanh)
        #L = (100 * self.scale_sent_embs) * L
        L = T.abs_(self.scale_sent_embs) * L
        return L

    def sample(self, L, num_samples, random_sampling=False):

        # For numerical reasons we need float64 values.
        L = L.astype(np.float64)
        e_vectors, e_values = self.dpp.eigendecomposition(L)
        samples = []
        max_iters = 100
        iters = 0
        while len(samples) < num_samples:
            samples.extend(self.dpp.get_samples(e_vectors, e_values, num_samples-len(samples)))
            # Remove empty sets
            samples = filter(lambda s: 1 in s, samples)
            iters += 1
            if iters >= max_iters and len(samples) > 0:
                break
        max_set_size = max(np.count_nonzero(samples, axis=1))
        return samples, max_set_size

    def expected_set_size(self, L):
        e_vectors, e_values = self.dpp.eigendecomposition(L)
        expected_set_size = np.dot(e_values, 1.0/(e_values+1.0))

        # We need to scale 'expected_set_size' because we remove empty sets while sampling.
        num_sents = np.shape(L)[0]
        L_I = L + np.eye(num_sents)
        det_L_I = np.linalg.det(L_I)
        factor = det_L_I / (det_L_I - 1.0)
        expected_set_size *= factor

        return expected_set_size

    def _get_embeddings_from_samples(self, samples, sent_embs):
        sents_embedded = [sent_embs[np.nonzero(sample)[0], :] for sample in samples]
        return sents_embedded

    def get_padded_embeddings_from_samples(self, samples, sent_embs):
        """Get embedding from a set of samples and pad each sample (a set of sentences)
        to the maximal set size (maximal number of sentences).

        :type samples: np.array
        :param samples: Each row of the matrix denotes a sample. The sentence i is in the j-th set iff the entry (j,i)
        of samples_t is 1.
        :type sent_embs: np.array
        :param sent_embs: The entence embedding matrix. Each row of the matrix contains the
        embedding vector of one sentence.

        :type max_length: int
        :param max_length: The maximum number of sentences in a sample.

        :type return value: np.ndarray (3 dimensional)
        :param return value: The entry (i,j,k) of the return value represents the k-th entry of the j-th sentence
        embedding vector of the i-th sample. Additionally, the last rows (2nd dimension) are padded with zeros such that
        each sample has the same number of sentences.
        """

        sents_embedded = self._get_embeddings_from_samples(samples, sent_embs)
        item_lengths = [sample.shape[0] for sample in sents_embedded]
        max_length = max(item_lengths)
        padded_embs =  [np.vstack((sample, np.zeros((max_length-l, sent_embs.shape[1])))) for sample, l in zip(sents_embedded, item_lengths)]
        return padded_embs, item_lengths

    def _get_padded_embeddings_from_sample_t(self, sample_t, sent_embs_t, max_length):
        """Get embedding from a set of samples and pad it """
        idxs = T.eq(sample_t, 1).nonzero()[0]
        sent_embs = sent_embs_t[idxs,:]
        size = sent_embs.shape
        padded = T.concatenate([sent_embs, T.zeros((max_length-size[0], size[1]))])
        return padded, size[0]

    def get_padded_embeddings_from_samples_t(self, samples_t, sent_embs_t, max_length):
        """Get embedding from a set of samples and pad each sample (a set of sentences)
        to the maximal set size (maximal number of sentences).

        :type samples_t: T.imatrix
        :param samples_t: Each row of the matrix denotes a sample. The sentence i is in the j-th set iff the entry (j,i)
        of samples_t is 1.
        :type sent_embs_t: T.matrix
        :param sent_embs_t: The entence embedding matrix. Each row of the matrix contains the
        embedding vector of one sentence.

        :type max_length: T.iscalar
        :param max_length: The maximum number of sentences in a sample.

        :type return value: T.tensor3
        :param return value: The entry (i,j,k) of the return value represents the k-th entry of the j-th sentence
        embedding vector of the i-th sample. Additionally, the last rows (2nd dimension) are padded with zeros such that
        each sample has the same number of sentences.
        """

        (padded, sent_count), _ = theano.scan(self._get_padded_embeddings_from_sample_t,
                                sequences = samples_t,
                                outputs_info = [None, None],
                                non_sequences = [sent_embs_t, max_length])
        return padded, sent_count

    def regularizer_cost(self, L_t):
        """ Penalize if the expected set size is far from the target value

        :type L_t: T.matrix
        :param L_t: The Ensemble matrix

        :type return value: T.fscalar
        :param return value: The Cost
        . """

        num_sents = T.shape(L_t)[0]

        # Calculate expectation value
        #eigenvalues, _ = T.nlinalg.eigh(L_t)
        #expected_set_size = T.dot(eigenvalues, 1 / (eigenvalues + 1))
        K = T.eye(num_sents) - T.nlinalg.matrix_inverse(L_t + T.eye(num_sents))
        expected_set_size = T.nlinalg.trace(K)

        # We need to scale 'expected_set_size' because we remove empty sets while sampling.
        num_sents = T.shape(L_t)[0]
        L_I = L_t + T.eye(num_sents)
        det_L_I = Det()(L_I)
        factor = det_L_I / (det_L_I - 1.0)
        expected_set_size *= factor

        return 2*(expected_set_size-self.args.num_target_sentences)*self.args.regularizer, expected_set_size

    def cost(self, L, samples, cost_enc, sents_count):
        """ Calculates the cost for the generator. This is used for the gradient calculation.

        :type L_t: T.matrix
        :param L_t: The Ensemble matrix
        :type samples: T.matrix, shape = (num_samples, num_items_in_set)
        :param samples: Each row contains one sample
        :type cost_enc: T.vector
        :param cost_enc: The cost of the encoder (1 scalar value for every sampled set)

        :type: T.scalar
        :return: The cost

        """

        #log_prob = self.dpp.lprob_z(L, sample.dimshuffle('x',0))[0]
        reg_cost, _ = self.regularizer_cost(L)
        log_prob = self.dpp.lprob_z(L, samples)
        cost_enc_disc = theano.gradient.disconnected_grad(cost_enc)
        sents_count_disc = theano.gradient.disconnected_grad(sents_count)
        reg_cost_disc = theano.gradient.disconnected_grad(reg_cost)
        gen_cost = T.mean(log_prob*cost_enc_disc + cost_enc + reg_cost_disc*log_prob*sents_count_disc, axis=0)
        return log_prob, cost_enc, gen_cost

    def l2_cost(self):
        l2 = 0
        for p in self.params:
            l2 += T.sum(p**2)
        return l2*self.args.l2_gen_reg

    def get_params(self):
        params = []
        for layer in self.ff_layers:
            params.extend(layer.params)
        if self.args.sample_all_sentences:
            return params
        else:
            return params + [self.scale_sent_embs]

    def set_params(self, param_list):
        start = 0
        for layer in self.ff_layers:
            end = start + len(layer.params)
            layer.params = param_list[start:end]
            start = end
        if not self.args.sample_all_sentences:
            self.scale_sent_embs = param_list[-1]





