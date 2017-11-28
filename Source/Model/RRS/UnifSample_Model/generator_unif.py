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

        """
        if not self.args.sample_all_sentences:
            self.scale_sent_embs = theano.shared(
                #(1.0/100)*
                (np.float64(args.num_target_sentences)).astype(theano.config.floatX), 'scale_sent_embs'
            )
        """

        num_layers = args.gen_num_hidden_layers
        num_units = args.gen_num_hidden_units
        activation = tanh
        self.ff_layers = []
        for i in range(num_layers):
            if i == 0 and i == num_layers - 1:
                self.ff_layers.append(Layer(self.dim_flic, self.dim_sent_emb, activation=activation))
            elif i == 0 and i != num_layers - 1:
                self.ff_layers.append(Layer(self.dim_flic, num_units, activation=activation))
            elif i != 0 and i == num_layers - 1:
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

    def sample(self, num_sents_in_reviews, num_samples):
        p = float(self.args.num_target_sentences)/num_sents_in_reviews
        samples = []
        while len(samples) < num_samples:
            samples = np.random.choice(2, (num_samples, num_sents_in_reviews), p=np.array([1-p,p]))
            samples = filter(lambda s: 1 in s, samples)

        max_set_size = max(np.count_nonzero(samples, axis=1))
        return samples, max_set_size


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


    def cost(self, num_sents_in_reviews, samples, cost_enc):
        # Log probability of the sampled sets
        probs = float(self.args.num_target_sentences)/num_sents_in_reviews
        def calc_prob(sample):
            l_probs = sample*T.log(probs) + (1 - sample) * T.log(1 - probs)

            return T.sum(l_probs, axis=0)

        lprobs, update = theano.scan(fn=calc_prob, sequences=samples)
        cost_enc_disc = theano.gradient.disconnected_grad(cost_enc)
        gen_cost = T.mean(lprobs*cost_enc_disc + cost_enc)
        return lprobs, cost_enc, gen_cost

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
            return params

    def set_params(self, param_list):
        start = 0
        for layer in self.ff_layers:
            end = start + len(layer.params)
            layer.params = param_list[start:end]
            start = end





