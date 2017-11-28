from Util.nn.basic import Layer
import theano.tensor as T
import theano
import numpy as np

class InputLayer(Layer):

    def __init__(self, embedding_layer):
        self.embedding_layer = embedding_layer

    def forward(self, x, list_lengths):
        """


        :param x:
        :param list_lengths:
        :return:
        """
        # Get word embeddings
        word_embs = self.embedding_layer.forward(x)

        # Reshapes word_embs into a 3d Tensor of the following shape:
        # (num_sentences, maximum number of words in a sentence, word embedding dimension).
        # The word_embs gets padded in the second dimension such that the resulting object is a valid 3d Tensor.
        max_list_length = T.max(list_lengths)
        def pad_word_embs(length, start, embs, max_length):
            sent_embedded = embs[start:(start + length), :]
            z = T.zeros((max_length - length, embs.shape[-1]))
            vec = T.concatenate([sent_embedded, z], axis=0)
            mask = T.concatenate((T.ones((length,),dtype=theano.config.floatX),
                                 T.zeros((max_list_length - length,),dtype=theano.config.floatX))
                                )
            return vec, mask, start + length

        word_embs, updates = theano.scan(pad_word_embs,
                                         outputs_info=[None, None, np.int32(0)],
                                         sequences=list_lengths,
                                         non_sequences=[word_embs, max_list_length])



        return word_embs[0].dimshuffle(1,0,2), word_embs[1].T


