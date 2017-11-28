import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nlinalg import Det
from theano.ifelse import ifelse
from theano.printing import Print

class DPP:
    """A class modeling a Determinantal Point Process.

    Because this class uses a mixture of numpy methods (for sampling) and theano functions (for everything else), it is
    non-trivial to use this class. In the following we show an example for how to use it:

    # Build the ensemble matrix L
    import theano.tensor as T
    dpp = DPP()
    L = dpp.build_L(Sentences, Context)     # 'Sentences' and 'Context' are theano variables.
    f = theano.function([Sentences, Context], L)
    L_val = f(sent, cont)                   # 'sent' and 'cont' are the inputs to the theano functions.
                                            # 'L_val' is now a numpy array.

    # Sampling
    e_vectors, e_values = dpp.eigendecomposition(L_val)
    sampled_sets = []
    for i in range(10):                     # We sample 10 sets.
        sample = dpp.sample(e_vectors, e_values)
        sampled_sets.append(sample)


    # Calculate the log probabilities of the sampled sets
    lprobs = lprob_z(L, sampled_sets)
    """

    def __init__(self):
        pass

    def build_L(self, sentences, context, activation = T.nnet.sigmoid, biased_diagonal = False):
        """Constructs the matrix L (L-Ensemble from the paper 'Determinantal point processes for
        machine learning' (Kulesza, Taskar, 2013). L_mn = p_m'*p_n*(2*f(p_m'*context)-1)*(2*f(p_n'*context)-1),
        where p_m and p_n are the feature vector of items m and n. f is the activation function and context the
        context vector._

        :type sentences: T.matrix, shape = (num_items_in_set, dim_per_item)
        :param sentences: The input matrix for the DPP. Each row of the DPP encodes the feature vector for one item
        of a ground set S.

        :type context: T.vector, shape = (dim_per_item)
        :param context: The feature vector encoding the context

        :type activation: Theano tensor function
        :parameter activation: An activation function (a sigmoid is recommended to get reasonable values).

        :type return value: T.matrix, shape = (num_items_in_set, num_items_in_set)
        :return return value: The Ensemble Matrix L.

        """

        f_sentT_cont = activation(T.dot(sentences, context))
        f2_1 = f_sentT_cont

        B = sentences * f2_1.dimshuffle((0, 'x'))
        B_BT = T.dot(B, B.T)
        if biased_diagonal:
            return B_BT + T.identity_like(B_BT)*0.1
        else:
            return B_BT

    def build_L_independent(self, sentences, context, activation = T.nnet.sigmoid):
        """Constructs the L-Ensemble matrix where all but the diagonal is 0. This models independence between the
        items in the sets."""
        f_sentT_cont = activation(T.dot(sentences, context))
        diag = f_sentT_cont**2
        L = T.diag(diag)
        return L

    def build_L_const_context(self, sentences, activation = T.nnet.sigmoid, biased_diagonal=False):
        B_BT = T.dot(sentences, sentences.T)
        if biased_diagonal:
            return B_BT + T.identity_like(B_BT)*0.1
        else:
            return B_BT

    def _orthonormal_basis(self, V, e_i):
        """Calculates an orthonormal basis for the subspace spanned by the columns of 'V' orthogonal to a vector of the
        standard basis.

        :type L: numpy.array
        :param L: An array of linearly independent columns. The number of rows of L must be at least the number of columns.

        :type e_i: Int
        :param e_i: An index (counting starts from 0) denoting the $e_i$-th unit basis vector (i.e. [0,0,...,1,0,..,0],
        where exactly the $e_i$-th entry of the vector is '1' and '0' otherwise.)

        :type return value: numpy.array
        :param return value: An array whose columns are an orthonormal basis for the subspace of 'V' orthogonal to the
        unit basis vector given by 'e_i'.

        """

        # Idea: We first construct the subspace of V orthogonal to the i-th unit basis vector and then use the standard
        # np.linalg.qr decomposition to find the orthonormal basis for this space.

        num_rows, num_cols = V.shape
        if num_cols <= 1:
            return np.empty((0,0))

        # Switch the first column of V with the one containing the max value w.r.t to the i-th row.
        max_ind = np.argmax(abs(V[e_i, :]))
        vec_e_i = np.copy(V[:, max_ind])
        V[:, max_ind] = V[:, 0]
        V[:, 0] = vec_e_i

        # We do one step of Gaussian elimination to find a set of vectors belonging to the subspace of V orthogonal to e_i
        # (i.e. we set the first column (and hence also the i-th row) of V to 0).

        row_e_i = np.copy(V[e_i, :])
        for l in xrange(num_rows):
            a_l0 = V[l, 0] / row_e_i[0]
            V[l, :] -= row_e_i * a_l0

        # QR decomposition
        Q,R = np.linalg.qr(V[:,1:])

        return Q

    def eigendecomposition(self, L):
        """Calculates the eigendecomposition of the ensemble matrix L.

        :type L: numpy.array, shape = (num_items_in_set, num_items_in_set)
        :param L: The ensemble matrix L of the DPP. L is expected to be symmetric and real.

        :type return value: (numpy.array, numpy.array) of shapes (num_items_in_set, num_items_in_set) and
        (num_items_in_set,) respectively.
        :return return value: The first entry of the tuple is a matrix containing the eigenvector of L in its columns.
        The second entry of the tuple is a vector containing the eigenvalues of L (in the same order as the
        associated eigenvectors).

        """

        e_values, e_vectors = np.linalg.eigh(L)
        return (e_vectors, e_values)

    def get_samples(self, e_vectors, e_values, num_samples):
        """Samples from the DPP using the eidgendecomposition from the ensemble matrix L. It does this using the
        algorithm from Hough et al. ('Determinantal processes and independence', 2006).

        :type e_vectors: numpy.array, shape = (num_items_in_set, num_items_in_set)
        :param e_vectors: A matrix containing the eigenvectors of the matrix L in the columns.

        :type e_values: numpy.array, shape = (num_items_in_set,)
        :param e_values: A vector containg the eigenvalues of the matrix L. The i-th entry of the vector is the
        eigenvalue belonging to the eigenvector stored at e_vectors[:,i].

        :type return value: numpy.array, shape = (num_samples, num_items_in_set)
        :return return value: Each row of the array is a boolean vector representing a sampled set. The i-th entry of
        such a vector is one iff the i-th item is in the sampled set.

        """
        N = len(e_values)
        if not (N,N) == e_vectors.shape:
            raise ValueError("The shapes of the eigenvector matrix and the eigenvalue vector don't agree.")

        samples = []
        while len(samples) < num_samples:
            # Which subset of the eigenvector should be used for sampling?
            r = np.random.rand(N)
            probs = np.divide(e_values, e_values+1)
            b = np.random.rand(N) <= probs
            V = e_vectors[:,b]

            # Sampling using the sampled eigenvector subset given by V
            Y = np.zeros((N,))
            while V.shape[1] > 0:
                probs = np.sum(V**2, axis=1)/float(V.shape[1])
                assert(abs(probs.sum()-1) < 1e-10)
                multi_dist = np.random.multinomial(1, probs)
                e_i = np.nonzero(multi_dist)[0][0]

                Y[e_i] += 1
                V = self._orthonormal_basis(V,e_i)
            samples.append(Y)
        return np.array(samples, dtype=np.int32)

    def prob_z(self, L, samples):
        """Calculate the probability of sampled sets given the ensemble matrix L.

        :type L: T.matrix, shape = (num_items_in_set, num_items_in_set)
        :param L: The ensemble matrix L

        :type samples: T.matrix, shape = (num_samples, num_items_in_set)
        :param z: Each row of the matrix is a boolean vector representing a sampled set. The i-th entry of
        such a row-vector is one iff the i-th item is in the sampled set.

        :type return value: T.vector
        :return return value: Each element of the vector represents the probability of the i-th sampled set w.r.t.
        the ensemble matrix L.

        """

        size = T.shape(samples)
        num_samples = size[0]
        num_sets = size[1]
        L_I = L + T.eye(num_sets)
        det_L_I = Det()(L_I)
        factor = det_L_I/(det_L_I-1.0)      # 'factor' acounts for the fact that we filter empty sets

        # Probability of one sampled set
        def calc_prob(sample):
            num_items_in_set = T.neq(sample, 0).sum()
            L_S = L[:, (sample > 0).nonzero()][(sample > 0).nonzero(), :]
            L_S = L_S.reshape((num_items_in_set, num_items_in_set))

            # The next line needs to be uncommented if there is a possibility that one or more sample is the empty set.
            #prob = ifelse(T.eq(num_items_in_set, 0), (1.0), Det()(L_S).astype('float32'))
            prob = Det()(L_S).astype('float64')
            return prob * factor / det_L_I

        probs, update = theano.scan(fn=calc_prob, sequences=samples)
        return probs

    def lprob_z(self, L, samples):
        """Calculate the log-probability of sampled sets given the ensemble matrix L.

        :type L: T.matrix, shape = (num_items_in_set, num_items_in_set)
        :param L: The ensemble matrix L

        :type samples: T.matrix, shape = (num_samples, num_items_in_set)
        :param z: Each row of the matrix is a boolean vector representing a sampled set. The i-th entry of
        such a row-vector is one iff the i-th item is in the sampled set.

        :type return value: T.vector
        :return return value: Each element of the vector represents the log-probability of the i-th sampled set w.r.t.
        the ensemble matrix L.

        """

        return T.log(self.prob_z(L, samples))
