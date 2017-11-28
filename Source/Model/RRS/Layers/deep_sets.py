import theano.tensor as T
from theano.printing import Print
from Util.nn.basic import Layer, apply_dropout
from Util.nn.initialization import create_shared, ReLU, sigmoid, tanh, softmax, linear, get_activation_by_name
from theano.ifelse import ifelse

class PermEquivLayer(Layer):
    """ Implements a permuation equivariant layer as described in the paper 'Deep Learning with Sets and Point Clouds
    (Ravanbakhsh et al., 2017).

    A permutation equivariant layer implements y = f(xL + 11'xG + b), where x is a matrix of size NxK with N items stored
    in the rows. Each item hence is represented by a feature vector of dimension K. L and G are matrices of size KxP,
    where P is the output dimension. 11' is a matrix consisting of all ones and f is a (non-linear) activation function.
    b is a matrix of matrix of size NxP, where each row contains the same values.
    y is of size NxP. Each row of y belongs to one item of the N items.

    """

    def __init__(self, n_in, n_out, activation = sigmoid, has_bias=True):
        """
        :type n_in: int
        :param n_in: The input dimension of each item in the input set.
        :type n_out: int
        :param n_out: The output dimension for each item in the input set (i.e. P).
        :type activation: Theano tensor function
        :param activation: A (non-linear) activation function
        :type has_bias: bool
        :param has_bias: Whether the layer should use the bias b.

        """
        super(PermEquivLayer, self).__init__(n_in, n_out, activation, False, has_bias)

    def create_parameters(self):
        # Use parameter initalization from the base class
        # The first P columns of the resulting matrix W represents the matrix L
        # and the last P columns represents the matrix G.
        self.initialize_params(self.n_in, 2*self.n_out, self.activation)
        if self.has_bias:
            self.b = create_shared(self.b.get_value()[:self.n_out])

    def forward(self, x):
        """Return y = f(xL + 11'xG + b), where f is the activation function applied elementwise and L,G,b are the
        shared variables of the layer.

        :type x: T.tensor, shape = (batch_size, num_items_in_set, dim_per_item)
        :param x: The input matrix. The first dimension goes over all sets in the batch, the second dimension represents
        all items in the set and the third dimension encodes the feature vectors of the items.

        :type return value: T.tensor, shape = (batch_size, num_items_in_set, dim_per_item)
        :param x: Returns y = f(xL + 11'xG + b)
        """

        xG = T.dot(x, self.W[:, self.n_out:])
        s = T.shape(x)
        if self.has_bias:
            return self.activation(
                T.dot(x, self.W[:, :self.n_out]) + T.batched_dot(T.ones((s[0], s[1], s[1])), xG) + self.b
            )
        else:
            return self.activation(
                T.dot(x, self.W[:, :self.n_out]) + T.batched_dot(T.ones((s[0], s[1], s[1])), xG)
            )

    @property
    def params(self):
        if self.has_bias:
            return [ self.W, self.b ]
        else:
            return [ self.W ]

    @params.setter
    def params(self, param_list):
        self.W.set_value(param_list[0].get_value())
        if self.has_bias: self.b.set_value(param_list[1].get_value())

class DeepSetLayer(Layer):
    """A Deep Set Layer as described in the paper 'Deep Sets' (Zaheer et al., 2017).

    As input a matrix needs to be given to the model, where each row contains the feature vector of an item of a set.
    The matrix needs to have size (num_items_in_set, num_in), where 'num_in' is the size of each feature vector.
    Additionally, a context vector of size 'context_size' can be given to the model.
    The model will then calculate a deep representation for the whole set and the context
    and give a vector of size 'num_out' as its output.

    """

    def __init__(self, num_in, num_hidden, context_size, num_perm_equiv_layers = 1, dropout = None, activation=tanh):
        """
        :type num_in: Int
        :param num_in: The size (length) of each feature vector in the set. (NOT the number of items in the set)
        :type context_size: Int
        :param context_size: The length of the context feature vector. 0 if no context.
        :type num_out: Int
        :param num_out: The length of the result vector (the output of the forward progagation algorithm).
        :type num_hidden: Int
        :param num_hidden: The number of hidden units in each Permutation Equivariant Layer.
        :type num_perm_equiv_layers: Int
        :param num_perm_equiv_layers: The number of Permutation Equivariant Layers (i.e. number of hidden layers).
        :type context_size: Int
        :param context_size: The length of the context feature vector. 0 if no context.
        :type activation: Theano tensor function
        :param activation: The activation function

        """
        print("Using DeepSetLayer 3.")

        self.activation = activation
        self.num_in = num_in
        self.num_hidden = num_hidden
        self.context_size = context_size
        self.num_perm_equiv_layers = num_perm_equiv_layers
        self.dropout = dropout
        if num_perm_equiv_layers <= 0:
            raise(ValueError, "The number of hidden layers must at least be one!")

        # Sub layers
        self.hidden_layers = []
        self.internal_layers = []
        for i in range(self.num_perm_equiv_layers):
            if i == 0:
                self.hidden_layers.append(PermEquivLayer(num_in, num_hidden, self.activation))
            else:
                self.hidden_layers.append(PermEquivLayer(num_hidden, num_hidden, self.activation))

        self.linear_layer = Layer(n_in=self.num_hidden + self.context_size, n_out=1, activation=linear)
        self.internal_layers.extend(self.hidden_layers)
        self.internal_layers.append(self.linear_layer)

    def forward(self, sentences, context):
        """ Apply the forward propagation algorithm on the Deep Set Layer.

        The following function g(x) is calculated:
        f(x) = MaxPool(PermEquivLayers(activation([W1*x + b1]).
        g(x) = LogReg([f(x), context)])

        :type sentences: T.matrix
        :param x: Each row of the matrix contains the embedding vector of one item in the set.
        :type context: T.vector
        :param context: The context vector.

        :type return value: float (between 0 and 1)
        :return return value: g(x) (the probability that the product described by sentences fits
        well to the context product.
        """

        # Permutation equivariant layers
        for perm_equiv_layer in self.hidden_layers:
            sentences = perm_equiv_layer.forward(sentences.dimshuffle('x', 0, 1)).dimshuffle(1,2)
            if self.dropout is not None:
                sentences = apply_dropout(sentences, self.dropout)

        # Pooling layer
        pooled_output = T.max(sentences, axis=0)

        if self.dropout is not None:
            context = apply_dropout(context, self.dropout)

        # Concatenate sentences and context to make one big vector
        sent_plus_context = T.concatenate([pooled_output, context], axis=0)

        # Logistic regression layer
        scal = self.linear_layer.forward(sent_plus_context)[0]
        log_reg = 1.0/(1.0+T.exp(-scal))

        return log_reg

    def get_params(self):
        return [x for layer in self.internal_layers for x in layer.params]

    def set_params(self, param_list):
        start = 0
        for layer in self.internal_layers:
            end = start + len(layer.params)
            layer.params = param_list[start:end]
            start = end

    params = property(get_params, set_params)
