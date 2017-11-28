import cPickle as pickle
import random
import numpy as np
from fast_train import Trainer
import argparse
import os
from timeit import default_timer as timer
from multiprocessing import Pool

processed_data_path = '../../../Data/Processed_Data/'
category_list = ['Wine']
dim_att = 100
n_steps = 50
eta_0 = 0.1
power = 0.01
num_workers_for_noise_data = 1

class FLIC(object):
    """Class for training and retrieving a FLIC model.

    Example Usage:

    # Training
    data = pickle.load(open(data_path, 'rb'))   # data_path to a pickle file containing a FLIC dataset.
    flic = FLIC()
    flic.train(data, noise_data=None, dim_att=100, n_steps=50)
    flic.store_model(path)

    # Retrieving a latent vector after training
    flic = FLIC()
    flic.load_model(path)
    lat_vec = flic[Object_ID]   # Object_ID is a string denoting the stringkey of a product. 'lat_vec' is now a numpy
                                # vector containing a 'dim_att' dimensional latent vector.

    """

    def __init__(self):
        """Initalizes an empty FLIC model."""
        self._obj_to_index = {}
        self._index_to_obj = {}
        self._w_unaries = None
        self._w_att = None
        self._trained = False
        self._loaded = False

    def noise(self, data):
        """Constructs the noise data to train the FLIC model with Noise Constrastive Estimation (NCE).

        This function returns the noise data needed by the function self.train(). It construct the noise data
        by sampling sets using object ID's given by 'data'. Each object ID gets included in each sampled set
        with probability according to its relative frequency in 'data'.

        :type data: set of frozensets. Each item in the frozensets is a string.
        :param data: The dataset of the model. Each frozenset contains the object ID's belonging to one set.

        :type return value: set of frozensets. Each item in the frozensets is a string.
        :return return value: Each frozenset contains the object ID's belonging to one noise set. The number of such
        frozensets is the same as the number of frozensets in 'data'.

        """

        # Measure time for building noise data
        start = timer()


        # Collect object ID's and their relative frequencies.
        rel_freq = {}
        for s in data:
            for i in s:
                if i in rel_freq:
                    rel_freq[i] += 1
                else:
                    rel_freq[i] = 1
        for i in rel_freq:
            rel_freq[i] /= float(len(data))


        # Sample noise data sets consisting of object ID's. Each object ID gets added to each sampled set with
        # probability according to its relative frequency in 'data'.
        # This sampling process is parallelized.
        random.seed(0)
        noise_data = set()
        pool =  Pool(processes=num_workers_for_noise_data)
        noise_data = set()
        for process in range(num_workers_for_noise_data):
            rand_generator = random.Random()
            rand_generator.jumpahead(random.randint(1,10000000))
            pool.apply_async(_sample_sets, (rel_freq, len(data)/num_workers_for_noise_data, rand_generator),
                             callback= noise_data.update)
        pool.close()
        pool.join()

        # Sample remaining sets such that 'noise_data' contains the same number of elements as 'data'.
        while len(noise_data) < len(data):
            rand_generator = random.Random()
            rand_generator.jumpahead(random.randint(1, 10000000))
            noise_data.update(_sample_sets(rel_freq, len(data)-len(noise_data), rand_generator))

        print("Time for constructing the noise data = {} ".format(timer()  - start))
        print("Num items = {}".format(len(rel_freq)))
        print("Num sets = {}".format(len(data)))

        return noise_data

    def _transform_data(self, data, noise_data):
        """Transforms the 2 data sets data and noise_data into a new format, such that the Trainer class can be called
        using the resulting data sets as arguments. Using a dictionary (see return value 3)), all products get
        assigned an integer in the range [0,1,...,#objects]. #objects is the total number of objects product appearing
        in 'data' and 'noise_data'.

        :type data: set of frozensets. Each item in the frozensets is a string.
        :param data: The dataset of the model. Each frozenset contains the object ID's belonging to one set.
        These frozensets will be used to train the model.

        :type noise_data: set of frozensets. Each item in the frozensets is a string.
        :param noise_data: The noise dataset of the model. Each frozenset contains the object ID's belonging to one set.

        :type return value: A tuple: (set of list of int, set of list of int,
                                      dict(string -> int), dict(int -> string))
        :return return value: A tuple consisting of the following 4 elements:
                              1) A list, where each item of the list consists of a list of indices denoting
                                 the objects in a corresponding frozenset of 'data'.
                              2) A list, where each item of the list consists of a list of indices denoting
                                 the objects in a corresponding frozenset of 'noise_data'.
                              3) A dictionary which maps objects ID's to integer in the range [0,1,...,#objects-1],
                                 where #objects is the total number of objects (products) appearing in 'data'.
                              4) A dictionary which maps integers in the range [0,1,...,#objects-1] to object ID's.
        """

        # Construct return values 1, 3 and 4 (i.e. transform 'data' and build
        # dictionaries from object to index and vice versa).
        data_l = []
        noise_data_l = []
        obj_to_index = {}
        index_to_obj = {}
        num_items = 0
        for s in data:
            l = []
            for i in s:
                # Add object to dictionaries if necessary
                if i not in obj_to_index:
                    obj_to_index[i] = num_items
                    index_to_obj[num_items] = i
                    num_items += 1

                l.append(obj_to_index[i])
            data_l.append(l)

        # Construct return value 2 (transformed noise_data)
        for s in noise_data:
            l = []
            for i in s:
                l.append(obj_to_index[i])
            noise_data_l.append(l)

        return (data_l, noise_data_l, obj_to_index, index_to_obj)

    def train(self, data, noise_data=None, dim_att=100, n_steps=10, eta_0=0.1, power=0.01):
        """Train the FLIC model.

        After training, the latent vectors w_unaries and w_att can be retrieved by calling 'get_w_unaries()'
        and get_w_att(), respectively.

        :type data: set of frozensets. Each item in the frozensets is a string.
        :param data: The dataset of the model. Each frozenset contains the object ID's belonging to one set.
        These frozensets will be used to train the model.

        :type noise_data: Either None or a set of frozensets. Each item in the frozensets is a string.
        :param noise_data: The noise dataset of the model. Each frozenset contains the object ID's belonging to one set.
        These frozensets will be used to train the model. If noise_data is None, then the noise_data will get generated
        in this function.

        :type dim_att: int
        :param dim_att: The dimension of the latent attraction vectors.

        :type n_steps: int
        :param n_steps: A number proportional to the number of steps of the FLIC model. If n_steps gets increased by
        1, the whole training set is looked at around 5 additional times.

        :type eta_0: float
        :param eta_0: A parameter used to construct the learning rate. Recommended values are in the magnitude of 0.1.

        :type power: float
        :param power: A parameter used to construct the learning rate. Recommended values are in [0.001, 0.01].

        """

        # Construct noise data if necessary
        if noise_data is None:
            noise_data = self.noise(data)

        # Transform the data sets to bring them in the right shape for a call to the 'Trainer' class.
        print("Transforming Data.")
        (data_trf, noise_data_trf, self._obj_to_index, self._index_to_obj) = self._transform_data(data, noise_data)
        n_items = len(self._obj_to_index)

        # Train model
        unaries_noise = np.asarray(1e-3*np.random.rand(n_items), dtype=np.float64)

        trainer = Trainer(data_trf, noise_data_trf, unaries_noise, n_items=n_items, dim_att=dim_att, dim_rep=0)
        print("Begin training...")
        # For 'dim_att'=10, the parameter values 'eta_0'=0.1 and 'power'=0.01 are recommended.
        # For 'dim_att'=100, 'power' can be reduced to 0.001 (or also 0.01).
        start = timer()
        trainer.train(n_steps=n_steps, eta_0=eta_0, power=power)
        print("Training time = {}.".format(timer()-start))

        # Collect the latent vectors
        self._trained = True
        self._w_unaries = trainer.unaries
        self._w_att = trainer.W_att
        self._dim_att = dim_att

    def store_model(self, data_path):
        if not self._trained and not self._loaded:
            raise(ImportError, "No model available for storing.")
        f = open(data_path, 'wb')
        pickle.dump(self._obj_to_index, f)
        pickle.dump(self._index_to_obj, f)
        pickle.dump(self._w_unaries, f)
        pickle.dump(self._w_att, f)
        f.close()

    def load_model(self, data_path):
        f = open(data_path, 'rb')
        self._obj_to_index = pickle.load(f)
        self._index_to_obj = pickle.load(f)
        self._w_unaries = pickle.load(f)
        self._w_att = pickle.load(f)
        self._dim_att = self._w_att[0].shape[0]
        self._loaded = True

    def set_model_by_hand(self, keys, w_unaries, w_att, path_to_store):
        self._obj_to_index = {}
        self._index_to_obj = {}
        #self._w_unaries = np.zeros((len(keys),), dtype=float)
        #self._w_att = np.zeros((len(keys), w_att.shape[1]))
        self._w_unaries = w_unaries
        self._w_att = w_att

        index = 0
        for k in keys:
            if k not in self._obj_to_index:
                self._obj_to_index[k] = index
                self._index_to_obj[index] = k
                index += 1

        self._loaded = True
        self.store_model(path_to_store)


    def __getitem__(self, object_ID):
        """Get the attraction (latent) vector for an object.

        :type object_ID: string
        :param object_ID: The stringkey for a product in the data set.

        :type return value: numpy.ndarray(shape=(dim_att,))
        :return return value: The latent attraction vector for the product with the key 'object_ID'

        """
        if object_ID not in self._obj_to_index:
            raise KeyError("This FLIC model has no key '{}'.".format(object_ID))
        return self.get_w_att()[self._obj_to_index[object_ID]]


    # Properties
    def get_dim_att(self):
        if not self._trained and not self._loaded:
            raise ValueError('FLIC model is not yet trained or loaded.')
        return self._dim_att

    def set_dim_att(self):
        raise RuntimeError('It is not allowed to set w_att!')

    def get_w_unaries(self):
        if not self._trained and not self._loaded:
            raise ValueError('FLIC model is not yet trained or loaded.')
        return self._w_unaries

    def set_w_unaries(self, w):
        if not self._trained and not self._loaded:
            raise ValueError('FLIC model is not yet trained or loaded.')
        if np.shape(w) != np.shape(self._w_unaries) or self._w_unaries == None:
            raise ValueError('Shapes do not agree.')
        self._w_unaries = w

    def get_w_att(self):
        if not self._trained and not self._loaded:
            raise ValueError('FLIC model is not yet trained or loaded.')
        return self._w_att

    def set_w_att(self, w):
        if not self._trained and not self._loaded:
            raise ValueError('FLIC model is not yet trained or loaded.')
        if np.shape(w) != np.shape(self._w_att) or self._w_att == None:
            raise ValueError('Shapes do not agree.')
        self._w_att = w

    w_unaries = property(get_w_unaries, set_w_unaries, doc='The utility vectors of the items in the model '
                                                           '(a dict from object ID\'s to numpy vectors).')
    w_att = property(get_w_att, set_w_att, doc='The latent attraction vectors of the items in the model '
                                               '(a dict from object ID\'s to numpy vectors).')

    dim_att = property(get_dim_att, set_dim_att, doc='The dimension of the latent attraction vector')

def _sample_sets(rel_freq, num_sets, rand_generator):
    """Samples noise data sets.

     Note: This function is only for internal use of the FLIC class. It is stored outside of the FLIC class
     because otherwise it can't be called by the Pool class of the multiprocessing module.

     """

    sets = set()
    while len(sets) < num_sets:
        l = []
        for i in rel_freq:
            r = rand_generator.random()
            if r < rel_freq[i]:
                l.append(i)
        sets.add(frozenset(l))
    return sets

def parse_arguments():
    parser = argparse.ArgumentParser(description="This scripts trains the FLIC model for given FLIC data sets. The "
                                                 "resulting models will be stored on disk.")
    parser.add_argument("-p", "--processed_data_path",
                        help="The path where the folder containing the FLIC data set (which should be named "
                             "'FLIC_sets.pkl') is stored. The resulting 'FLIC_model.pkl' will also be stored "
                             "here (Default: '../../../Data/Processed_Data/')")
    parser.add_argument("-c", "--category_list", nargs='+',
                        help= "A list of categories for which FLIC model will be trained. (Default: Wine).")
    parser.add_argument("-d" ,"--dim_att", help="The dimension of the latent attraction vectors. "
                                                "(Default: 100)", type=int)
    parser.add_argument("-n" ,"--number_steps", help="A number proportional to the number of steps of the FLIC model. "
                                                     "If this number gets increased by 1, the whole training set is"
                                                     "looked at about 5 additional times. (Default: 50)", type=int)
    parser.add_argument("-w" ,"--num_workers", help="The number of processes to work on constructing the noise data. "
                                                "(Default: 1)", type=int)
    parser.add_argument("--eta_0", help="A parameter used to construct the learning rate. Recommended values are in "
                                        "the magnitude of 0.1. (Default: 0.1)", type=float)
    parser.add_argument("--power", help="A parameter used to construct the learning rate. Recommended values are in "
                                        "the magnitude of [0.001, 0.01]. (Default: 0.01)", type=float)

    args = parser.parse_args()
    if args.processed_data_path:
        global processed_data_path
        processed_data_path = args.processed_data_path
    if args.category_list:
        global category_list
        category_list = args.category_list
    if args.dim_att:
        global dim_att
        dim_att = args.dim_att
    if args.number_steps:
        global n_steps
        n_steps = args.number_steps
    if args.num_workers:
        global num_workers_for_noise_data
        num_workers_for_noise_data = args.num_workers
    if args.eta_0:
        global eta_0
        eta_0 = args.eta_0
    if args.power:
        global power
        power = args.power

def data_subset(data, p):
    """ Return the p-th percentage of 'data'. Which items of 'data' get returned is determined
    by the iterator which is called on 'data'.

    :type data: set
    :param data: A set of arbitrary items

    :type return value: The same as data
    :return return value: The p-th percentage of 'data' (i.e. about p*len(data) elements are returned).
    """

    count = 0
    num_items = len(data)
    new_data = set()
    for item in data:
        if count > num_items*p:
            break
        new_data.add(item)
        count += 1
    return new_data

def main():
    """For each category given in 'category_list', this function train the FLIC model and store it in the
    folder where the corresponding training data sets are.
    """

    parse_arguments()

    for category in category_list:

        # Load data
        print("Loading FLIC sets for category {}.".format(category))
        data_path = processed_data_path + '/' + category + '/FLIC_sets.pkl'
        data = pickle.load(open(data_path, 'rb'))
        print("FLIC sets loaded. Constructing noise data...")
        flic = FLIC()

        # Load or construct noise data
        noise_path = processed_data_path + '/' + category + '/FLIC_noise_sets.pkl'
        data_loaded = False
        if 'FLIC_noise_sets.pkl' in os.listdir(processed_data_path + '/' + category + '/'):
            f = open(noise_path, 'rb')
            noise_data = pickle.load(f)
            f.close()
            if (len(noise_data) == len(data)):
                print("Noise data loaded.")
                data_loaded = True
        if not data_loaded:
            noise_data = flic.noise(data)
            f = open(noise_path, 'wb')
            pickle.dump(noise_data, f)
            f.close()
            print("Noise data constructed.")

        # Train and store model
        print("Begin training {}...\n".format(category))
        flic.train(data, noise_data, dim_att=dim_att, n_steps=n_steps, eta_0=eta_0, power=power)
        print("\nTraining of category {} finished. Storing model...".format(category))
        flic.store_model(processed_data_path + '/' + category + '/FLIC_model_attdim' + str(dim_att) + '_nsteps' + str(n_steps) +'.pkl')
        print("Model stored.\n")

if __name__ == '__main__':
    main()
