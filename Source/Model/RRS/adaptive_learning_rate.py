from Util.data_helper import store, load
import sys
import theano
import numpy as np


class adaptive_learning_rate:
    """Class which decreases the learning rates of 1-2 learning rates depending on a validation error."""
    def __init__(self, lr_1 = None, lr_2 = None):
        """Setting the learning rates of lr_1 and lr_2 (only if they are given as the arguments."""

        self.val_error = []
        self.num_stored_errors = 3
        self.division_factor = 4
        if lr_1:
            lr_1.set_value(1e-3)
        if lr_2:
            lr_2.set_value(1e-3)

    def _striclty_increasing_val_errors(self):
        if len(self.val_error) < self.num_stored_errors:
            return False

        start = 0
        for error in self.val_error:
            if start >= error:
                return False
            else:
                start = error

        return True


    def adapt_learning_rate(self, val_error, lr_1, lr_2 = None):
        """ Decreases the learning rate if val_error is bigger than its predecessor.

        :type val_error: float
        :param val_error: The error on the validation set

        :type lr_1: theano.shared_variable
        :param lr_1: The first learning rate.

        :type lr_2: theano.shared_variable
        :param lr_2: The second, optional learning rate.

        :type return value: Boolean
        :param return value: True if at least one learning rate was decreased in this function, and False otherwise.
        """

        self.val_error.append(val_error)
        if len(self.val_error) > self.num_stored_errors:
            self.val_error.pop(0)


        if self._striclty_increasing_val_errors():
            lr_1.set_value(np.float64(lr_1.get_value()/self.division_factor).astype(theano.config.floatX))
            print("Learning rate(s) adapted.")
            print("Learning rate 1 = {}".format(lr_1.get_value()))

            if lr_2 is not None:
                lr_2.set_value(np.float64(lr_2.get_value()/self.division_factor).astype(theano.config.floatX))
                print("Learning rate 2 = {}".format(lr_2.get_value()))

            self.val_error = [val_error]
            return True

        else:
            return False

    def make_smaller_unconditionally(self, lr_1, lr_2=None):
        lr_1.set_value(np.float64(lr_1.get_value() / self.division_factor).astype(theano.config.floatX))

        if lr_2 is not None:
            lr_2.set_value(np.float64(lr_2.get_value() / self.division_factor).astype(theano.config.floatX))



