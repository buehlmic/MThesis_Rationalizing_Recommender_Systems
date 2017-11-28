# coding: utf-8
''' 
  @author: Sebastian Tschiatschek (Sebastian.Tschiatschek@microsoft.com)
  @modified: Michael Bühler (buehler_michael@bluewin.ch)
'''

from __future__ import division, print_function

from itertools import chain

import time
import numpy as np

from _flm import flm_train, flm_obj
#import NCE
import IPython


def fast_sample(probabilities, n_samples):
    out_size = _ffi.new("int *")
    time_s = time.time()
    result = _lib.sample(
        _ffi.cast("const double *", probabilities.ctypes.data),
        np.size(probabilities),
        n_samples,
        out_size)
    samples = []
    current = []
    time_s = time.time()
    for i in range(out_size[0]):
        x = result[i]
        if x == -1:
            samples.append(current)
            current = []
        else:
            current.append(x)
    return samples


class Trainer:
    def __init__(self, model_data, noise_data, unaries_noise,
                 unaries=None, n_items=None, dim_att=5, dim_rep=5):
        data = []
        for i, subset in enumerate(chain(model_data, noise_data)):
            subset = list(subset)
            label = 1 if i < len(model_data) else 0
            if data:
                data.append(-1)
            data.append(label)
            data.extend(subset)

        self.data = data
        self.data_size = len(data)
        self.num_samples = len(model_data) + len(noise_data)
        self.orig_data = model_data
        self.orig_nois = noise_data

        assert n_items is not None
        self.n = n_items
        self.dim_rep = dim_rep
        self.dim_att = dim_att

        # initalize weights
        self.W_rep = 1e-3 * np.asarray(np.random.rand(*(self.n, dim_rep)), dtype=np.float64)
        self.W_att = 1e-3 * np.asarray(np.random.rand(*(self.n, dim_att)), dtype=np.float64)

        self.unaries_noise = np.array(unaries_noise, dtype=np.float64)
        self.unaries = np.copy(self.unaries_noise)
        self.iteration = 0
        self.n_logz = np.array([-np.sum(np.log(1 + np.exp(self.unaries)))], dtype=np.float64)

    def train(self, n_steps, eta_0, power, eta_rep=None, eta_att=None, reg_rep=0.0, reg_att=0.0, plot=False, verbose=False):
        step = 100 * self.data_size
        if plot:
            from matplotlib import pyplot as plt
        n_steps *= self.num_samples*5+1

        # if no learning rates for rep, att are given, set them equal to eta_0
        if eta_rep is None:
            eta_rep = eta_0

        if eta_att is None:
            eta_att = eta_0

        values = []
        if plot:
            values.append(self.objective())

        assert self.W_rep.shape == (self.n, self.dim_rep)
        assert self.W_att.shape == (self.n, self.dim_att)
        assert np.size(self.unaries) == self.n
        print("!!! sum_W_rep=%f, sum_W_att=%f" % (np.sum(self.W_rep), np.sum(self.W_att)))
        print("!!! dev_unaries=%f" % (np.sum((self.unaries - self.unaries_noise)**2 )))
        flm_train(
            self.data, self.data_size, n_steps,
            eta_0, eta_rep, eta_att, power, 0,
            self.unaries_noise,
            self.W_rep,
            self.W_att,
            self.unaries,
            self.n_logz,
            self.n, self.dim_rep, self.dim_att,
            reg_rep, reg_att)

        # # compute normalized similarities
        # import numpy.linalg
        # if self.dim_rep > 1:
        #     sim = np.zeros((self.dim_rep, self.dim_rep))
        #     M = self.W_rep
        #     for i in range(self.dim_rep):
        #         for j in range(self.dim_rep):
        #             if i >= j:
        #                 continue
        #             v1 = np.copy(M[:, i])
        #             v2 = np.copy(M[:, j])
        #             v1 /= np.linalg.norm(v1)
        #             v2 /= np.linalg.norm(v2)
        #             sim[i, j] = v1.T.dot(v2)
        #     # print(sim)
        #     idx = np.argmax(sim)
        #     idx = np.unravel_index(idx, np.shape(sim))
        #     if sim[idx] > 0.9 and np.linalg.norm(M[:, idx[0]]) > 1.:
        #         print("Resetting: ", idx)
        #         M[:, idx[1]] += M[:, idx[0]]
        #         M[:, idx[0]] = 1e-3 * np.random.rand(self.n)

        # if self.dim_att > 1:
        #     sim = np.zeros((self.dim_att, self.dim_att))
        #     M = self.W_att
        #     for i in range(self.dim_att):
        #         for j in range(self.dim_att):
        #             if i >= j:
        #                 continue
        #             v1 = np.copy(M[:, i])
        #             v2 = np.copy(M[:, j])
        #             v1 /= np.linalg.norm(v1)
        #             v2 /= np.linalg.norm(v2)
        #             sim[i, j] = v1.T.dot(v2)
        #     # print(sim)
        #     idx = np.argmax(sim)
        #     idx = np.unravel_index(idx, np.shape(sim))
        #     if sim[idx] > 0.9 and np.linalg.norm(M[:, idx[0]]) > 1.:
        #         print("Resetting: ", idx)
        #         M[:, idx[1]] += M[:, idx[0]]
        #         M[:, idx[0]] = 1e-3 * np.random.rand(self.n)

        if plot:
            values.append(self.objective())
        if plot:
            plt.plot(values, 'bo--')
            plt.show()

    def objective(self, ldata=None, ldata_noise=None):
        from DiversityModel import ModularFun, DiversityFun

        f_noise = ModularFun(range(self.n), self.unaries_noise)
        f_model = DiversityFun(range(self.n), self.dim_rep, self.dim_att)
        f_model.n_logz = self.n_logz
        f_model.W_rep = self.W_rep
        f_model.W_att = self.W_att
        f_model.utilities = self.unaries

        if ldata is None:
            ldata = self.orig_data
        if ldata_noise is None:
            ldata_noise = self.orig_nois

        # print("*** OBJECTIVE=", NCE(f_model, f_noise)._objective(self.orig_data, self.orig_nois))

        # TODO: Set up the following code again.
        assert False
        #return NCE(f_model, f_noise)._objective(ldata, ldata_noise)
        return None

if __name__ == "__main__":
    print("test.")
    W = np.random.rand(30, 5)
    IPython.embed()
