/* 
  @author: Sebastian Tschiatschek (Sebastian.Tschiatschek@microsoft.com)
  @modified: Michael Bühler (buehler_michael@bluewin.ch)
*/

#include <cmath>
#include <random>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
#include "pybind11/include/pybind11/numpy.h"
// #include "bk/graph.h"
#include <chrono>

namespace py = pybind11;


template <typename T> class np_wrap {
  public:
    np_wrap(py::array_t<T>& array) : buf(array.request()) {
        data_ = (T*) buf.ptr;
    }

    size_t index(size_t i) const {
        return (i * buf.strides[0]) / buf.itemsize;
    }

    size_t index(size_t i, size_t j) const {
        return (i * buf.strides[0] + j * buf.strides[1]) / buf.itemsize;
    }

    T operator [](const std::pair<size_t, size_t>& p) const {
        return data_[index(p.first, p.second)];
    }

    T& operator [](const std::pair<size_t, size_t>& p) {
        return data_[index(p.first, p.second)];
    }

    T operator [](size_t i) const {
        return data_[index(i)];
    }

    T& operator [](size_t i) {
        return data_[index(i)];
    }

  public:
    py::buffer_info buf;

  private:
    T* data_;
};


template <typename T> py::array_t<T> create_array(const std::vector<size_t>& dims) {
    // From http://pybind11.readthedocs.org/en/latest/advanced.html
    std::vector<size_t> strides(dims.size());
    size_t total_jump = sizeof(T);
    for (size_t i=strides.size()-1; i<strides.size(); i--) {
        strides[i] = total_jump;
        total_jump *= dims[i];
    }
    return py::array(py::buffer_info(
        nullptr,  /* Pointer to data (nullptr -> ask NumPy to allocate!) */
        sizeof(T),  /* Size of one item */
        py::format_descriptor<T>::value(),  /* Buffer format */
        dims.size(),  /* How many dimensions? */
        dims,  /* Number of elements for each dimension */
        strides  /* Strides for each dimension */
    ));
}


double sigmoid(double x) {
    if (x >= 0) {
        return 1. / (1 + exp(-x));
    } else {
        return exp(x) / (1 + exp(x));
    }
}


double _flm_obj(size_t n_S, int *S, size_t n, size_t dim_rep, size_t dim_att,
        double n_logz,
        double *utilities, double *W_rep, double *W_att) {
#define IDX_REP(w, v) ((w) * dim_rep + (v))
#define IDX_ATT(w, v) ((w) * dim_att + (v))
    // empty set as input
    /* if (n_S == 0) {
        return n_logz - 1000.;
    } */
    // std::cout << "Obj start." << std::endl;

    double value = n_logz;
    int el;

    // utilities
    for (size_t i = 0; i < n_S; i++) {
        el = S[i];
        value += utilities[el];
    }

    double max;
    double w;

    // repulse facloc
    for (size_t d = 0; d < dim_rep; d++) {
        max = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < n_S; i++) {
            el = S[i];
            w = W_rep[IDX_REP(el, d)];
            if (max < w) {
                max = w;
            }
            value -= w; // TODO
        }
        if (n_S == 0)
            break;
        value += max;
    }

    // attractive facloc
    for (size_t d = 0; d < dim_att; d++) {
        max = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < n_S; i++) {
            el = S[i];
            w = W_att[IDX_ATT(el, d)];
            if (max < w) {
                max = w;
            }
            value += w; // TODO
        }
        if (n_S == 0)
            break;
        value -= max;
    }

    // std::cout << "Obj end." << std::endl;
    return value;
}

double flm_obj(py::array_t<int32_t> S, double n_logz,  py::array_t<double> utilities, py::array_t<double> W_rep, py::array_t<double> W_att) {
    np_wrap<int32_t> wrap_S(S);

    np_wrap<double> wrap_utilities(utilities);
    np_wrap<double> wrap_W_rep(W_rep);
    np_wrap<double> wrap_W_att(W_att);

    size_t n = wrap_W_rep.buf.shape[0];
    size_t dim_rep = wrap_W_rep.buf.shape[1];
    size_t dim_att = wrap_W_att.buf.shape[1];

    return _flm_obj(wrap_S.buf.shape[0], &wrap_S[0], n, dim_rep, dim_att,
            n_logz, &wrap_utilities[0], &wrap_W_rep[0], &wrap_W_att[0]);
}

#define INDEX(i,j,stride) i*stride + j

void _flm_grad(size_t n_S, int *S,
        size_t n, size_t dim_rep, size_t dim_att,
        double *n_logz,
        double *wrap_utilities,
        double *wrap_W_rep, double *wrap_W_att,
        double *wrap_grad_n_logz,
        double *grad_utilities,
        double* wrap_grad_W_rep, double *wrap_grad_W_att) {
    int el;

    // std::cout << "Grad start." << std::endl;
    // std::cout << "n_S: " << n_S << std::endl;

    // utilities
    for (size_t i = 0; i < n_S; i++) {
        el = S[i];
        // TODO: FIX grad_utilities[el] += 1.;
        grad_utilities[el] += 1;
    }

    // std::cout << "ut ok  " << std::endl; 

    double w;
    size_t max_idx;
    double max;

    // repulse facloc
    for (size_t d = 0; d < dim_rep; d++) {
        max = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < n_S; i++) {
            el = S[i];
            w = wrap_W_rep[INDEX(el, d, dim_rep)];
            if (max < w) {
                max = w;
                max_idx = el;
            }
            wrap_grad_W_rep[INDEX(el, d, dim_rep)] -= 1; // TODO
        }
        if (n_S == 0)
           break;
        wrap_grad_W_rep[INDEX(max_idx, d, dim_rep)] += 1;
    }

    // std::cout << "rep ok  " << std::endl;

    // attractive facloc
    for (size_t d = 0; d < dim_att; d++) {
        max = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < n_S; i++) {
            el = S[i];
            w = wrap_W_att[INDEX(el, d, dim_att)];
            if (max < w) {
                max = w;
                max_idx = el;
            }
            // std::cout << "w: " << w << "    el: " << el << std::endl;
            wrap_grad_W_att[INDEX(el, d, dim_att)] += 1; // TODO
        }
        if (n_S == 0)
           break;
        // std::cout << "max_idx: " << max_idx << std::endl;
        wrap_grad_W_att[INDEX(max_idx, d, dim_att)] -= 1;
    }

    // std::cout << "att ok" << std::endl;

    // n_logZ
    for (size_t i = 0; i < 1; i++) {
        wrap_grad_n_logz[i] += 1;
    }

    // std::cout << "Grad end." << std::endl;
}


void flm_grad(py::array_t<int32_t> S,
        py::array_t<double> n_logz,
        py::array_t<double> utilities,
        py::array_t<double> W_rep, py::array_t<double> W_att,
        py::array_t<double> grad_n_logz,
        py::array_t<double> grad_utilities,
        py::array_t<double> grad_W_rep, py::array_t<double> grad_W_att) {

        np_wrap<int32_t> wrap_S(S);
        np_wrap<double> wrap_n_logz(n_logz);
        np_wrap<double> wrap_utilities(utilities);
        np_wrap<double> wrap_grad_utilities(grad_utilities);
        np_wrap<double> wrap_W_rep(W_rep);
        np_wrap<double> wrap_grad_W_rep(grad_W_rep);
        np_wrap<double> wrap_W_att(W_att);
        np_wrap<double> wrap_grad_W_att(grad_W_att);
        np_wrap<double> wrap_grad_n_logz(grad_n_logz);

        _flm_grad(wrap_S.buf.shape[0], &wrap_S[0], wrap_W_rep.buf.shape[0], wrap_W_rep.buf.shape[1], wrap_W_att.buf.shape[1],
		&wrap_n_logz[0], &wrap_utilities[0], &wrap_W_rep[0], &wrap_W_att[0],
                &wrap_grad_n_logz[0], &wrap_grad_utilities[0],
                &wrap_grad_W_rep[0], &wrap_grad_W_att[0]);
}


int indicator_vector_to_set(size_t n_elements, int* indicator_vector, int *set) {
    // index vector to set
    size_t set_idx = 0;
    for (size_t i = 0; i < n_elements; i++) {
        if (indicator_vector[i] == 1) {
            set[set_idx] = i;
            set_idx++;
        }
    }

    return set_idx;
}

double expit(double x) {
    if (x > 0) {
        return 1. / (1. + exp(-x));
    } else {
        return 1. - 1. / (1. + exp(x));
    }
}

double log1exp(double x) {
    if (x > 0) {
        return x + log1p(std::exp(-x));
    } else {
        return log1p(std::exp(x));
    }
}


// Parameters
// ==========
// data : the data samples separated by -1, first entry is the label 1 / 0
// data_size : the size of the above vector
// n_steps : how many SGD steps to perform
// eta_0, power, start_step : the step size used is
//
//      eta_0 / (start_step + iteration) ** power;
//
// weights : the weights will be stored here. Will not be initialized and the
//           provided data will be used as the first iterate. Assumed to be
//           stored in *column-first* order (FORTRAN). Should be of size n x m.
// unaries : the unaries will be stored here. Should be of size n
//
//
//////
//
/*             flm_train(
                self.data, self.data_size, step,
                eta_0, eta_rep, eta_att, power, i,
                self.unaries_noise,
                self.W_rep,
                self.W_att,
                self.unaries.ctypes.data,
                self.n_logz.ctypes.data,
                self.n, self.dim_rep, self.dim_att)*/

void _flm_apply_gradient(int32_t n, int32_t dim_rep, int32_t dim_att,
        double mu_0, double mu_rep, double mu_att, double additive_noise,
        double *unaries, double *W_rep, double *W_att, double *n_logz,
        double *grad_unaries, double *grad_W_rep, double *grad_W_att, double *grad_n_logz) {
#define IDX_REP(w, v) ((w) * dim_rep + (v))
#define IDX_ATT(w, v) ((w) * dim_att + (v))
    size_t tidx;

    // gradient for unaries
    for (size_t k = 0; k < n; k++) {
        unaries[k] += mu_0 * grad_unaries[k];
    }

    // gradient for repulsive model
    for (size_t j = 0; j < dim_rep; j++) {
        for (size_t k = 0; k < n; k++) {
            tidx = IDX_REP(k, j);
            W_rep[tidx] += mu_rep * grad_W_rep[tidx];
            if (W_rep[tidx] <= 0.) {
                 W_rep[tidx] = additive_noise * (
                    static_cast<double>(std::rand()) / RAND_MAX);
            }
        }
    }

    // gradient for attractive model
    for (size_t j = 0; j < dim_att; j++) {
        for (size_t k = 0; k < n; k++) {
            tidx = IDX_ATT(k, j);
            W_att[tidx] += mu_att * grad_W_att[tidx];
            if (W_att[tidx] <= 0.) {
                 W_att[tidx] = additive_noise * (
                    static_cast<double>(std::rand()) / RAND_MAX);
            }
        }
    }

    *n_logz += mu_0 * grad_n_logz[0];
}



void _flm_grad_and_apply(size_t n_S, int *S,
        int32_t n, int32_t dim_rep, int32_t dim_att,
        double mu_0, double mu_rep, double mu_att, double additive_noise,
        double *unaries, double *W_rep, double *W_att, double *n_logz) {
#define IDX_REP(w, v) ((w) * dim_rep + (v))
#define IDX_ATT(w, v) ((w) * dim_att + (v))
    size_t tidx;

    double w;
    size_t max_idx;
    double max;
    size_t el;

    // utilities
    for (size_t i = 0; i < n_S; i++) {
        unaries[S[i]] += mu_0;
    }

    // repulse facloc
    for (size_t d = 0; d < dim_rep; d++) {
        max = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < n_S; i++) {
            el = S[i];
            tidx = INDEX(el, d, dim_rep);
            w = W_rep[tidx];
            if (max < w) {
                max = w;
                max_idx = el;
            }
            W_rep[tidx] -= mu_rep;

            if (W_rep[tidx] <= 0.) {
                 W_rep[tidx] = additive_noise * (
                    static_cast<double>(std::rand()) / RAND_MAX);
            }
        }
        if (n_S == 0)
           break;
        W_rep[INDEX(max_idx, d, dim_rep)] += mu_rep;
    }


    // attractive facloc
    for (size_t d = 0; d < dim_att; d++) {
        max = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < n_S; i++) {
            el = S[i];
            tidx = INDEX(el, d, dim_att);
            w = W_att[tidx];
            if (max < w) {
                max = w;
                max_idx = el;
            }
            // std::cout << "w: " << w << "    el: " << el << std::endl;
            W_att[tidx] += mu_att;

            if (W_att[tidx] <= 0.) {
                 W_att[tidx] = additive_noise * (
                    static_cast<double>(std::rand()) / RAND_MAX);
            }
        }
        if (n_S == 0)
           break;
        // std::cout << "max_idx: " << max_idx << std::endl;
        W_att[INDEX(max_idx, d, dim_att)] -= mu_att;
    }

    *n_logz += mu_0;
}





// TODO: Improve implementation
double logaddexp(double a, double b) {
    return std::log(std::exp(a) + std::exp(b));
}

void reset_similar_dimensions(int32_t n, int32_t dim, double *W) {
#define IDX(w, v) ((w) * dim + (v))
    double norm_th = 1.0;

    for (size_t z=0; z < 10; z++) {
        double* Wcopy = new double[n * dim];
        double* norms = new double[dim];
        memcpy(Wcopy, W, dim * n * sizeof(double));

        // normalize vectors
        for (size_t d = 0; d < dim; d++) {
            double norm = 0.0;
            for (size_t k = 0; k < n; k++) {
                norm += Wcopy[IDX(k, d)] * Wcopy[IDX(k, d)];
            }
            norm = std::sqrt(norm);
            norms[d] = norm;
            for (size_t k = 0; k < n; k++) {
                Wcopy[IDX(k, d)] /= norm;
            }
        }

        // compute similarities
        double* sim = new double[dim * dim];
        for (size_t i = 0; i < dim; i++) {
            for (size_t j = i + 1; j < dim; j++) {
                sim[i, j] = 0;
                for (size_t l = 0; l < n; l++)
                    sim[i, j] += Wcopy[IDX(l, i)] * Wcopy[IDX(l, j)];
            }
        }

        // find largest similarity
        double max_sim = 0;
        size_t idx1 = 0;
        size_t idx2 = 0;
        for (size_t i = 0; i < dim; i++) {
            for (size_t j = i + 1; j < dim; j++) {
                if (sim[i ,j] > max_sim && norms[i] > norm_th) {
                    max_sim = sim[i, j];
                    idx1 = i;
                    idx2 = j;
                }
            }
        }
        // std::cout << "  max_sim: " << max_sim << "(" << idx1 << "," << idx2 << ", nrm: " << norms[idx1] << ")" << std::endl;

        if (idx1 != idx2 && sim[idx1, idx2] > 0.9 && norms[idx1] > norm_th) {
            std::cout << "Resetting idx " << idx1 << " because of idx " << idx2 << " (sim=" << sim[idx1, idx2] << ")." << std::endl;

            for (size_t i = 0; i < n; i++) {
                W[IDX(i, idx2)] += W[IDX(i, idx1)];
                double www = 1. / (1e-3 + W[IDX(i, idx2)]);
                if (www > 1.)
                    www = 1.;
                W[IDX(i, idx1)] = 1e-2 * www * (
                    static_cast<double>(std::rand()) / RAND_MAX);
            }
        }

        delete [] Wcopy;
        delete [] sim;
        delete [] norms;
    }
}

void flm_train(py::array_t<int32_t> py_data, int32_t data_size, int64_t n_steps,
           double eta_0, double eta_rep, double eta_att, double power, int32_t start_step,
           py::array_t<double> py_unaries_noise,
           py::array_t<double> py_W_rep, py::array_t<double> py_W_att,
           py::array_t<double> py_unaries,
           py::array_t<double> py_n_logz,
           int32_t n, int32_t dim_rep, int32_t dim_att,
           double reg_rep, double reg_att) {
#define IDX_REP(w, v) ((w) * dim_rep + (v))
#define IDX_ATT(w, v) ((w) * dim_att + (v))
    // settings; TODO make parameters
    double additive_noise_base = 1e-1;
    double additive_noise;

    // compute start indices of samples & number of noise/data samples
    np_wrap<int32_t> wrap_data(py_data);
    int32_t *tdata = &wrap_data[0];
    std::vector<long> start_indices;
    start_indices.reserve(data_size);
    start_indices.push_back(0);
    long n_noise = (tdata[0] == 0.);
    long n_model = (tdata[0] == 1.);
    for (size_t i = 1; i < data_size; i++) {
        if (tdata[i] == -1.) {
            assert(i + 1 < data_size);
            if (tdata[i+1] == 1.) {
                n_model++;
            } else {
                assert(tdata[i+1] == 0.);
                n_noise++;
            }
            start_indices.push_back(i + 1);
        }
    }
    long n_samples = n_noise + n_model;

    // array with labels
    double *labels = new double[n_samples];
    for (size_t idx = 0; idx < n_samples; idx++) {
        labels[idx] = static_cast<double>(tdata[start_indices[idx]]);
    }

    // array with lengt of samples / array with pointers to beginning of samples
    size_t *sample_lengths = new size_t[n_samples];
    int32_t **data = new int32_t*[n_samples];
    for (size_t idx = 0; idx < n_noise + n_model; idx++) {
        size_t start_idx = start_indices[idx];
        size_t end_idx;
        if (idx + 1 == start_indices.size()) {
            end_idx = data_size;
        } else {
            end_idx = start_indices[idx + 1] - 1;
        }

        size_t n_S = end_idx - start_idx - 1;
        sample_lengths[idx] = n_S;
        data[idx] = &tdata[start_idx + 1];
    }

    // compute normalizer of noise model
    np_wrap<double> wrap_unaries_noise(py_unaries_noise);
    double *unaries_noise = &wrap_unaries_noise[0];
    double logz_noise = 0.;
    for (size_t i = 0; i < n; i++) {
        logz_noise += log1exp(unaries_noise[i]);
    }

    double log_nu = std::log(
        static_cast<double>(n_noise) / static_cast<double>(n_model));

    clock_t t;
    t = clock();

    std::vector<size_t> perm(start_indices.size(), 0);
    for (int i = 0; i < start_indices.size(); i++) {
        perm[i] = i;
    }

    // wrap stuff
    np_wrap<double> wrap_unaries(py_unaries);
    double *unaries = &wrap_unaries[0];
    assert (wrap_unaries.buf.shape[0] == n);

    std::srand(start_step);
    std::cout << "INITIALIZING RNG TO " << start_step << std::endl;

    np_wrap<double> wrap_n_logz(py_n_logz);
    double *n_logz = &wrap_n_logz[0];
    np_wrap<double> wrap_W_rep(py_W_rep);
    double *W_rep = &wrap_W_rep[0];
    np_wrap<double> wrap_W_att(py_W_att);
    double *W_att = &wrap_W_att[0];

    // reserve memory for gradient
    double* grad_unaries = new double[n];
    double* grad_W_rep = new double[n * dim_rep];
    double* grad_W_att = new double[n * dim_att];
    double* grad_n_logz = new double[1];
    memset(grad_unaries, 0.0, n * sizeof(double));
    memset(grad_W_rep, 0.0, dim_rep * n * sizeof(double));
    memset(grad_W_att, 0.0, dim_att * n * sizeof(double));
    grad_n_logz[0] = 0.0;

    // backup of best model
    double* best_unaries = new double[n];
    double* best_W_rep = new double[n * dim_rep];
    double* best_W_att = new double[n * dim_att];
    double* best_n_logz = new double[1];

    std::vector<int32_t> sample;
    int i;
    double step = 1e-1;
    double old_obj = -std::numeric_limits<double>::infinity();;
    double best_obj = -std::numeric_limits<double>::infinity();;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    for (int64_t i = 0; i < n_steps; i++) {
        step = pow(static_cast<double>(i + 1), -power);

        // permute order of datasets after we have seen all sets
        if (i % start_indices.size() == 0) {
            std::random_shuffle(perm.begin(), perm.end());
        }

        //if (i % (5 * start_indices.size()) == 0) { // TODO i instead of ti
        if (i % (start_indices.size()) == 0) { // TODO i instead of ti
            double objective = 0;
            for (size_t sidx = 0; sidx < n_samples; sidx++) {
                double c_f_model = _flm_obj(sample_lengths[sidx], data[sidx],
                    n, dim_rep, dim_att,
                    n_logz[0], unaries, W_rep, W_att);

                double c_f_noise = -logz_noise;
                for (size_t j = 0; j < sample_lengths[sidx]; j++) {
                    c_f_noise += unaries_noise[data[sidx][j]];
                }

                double G = c_f_model - c_f_noise;
                double mul = labels[sidx] > .5 ? 1. : -1.;
                objective -= logaddexp(0, mul * (log_nu - G));
            }

            std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
            std::cout << "TOOK: " << std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count() << "ms" << std::endl;
            t1 = t2;

            std::cout << "Computing NCE objective... ";
            std::cout << objective << " (eta=" << step << ")" << std::endl;
            if (objective > best_obj) {
                // Model is the best we have seen that far
                // std::cout << "Best model. Storing." << std::endl;
                memcpy(best_unaries, unaries, n * sizeof(double));
                memcpy(best_W_rep, W_rep,  dim_rep * n * sizeof(double));
                memcpy(best_W_att, W_att, dim_att * n * sizeof(double));
                best_n_logz[0] = n_logz[0];

                best_obj = objective;
            } /* else {
                // std::cout << "Restoring previous best model..." << std::endl;
                memcpy(unaries, best_unaries, n * sizeof(double));
                memcpy(W_rep, best_W_rep, dim_rep * n * sizeof(double));
                memcpy(W_att, best_W_att, dim_att * n * sizeof(double));
                n_logz[0] = best_n_logz[0];
            } */
            // std::cout << std::endl;

            // check if we should reset any dimensions
            if (i % (5 * start_indices.size()) == 0) {
                std::cout << i / 5 / start_indices.size() << " steps done." << std::endl;
                if (dim_rep > 0) {
                    std::cout << "Resetting similar dimensions in repulsive weights." << std::endl;
                    reset_similar_dimensions(n, dim_rep, W_rep);
                }
                if (dim_att > 0) {
                    std::cout << "Resetting similar dimensions in attractive weights." << std::endl;
                    reset_similar_dimensions(n, dim_att, W_att);
                }

                double norm_W_rep = 0;
                for (size_t i = 0; i < dim_rep * n; i++) {
                    norm_W_rep += std::abs(W_rep[i]);
                }
                std::cout << "   norm W_rep: " << norm_W_rep << std::endl;
            }
        }

        // pick next index from permutation
        size_t idx = perm[i % start_indices.size()];
        assert (idx < start_indices.size());

	    additive_noise = additive_noise_base; // * step;

        // compute value of noise and model
        // double f_model = *n_logz;
        double f_noise = -logz_noise;

        for (size_t k = 0; k < sample_lengths[idx]; k++) {
            f_noise += unaries_noise[data[idx][k]];
        }

        size_t n_S = sample_lengths[idx]; // length of set S

        double f_model = _flm_obj(n_S, data[idx],
            n, dim_rep, dim_att,
            n_logz[0], unaries, W_rep, W_att);

        // We can now take the gradient step.
        double label = labels[idx];
        double factor = step * (label - expit(f_model - f_noise - log_nu));

        // compute gradient
        _flm_grad_and_apply(n_S, data[idx],
            n, dim_rep, dim_att,
            factor * eta_0, factor * eta_rep, factor * eta_att, additive_noise,
            unaries, W_rep, W_att, n_logz);

    }

    // copy best model
    memcpy(unaries, best_unaries, n * sizeof(double));
    memcpy(W_rep, best_W_rep, dim_rep * n * sizeof(double));
    memcpy(W_att, best_W_att, dim_att * n * sizeof(double));
    n_logz[0] = best_n_logz[0];

    double objective = 0;
    for (size_t sidx = 0; sidx < n_samples; sidx++) {
        double c_f_model = _flm_obj(sample_lengths[sidx], data[sidx],
            n, dim_rep, dim_att,
            n_logz[0], unaries, W_rep, W_att);

        double c_f_noise = -logz_noise;
        for (size_t j = 0; j < sample_lengths[sidx]; j++) {
            c_f_noise += unaries_noise[data[sidx][j]];
        }

        double G = c_f_model - c_f_noise;
        double mul = labels[sidx] > .5 ? 1. : -1.;
        objective -= logaddexp(0, mul * (log_nu - G));
    }

    for (size_t t = 0; t < n * dim_rep; t++) {
        objective -= reg_rep * std::abs(W_rep[t]);
    }

    for (size_t t = 0; t < n * dim_att; t++) {
        objective -= reg_att * std::abs(W_att[t]);
    }

    // for (size_t t = 0; t < n; t++) {
    //     objective -= reg_ut * std::pow(unaries_noise[t] - unaries[t], 2.);
    // }

    std::cout << "NCE objective: " << objective << std::endl;


    // free up used space
    delete [] grad_unaries;
    delete [] grad_W_rep;
    delete [] grad_W_att;
    delete [] grad_n_logz;

    delete [] best_unaries;
    delete [] best_W_rep;
    delete [] best_W_att;
    delete [] best_n_logz;

    delete [] sample_lengths;
    delete [] labels;
    delete [] data;

    t = clock() - t;
    // std::cout << "It took me "  << (((float)t)/CLOCKS_PER_SEC) << "seconds" << std::endl;
}

void flm_sample(int32_t n_samples, py::array_t<int32_t>  samples, py::array_t<int32_t> given, py::array_t<int32_t> excluded, double n_logz,  py::array_t<double> utilities, py::array_t<double> W_rep, py::array_t<double> W_att) {
    np_wrap<int32_t> wrap_given(given);
    np_wrap<int32_t> wrap_excluded(excluded);
    np_wrap<int32_t> wrap_samples(samples);
    np_wrap<double> wrap_utilities(utilities);
    np_wrap<double> wrap_weights(W_rep);
    np_wrap<double> wrap_W_rep(W_rep);
    np_wrap<double> wrap_W_att(W_att);

    // double value = n_logz;

    size_t n = wrap_weights.buf.shape[0];
    size_t n_elements = wrap_samples.buf.shape[1];
    size_t dim = wrap_weights.buf.shape[1];
    size_t dim_rep = wrap_W_rep.buf.shape[1];
    size_t dim_att = wrap_W_att.buf.shape[1];

    size_t burn_in = 500;
    size_t skip = 100;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);
    std::uniform_int_distribution<> rnd_index(0, n_elements - 1);

    int indicator_vector[n_elements];
    int set[n_elements];  // set corresponding to the indicator vector
    size_t n_set;
    double v1;
    double v2;
    double gain;
    double p_add;

    int32_t *s_ptr = (int32_t*) samples.request().ptr;

    for (size_t idx = 0; idx < n_elements; idx++) {
        if (wrap_given[idx]) {
            indicator_vector[idx] = 1;
        }
        if (wrap_excluded[idx]) {
            indicator_vector[idx] = 0;
        }
    }

    size_t el;
    for (size_t sample_idx = 0; sample_idx < n_samples; sample_idx++) {
        for (size_t outer = 0; outer < burn_in; outer++) {
            for (size_t t_el = 0; t_el < n_elements; t_el++) {
                el = rnd_index(gen);
                if (wrap_given[el] || wrap_excluded[el]) {
                    continue;
                }

                indicator_vector[el] = 1;
                n_set = indicator_vector_to_set(n_elements, indicator_vector, set);
                v1 = _flm_obj(n_set, set, n, dim_rep, dim_att, n_logz, &wrap_utilities[0], &wrap_W_rep[0], &wrap_W_att[0]);

                indicator_vector[el] = 0;
                n_set = indicator_vector_to_set(n_elements, indicator_vector, set);
                v2 = _flm_obj(n_set, set, n, dim_rep, dim_att, n_logz, &wrap_utilities[0], &wrap_W_rep[0], &wrap_W_att[0]);

                gain = v1 - v2;
                p_add = sigmoid(gain);
                if (dis(gen) < p_add) {
                    indicator_vector[el] = 1;
                }
                else {
                    indicator_vector[el] = 0;
                }
            }
        }

        // copy sample
        for (size_t idx = 0; idx < n_elements; idx++) {
            // wrap_samples[sample_idx, idx] = indicator_vector[idx];
            s_ptr[sample_idx * n_elements + idx] = indicator_vector[idx];
        }
    }
}





PYBIND11_PLUGIN(_flm) {
    py::module m("_flm", "Routines for the FLM model");
    // m.def("grid_solve", &grid_solve,
    //       "Solve the mean-cut discretization algorithm");
    m.def("flm_obj", &flm_obj,
          "Compute the flm objective");
    m.def("flm_grad", &flm_grad,
          "Compute the gradient of the objective");
    m.def("flm_sample", &flm_sample,
          "Sample from the model");
    m.def("flm_train", &flm_train,
          "Train an FLM model");
    return m.ptr();
}

