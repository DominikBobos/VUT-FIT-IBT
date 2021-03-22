#!/usr/bin/env python
import numpy as np
import scipy.linalg as spl
import pytel.utils as utils
import h5py

def uppertri_indices(dim, isdiag=False):
    """ [utr utc]=uppertri_indices(D, isdiag) returns row and column indices
    into upper triangular part of DxD matrices. Indices go in zigzag feshinon
    starting by diagonal. For convenient encoding of diagonal matrices, 1:D
    ranges are returned for both outputs utr and utc when ISDIAG is true.
    """
    if isdiag:
        utr = np.arange(dim)
        utc = np.arange(dim)
    else:
        utr = np.hstack([np.arange(ii)     for ii in range(dim,0,-1)])
        utc = np.hstack([np.arange(ii,dim) for ii in range(dim)])
    return utr, utc

def uppertri_from_sym(covs_full, utr, utc):
    """ covs_ut2d = uppertri_from_sym(covs_full) reformat full symmetric matrices
    stored in 3rd dimension of 3D matrix into vectorized upper triangual
    matrices efficiently stored in columns of 2D matrix
    """
    dim, dim, n_mix = covs_full.shape

    covs_ut2d = np.zeros(((dim**2+dim)//2, n_mix), dtype=covs_full.dtype)
    for ii in range(n_mix):
        covs_ut2d[:, ii] = covs[:,:,ii][(utr, utc)]
    return covs_ut2d

def uppertri_to_sym(covs_ut2d, utr, utc):
    """ covs = uppertri_to_sym(covs_ut2d) reformat vectorized upper triangual
    matrices efficiently stored in columns of 2D matrix into full symmetric
    matrices stored in 3rd dimension of 3D matrix
    """
    (ut_dim, n_mix) = covs_ut2d.shape
    dim = (np.sqrt(1 + 8 * ut_dim)- 1) // 2

    covs_full = np.zeros((dim, dim, n_mix), dtype=covs_ut2d.dtype)
    for ii in range(n_mix):
        covs_full[:,:,ii][(utr, utc)] = covs_ut2d[:,ii]
        covs_full[:,:,ii][(utc, utr)] = covs_ut2d[:,ii]
    return covs_full

def uppertri1d_from_sym(cov_full, utr, utc):
    return cov_full[(utr, utc)]

def uppertri1d_to_sym(covs_ut1d, utr, utc):
    return uppertri_to_sym(np.array(covs_ut1d)[:,None], utr, utc)[:,:,0]


def gmm_init_1g(dim, is_full_cov=False):
  """ [weights, means, covs] = gmm_init(dim, is_full_cov) initializes weights,
  means and covs to such values so that the following call gmm_reestimate
  simply esimates single Gaussian model.
  """
  weights=np.zeros(1)
  means = np.zeros((1,dim))
  if is_full_cov:
    covs = np.c_[np.ones((1,dim)), np.zeros((1,(dim**2-dim)//2))];
  else:
    covs = np.ones((1,dim))
  return weights, means, covs


def gmm_init_from_labeled_data(data, class_ids, is_full_cov=False, chunk_size=102400):
  import scipy.sparse
  import numexpr as ne, sys
  posters = scipy.sparse.csr_matrix((np.ones_like(data[:,0]), np.c_[:len(data), class_ids].T))
  utr, utc = uppertri_indices(data.shape[1], not is_full_cov)
  N = np.array(posters.sum(axis=0)).ravel()
  F = posters.T.dot(data)
  S = 0.0
  # Do 2nd order statistics in chunks to be more memory efficient
  for ii in np.arange(len(data)//chunk_size+1):
    print("\r", ii, '/', len(data) // chunk_size+1)
    sys.stdout.flush()
    sl = slice(ii*chunk_size, min(len(data), (ii+1)*chunk_size))
    S += posters[sl].T.dot(data[sl,utr] * data[sl,utc])
  return gmm_update(N, F, S)


#def inv_posdef_and_logdet(M):
#    U = np.linalg.cholesky(M)
#    logdet = 2*np.sum(np.log(np.diagonal(U)))
#    #invU = spl.solve_triangular(U,np.identity(M.shape[0],M.dtype), lower=True)
#    #invM = np.dot(invU.T,invU)
#    invM = spl.solve(M, np.identity(M.shape[0],M.dtype), sym_pos=True)
#    return invM, logdet

def gmm_eval_prep(weights, means, covs):
    n_mix, dim = means.shape
    GMM = dict()
    is_full_cov = covs.shape[1] != dim
    GMM['utr'], GMM['utc'] = uppertri_indices(dim, not is_full_cov)

    if is_full_cov:
        GMM['gconsts'] = np.zeros(n_mix)
        GMM['invCovs'] = np.zeros_like(covs)
        GMM['invCovMeans']=np.zeros_like(means)
        for ii in range(n_mix):
            uppertri1d_to_sym(covs[ii], GMM['utr'], GMM['utc'])
            invC, logdetC = utils.inv_posdef_and_logdet(uppertri1d_to_sym(covs[ii], GMM['utr'], GMM['utc']))

            #log of Gauss. dist. normalizer + log weight + mu' invCovs mu
            invCovMean = invC.dot(means[ii])
            GMM['gconsts'][ii] = np.log(weights[ii]) - 0.5 * (logdetC + means[ii].dot(invCovMean) + dim * np.log(2.0*np.pi))
            GMM['invCovMeans'][ii] = invCovMean

            #Iverse covariance matrices are stored in columns of 2D matrix as vectorized upper triangual parts ...
            GMM['invCovs'][ii] = uppertri1d_from_sym(invC, GMM['utr'], GMM['utc']);
        # ... with elements above the diagonal multiply by 2
        GMM['invCovs'][:,dim:] *= 2.0
    else: #for diagonal
        GMM['invCovs']  = 1 / covs;
        GMM['gconsts']  = np.log(weights) - 0.5 * (np.sum(np.log(covs) + means**2 * GMM['invCovs'], axis=1) + dim * np.log(2.0*np.pi))
        GMM['invCovMeans'] = GMM['invCovs'] * means

    # for weight = 0, prepare GMM for uninitialized model with single gaussian
    if len(weights) == 1 and weights[0] == 0:
        GMM['invCovs']     = np.zeros_like(GMM['invCovs'])
        GMM['invCovMeans'] = np.zeros_like(GMM['invCovMeans'])
        GMM['gconsts']     = np.ones(1)
    return GMM

def gmm_eval(data, GMM, return_accums=0):
    """ llh = GMM_EVAL(d,GMM) returns vector of log-likelihoods evaluated for each
    frame of dimXn_samples data matrix using GMM object. GMM object must be
    initialized with GMM_EVAL_PREP function.

    [llh N F] = GMM_EVAL(d,GMM,1) also returns accumulators with zero, first order statistic.

    [llh N F S] = GMM_EVAL(d,GMM,2) also returns accumulators with second order statistic.
    For full covariance model second order statiscics, only the vectorized upper
    triangual parts are stored in columns of 2D matrix (similarly to GMM.invCovs).
    """
    # quadratic expansion of data
    data_sqr = data[:, GMM['utr']] * data[:, GMM['utc']]  # quadratic expansion of the data
    #computate of log-likelihoods for each frame and all Gaussian components
    gamma = -0.5 * data_sqr.dot(GMM['invCovs'].T) + data.dot(GMM['invCovMeans'].T) + GMM['gconsts']
    llh = utils.logsumexp(gamma, axis=1)

    if return_accums == 0:
        return llh

    gamma = np.exp(gamma.T - llh)
    N = gamma.sum(axis=1)
    F = gamma.dot(data)
    if return_accums == 1:
        return llh, N, F

    S = gamma.dot(data_sqr)
    return llh, N, F, S


def gmm_2model_eval(align_data, GMM, eval_data, eval_exp, return_accums=0):
    """ llh = GMM_EVAL(d,GMM) returns vector of log-likelihoods evaluated for each
    frame of dimXn_samples data matrix using GMM object. GMM object must be
    initialized with GMM_EVAL_PREP function.

    [llh N F] = GMM_EVAL(d,GMM,1) also returns accumulators with zero, first order statistic.

    [llh N F S] = GMM_EVAL(d,GMM,2) also returns accumulators with second order statistic.
    For full covariance model second order statiscics, only the vectorized upper
    triangual parts are stored in columns of 2D matrix (similarly to GMM.invCovs).
    """

    # quadratic expansion of data
    data_sqr  = align_data[:, GMM['utr']] * align_data[:, GMM['utc']]  # quadratic expansion of the align_data

    #computate of log-likelihoods for each frame and all Gaussian components
    gamma = -0.5 * data_sqr.dot(GMM['invCovs'].T) + align_data.dot(GMM['invCovMeans'].T) + GMM['gconsts']
    llh = utils.logsumexp(gamma, axis=1)

    if return_accums == 0:
        return llh

    gamma = np.exp(gamma.T - llh)

    N = gamma.sum(axis=1)
    F = gamma.dot(eval_data)

    if return_accums == 1:
        return llh, N, F

    data_sqr2 = eval_data[:, eval_exp['utr']] * eval_data[:, eval_exp['utc']]  # quadratic expansion of the align_data
    S = gamma.dot(data_sqr2)

    return llh, N, F, S


def gmm_floor_covs(weights, means, covs, floor_const):  
  # floor by fraction of average covariance
  n_mix, dim = means.shape
  is_diag_cov = covs.shape[1] == dim
  vFloor=covs.T.dot(weights / weights.sum()) * floor_const
  if is_diag_cov:
    print('Number of floored variances:', np.sum(covs < vFloor))
    covs = np.maximum(covs, vFloor)
  else:
    utr, utc = uppertri_indices(dim)
    r = spl.cholesky(uppertri1d_to_sym(vFloor, utr, utc))
    inv_r = spl.inv(r)

    for ii in range(n_mix):
      d, v = np.linalg.eigh(inv_r.T.dot(uppertri1d_to_sym(covs[ii], utr, utc).dot(inv_r)))
      mask = d < 1.0
      print('Number of floored variances:', mask.sum(), 'for mixture:', ii)
      d[mask] = 1.0
      covs[ii] = uppertri1d_from_sym(r.T.dot(v * d).dot(v.T).dot(r), utr, utc)
  return covs


def gmm_split(weights, means, covs, perturb_factor=1.0):
    n_mix, dim = means.shape
    if_full_cov = covs.shape[1] != dim

    if if_full_cov:
        utr, utc = uppertri_indices(dim)
        means = np.r_[means, means]
        for ii in range(n_mix):
            d, v = np.linalg.eig(uppertri1d_to_sym(covs[ii], utr, utc))
            i_max = np.argmax(d)
            perturb = perturb_factor * np.sqrt(d[i_max]) * v[:,i_max]
            means[ii+n_mix] += perturb
            means[ii]       -= perturb
    else:
        max_var_coo = (np.arange(n_mix), covs.argmax(axis=1))
        perturb = np.zeros_like(means)
        perturb[max_var_coo] = perturb_factor * np.sqrt(covs[max_var_coo])
        means = np.r_[means - perturb, means + perturb]
    covs  = np.r_[covs, covs]
    weights = np.r_[weights, weights]*0.5
    return weights, means, covs

def gmm_update(N,F,S):
    """ weights means covs = gmm_update(N,F,S) return GMM parameters, which are
    updated from accumulators
    """
    dim = F.shape[1]
    is_diag_cov = S.shape[1] == dim
    utr, utc = uppertri_indices(dim, is_diag_cov)
    sumN    = N.sum()
    weights = N / sumN
    means   = F / N[:,np.newaxis]
    covs    = S / N[:,np.newaxis] - means[:,utr] * means[:,utc]
    return weights, means, covs

#def logsumexp(x, axis=0):
#    xmax = x.max(axis)
#    ex=np.exp(x - np.expand_dims(xmax, axis))
#    x = xmax + np.log(np.sum(ex, axis))
#    not_finite = ~np.isfinite(xmax)
#    x[not_finite] = xmax[not_finite]
#    return x

def gmm_save(file_name, weights, means, covs):
    with h5py.File(file_name, 'w') as f:
        f.create_dataset('weights',data=weights)
        f.create_dataset('means', data=means)
        f.create_dataset('covs',   data=covs)

def gmm_load(file_name):
    with h5py.File(file_name, 'r') as f:
        return np.array(f['weights']), np.array(f['means']), np.array(f['covs'])

def save_acc(file_name, L, N, F, S=0.0):
    with h5py.File(file_name, 'w') as f:
        f.create_dataset('L',data=L)
        f.create_dataset('N',data=N)
        f.create_dataset('F',data=F)
        f.create_dataset('S',data=S)

def load_acc(file_name):
    with h5py.File(file_name, 'r') as f:
        return np.array(f['L']), np.array(f['N']), np.array(f['F']), np.array(f['S'])

def load_stats(file_name):
    with h5py.File(file_name, 'r') as f:
        return np.array(f['N']), np.array(f['F'])

def gmm_llhs(data, GMM):
    """ llh = GMM_EVAL(d,GMM) returns vector of log-likelihoods evaluated for each
    frame of dimXn_samples data matrix using GMM object. GMM object must be
    initialized with GMM_EVAL_PREP function.

    [llh N F] = GMM_EVAL(d,GMM,1) also returns accumulators with zero, first order statistic.

    [llh N F S] = GMM_EVAL(d,GMM,2) also returns accumulators with second order statistic.
    For full covariance model second order statiscics, only the vectorized upper
    triangual parts are stored in columns of 2D matrix (similarly to GMM.invCovs).
    """
    # quadratic expansion of data
    data_sqr = data[:, GMM['utr']] * data[:, GMM['utc']]  # quadratic expansion of the data

    #computate of log-likelihoods for each frame and all Gaussian components
    gamma = -0.5 * data_sqr.dot(GMM['invCovs'].T) + data.dot(GMM['invCovMeans'].T) + GMM['gconsts']

    return gamma

def norm_suff_stats_by_gmm(Ns, Fs, means, covs):
    """
    Parameters
    ----------
    N : zero order statistisc for R segments and GMM with C components, shape (R, C)
    F : first order statistisc for D-dimensional features, shape (R, C, D)
    means: GMM means, shape (C, D)
    covs: GMM covariance matrices, which are:
      - either diagonal, shape (C, D) or
      - or full, shape (C, (D**2+D)/2, contains upper triangular elemenst, which can be
        expanded using gmm.uppertri_indices()
    Returns
    ----------
    Fn: mean and variance normalized suffucient statistics, shape (R, C, D)
    """
    #F = F.reshape(N.shape[0], N.shape[1], -1)
    Fn = np.atleast_3d(Fs) - np.atleast_2d(Ns)[:,:,np.newaxis] * means[:,:]
    if covs.shape[1] == Fn.shape[1]:
        Fn /= covs
    else:
        utr, utc = gmm.uppertri_indices(means.shape[1])
        for ii in range(len(covs)):
            w,v=np.linalg.eigh(gmm.uppertri1d_to_sym(covs[ii], utr, utc))
            Fn[:,ii,:] = Fn[:,ii,:].dot(np.dot(v*(w**-0.5), v.T))
    return Fn




def kmeanspp(data, ncenters, power=1):
  # kmeanspp clusters data set using k-means++ algorithm, where
  # in each iteration, a data point is selected as a new centroid with probability
  # proportional to the squared distance from the currently closest centroid
  #
  # inputs:
  #   data - matrix with data points stored in rows (seems to work faster for data on f-order)
  #   ncenters - desired number of clusters
  # outputs:
  #   cluster_ids - vector of integer cluster labels

  import random, sys
  data_norm2 = np.sum(data**2, axis=1)
  cluster_ids = np.zeros(len(data), dtype=int)
  c = random.choice(data) # randomly select first centroid
  min_dist2 = c.dot(c) + data_norm2 - data.dot(2*c)
  for ii in range(1,ncenters):
    # probabilisticly select new centroid according to the k-means++ algorithm
    csd = np.cumsum(min_dist2**power)
    c = data[np.sum(csd <= np.random.rand()*csd[-1])]
    # update distances to the cosest centroid by considering distances to the new centroid
    c_dist2 = c.dot(c) + data_norm2 - data.dot(2*c)
    c_is_closer = np.nonzero(c_dist2 < min_dist2)[0]
    min_dist2[c_is_closer] = c_dist2[c_is_closer]
    cluster_ids[c_is_closer] = ii
    print("\r", ii)
    sys.stdout.flush()
  return cluster_ids


def test_kmeanspp():
  import pylab
  cols = [x+y for x in 'rgbmcyk' for y in '+x|_']
  nclusters = 28
  data=np.random.randn(10000,2)
  cid = kmeanspp(data, nclusters)
  for ii in range(nclusters):
    pylab.plot(data[cid==ii,0],data[cid==ii,1], cols[ii])
  return data, cid


if(__name__ == "__main__"):
    pass
