#!/usr/bin/env python

import os, errno, sys, signal
import numpy as np
import scipy.linalg as spl
import numexpr as ne
import logging
import itertools

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise
def mkdir_subdirs(seglist, out_dir='./'):
    """
    Create sub-directories relative to the out_dir according to seglist
    """
    for dir in set(map(os.path.dirname, seglist)):
        mkdir_p(out_dir+'/'+dir)

def softmax(loglhs, axis=0):
    return np.exp(loglhs - np.expand_dims(logsumexp(loglhs, axis=axis), axis=axis))

def softmax_numexpr(loglhs, axis=0):
    return np.exp(loglhs - np.expand_dims(logsumexp_numexpr(loglhs, axis=axis), axis=axis))


def logit(a):
    return np.log(a / (1 - a))


def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def sigmoid_numexpr(a):
    return ne.evaluate("1 / (1 + exp(-a))")


def probit(a):
    from scipy.special import erfinv
    return np.sqrt(2) * erfinv(2 * a - 1)


def invprobit(a):
    from scipy.special import erf
    return 0.5 * (1 + erf(a / np.sqrt(2)))


def get_class_means_between_and_within_covs(data, class_ids):
    from scipy.sparse import csr_matrix
    posters = csr_matrix((np.ones_like(data[:,0]), np.c_[:len(data), class_ids].T)).T
    N = np.array(posters.sum(axis=1))
    means  = np.array(posters.dot(data)) / N
    AC = (means - data.mean(axis=0)) * np.sqrt(N)
    AC = AC.T.dot(AC) / len(data)
    WC = np.cov(data.T,  bias=1) - AC
    return means, AC, WC


def logsumexp(x, axis=0):
    xmax = x.max(axis)
    x = xmax + np.log(np.sum(np.exp(x - np.expand_dims(xmax, axis)), axis))
    not_finite = ~np.isfinite(xmax)
    x[not_finite] = xmax[not_finite]
    return x


def logsumexp_numexpr(x, axis=0):
    xmax = x.max(axis)
    xmax_e = np.expand_dims(xmax, axis)
    t = ne.evaluate("sum(exp(x - xmax_e),axis=%d)" % axis)
    x =  ne.evaluate("xmax + log(t)")
    not_finite = ~np.isfinite(xmax)
    x[not_finite] = xmax_e[not_finite]
    return x


def inv_posdef_and_logdet(A):
    L = np.linalg.cholesky(A)
    logdet = 2*np.sum(np.log(np.diagonal(L)))
    #invL = spl.solve_triangular(L, np.identity(len(A),M.dtype), lower=True)
    #invA = np.dot(invL.T, invL)
    invA = spl.solve(A, np.identity(len(A), A.dtype), sym_pos=True)
    return invA, logdet


def gemm(a, b, out=None, alpha=1.0, beta=0.0, dtype=None, order='C'):
    import scipy.linalg as sp
    gemm = sp.get_blas_funcs('gemm', (a,b) if not dtype else (), dtype)
    a, trans_a = (a, False) if np.isfortran(a) else (a.T, True)
    b, trans_b = (b, False) if np.isfortran(b) else (b.T, True)
    if out is not None:
        order, out = ('F', out) if np.isfortran(out) else ('C', out.T)
    if order == 'C':
        a, b, trans_a, trans_b = b, a, not trans_b, not trans_a
    res = gemm(alpha=alpha, a=a, b=b, trans_a=trans_a, trans_b=trans_b, beta=beta, c=out, overwrite_c=True)
    if out is not None and res is not out:
        raise ValueError("output array is not acceptable")
    return res.T if order == 'C' else res


def profileit(func):
    """
    Decorator (function wrapper) that profiles a single function

    @profileit
    def func1(...)
    # do something
    pass
    """
    import cProfile, pstats
    def wrapper(*args, **kwargs):
        datafn = func.__name__ + ".profile"
        prof = cProfile.Profile()
        retval = prof.runcall(func, *args, **kwargs)
        prof.dump_stats(datafn)
        stats = pstats.Stats(datafn)
        stats.sort_stats("time")  # "cumulative"
        stats.print_stats(100)
        stats.print_callers()
        return retval
    return wrapper

def sound(y, fs=8000, bits=16):
    import subprocess, tempfile, os, scipy.io.wavfile
    fid, tfn = tempfile.mkstemp(suffix='.wav')
    os.close(fid)
    scipy.io.wavfile.write(tfn, fs, y.astype("int"+str(bits)))
    subprocess.call(["mplayer", tfn, "2>/dev/null"])
    subprocess.call(["rm", "-f",  tfn])

def soundsc(y, fs=8000, bits=16):
    x = y.astype(float)-y.min()
    x = (x/x.max()-0.5)*2**bits
    sound(x, fs, bits)

def get_rows_with_context2(a, start, end, left_ctx, right_ctx):
    """
    Returns a[start-left_ctx:end+right_ctx] unless rows before the first or
    after the last row are asked for. In such case, the first and/or the last
    row(s) are repeated as desired (i.e. to return array with 
    end-start+left_ctx+right_ctx rows)
    """
    X = a[max(0, start-left_ctx):end+right_ctx]
    return np.r_[np.repeat(X[[0]], max(0, left_ctx-start), axis=0), X, np.repeat(X[[-1]], max(0, end+right_ctx-len(a)), axis=0)]

def get_rows_with_repeats(a, start, end):
    """
    Returns a[start:end]. However, if start is negative or when end points
    behind last row of a, then a is coppied and the first or/and the last row(s)
    are repeated as required to always return end-start frames.
    """
    s, e = max(0, start), min(len(a), end)
    a = a[s:e]
    if start != s or end != e: # repeat first or/and last frame as required
        a = np.r_[np.repeat(a[[0]], s-start, axis=0), a, np.repeat(a[[-1]], end-e, axis=0)]
    return a

def row(v):
    return v.reshape((1, v.size))


def start_ends2frame_labels(start_ends, frame_rate=100.0):
    """ 
    Converts a list of start-end pairs to a bit-mask
    Useful for VAD
    """
    starts, ends = np.rint(start_ends*frame_rate).astype(int).T
    dur_mat = np.c_[starts - np.r_[0, ends[:-1]], ends-starts]
    return np.repeat(np.c_[np.zeros_like(starts, dtype=bool), np.ones_like(starts, dtype=bool)], dur_mat.flat)

def frame_labels2start_ends(speech_frames, frame_rate=100.0):
    decesions = np.r_[False, speech_frames, False]
    return np.nonzero(decesions[1:] != decesions[:-1])[0].reshape(-1,2) / frame_rate


class Logger:
    def __init__(self, filename=None):
        self.filename = filename
        if len(logging.getLogger().handlers) > 0:
            l = logging.getLogger()
            [l.removeHandler(h) for h in l.handlers]
        if self.filename is not None:
            mkdir_p(os.path.dirname(self.filename))
            logging.basicConfig(filename=filename, format='%(asctime)s: %(message)s', level=logging.INFO)
        else:
            logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger()
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
    def info(self, *arg):
        self.logger.info(*arg)
    def error(self, *arg):
        self.logger.error(*arg)

class AttrDict(dict):
    """ If you inherit from this class, you'll get an object
    with accessors based on named options
    ex: a=AttrDict(spam=2,eggs="bowl")
    a.spam is 2 and a.eggs is bowl
    If the desired key doesn't exist - return None
    """
    def __init__(self,*args,**kw):
        dict.__init__(self, *args, **kw)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            return None

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]


def segment_buffer_to_minibatch_iter(segment_buffer, minibatch_size=512):
    # make sure that all data matrices in each tupple are of the same length
    seg_lengths=zip(*[[len(e) for e in t] for t in segment_buffer])
    assert seg_lengths[1:]==seg_lengths[:-1]
    buffers = [np.concatenate(buf) for buf in zip(*segment_buffer)]
    shuffle = np.random.permutation(len(buffers[0]))
    n_minibatches = len(shuffle) // minibatch_size
    if n_minibatches==0: n_minibatches=1
    for minibatch in zip(*[np.array_split(buf.take(shuffle, axis=0), n_minibatches) for buf in buffers]):
        yield minibatch


def isplit_every(n, it, drop_uncomplete_last=True):
    it = iter(it)
    while True:
        buffer = list(itertools.islice(it, n))
        if not buffer or len(buffer) < n and drop_uncomplete_last:
            break
        yield buffer


def isplit_every_n(n, it):
    """
    Lazy split of iterator to subset of n items
    # Arguments
      n: number of items in split
      it: iterator to split
    # Returns
      generator of iterators
    # Example
    ```python
    #create iterator over infinit iterator, splitted by 10 items
    it = keras.utils.io_utils.isplit_every(10, itertools.count())
    ```
    """
    return itertools.takewhile(bool, (itertools.islice(it, n) for _ in itertools.count(0)))


def onehot_to_full(one_hot_lables, nlabels):
    y = np.zeros((len(one_hot_lables), nlabels))
    y[np.arange(len(one_hot_lables)), one_hot_lables] = 1
    return y

if(__name__=="__main__"):
    pass
