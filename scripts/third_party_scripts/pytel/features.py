#!/usr/bin/env python
import numpy as np
import scipy as sp
import scipy.fftpack

def rolling_window(a, window, shift=1):
    shape = a.shape[:-1] + ((a.shape[-1] - window) // shift + 1, window)
    strides = a.strides[:-1] + (a.strides[-1]*shift,a.strides[-1])
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def framing(a, window, shift=1):
    shape = ((a.shape[0] - window) // shift + 1, window) + a.shape[1:]
    strides = (a.strides[0]*shift,a.strides[0]) + a.strides[1:]
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


# Mel and inverse Mel scale warping functions
def mel_inv(x):
    return (np.exp(x/1127.)-1.)*700.


def mel(x):
    return 1127.*np.log(1. + x/700.)


#def mel_filter_bank_old(nfft, nbands, fs, fstart=0, fend=None, warp_fnc=mel, inv_warp_fnc=mel_inv):
#    """Returns mel filterbank as an array (nfft/2+1 x nbands)
#    nfft   - number of samples for FFT computation
#    nbands - number of filter bank bands
#    fs     - sampling frequency (Hz)
#    fstart - frequency (Hz) where the first filter strats
#    fend   - frequency (Hz) where the last  filter ends (default fs/2)
#    """
#    if not fend:
#      fend = 0.5 * fs
#
#    cbin = np.round(inv_warp_fnc(np.linspace(warp_fnc(fstart), warp_fnc(fend), nbands + 2)) / fs * nfft)
#    mfb = np.zeros((nfft // 2 + 1, nbands))
#    for ii in range(nbands):
#        mfb[cbin[ii]:  cbin[ii+1]+1, ii] = np.linspace(0., 1., cbin[ii+1] - cbin[ii]   + 1)
#        mfb[cbin[ii+1]:cbin[ii+2]+1, ii] = np.linspace(1., 0., cbin[ii+2] - cbin[ii+1] + 1)
#    return mfb
#
#
#def mfcc(x, WINDOW, NOVERLAP, NFFT, Fs, NBANKS, NCEPS, Fstart=32, Fend=None, power=2, energy=False):
#    """MFCC Mel Frequency Cepstral Coefficients
#    Returns NCEPS-by-M matrix of MFCC coeficients extracted form signal x,
#    where M is the number of extracted frames, which can be computed as
#    floor((length(S)-NOVERLAP)/(WINDOW-NOVERLAP)). Remaining parameters
#    have the following meaning:
#
#    NFFT          - number of frequency points used to calculate the discrete
#                    Fourier transforms
#    Fs            - sampling frequency [Hz]
#    WINDOW        - window lentgth for frame (in samples)
#    NOVERLAP      - overlapping between frames (in samples)
#    NBANKS        - numer of mel filter bank bands
#    NCEPS         - number of cepstral coefficients - the output dimensionality
#    """
#    # Initialize matrices representing Mel filterbank and DCT
#    mfb = mel_filter_bank(NFFT, NBANKS, Fs, Fstart, Fend)
#    S = np.abs(spectrogram(x, WINDOW, NOVERLAP, NFFT))
#    out = np.log(np.power(S, power).dot(mfb)).dot(dct_basis(NCEPS,NBANKS).T)
#    if energy:
#        out[:,0] = np.log(np.sum(S**2, axis=1))
#    return out


def spectrogram(x, window, noverlap=None, nfft=None, ZMEANSOURCE=False, PREEMCOEF=None):
    if np.isscalar(window): window = np.hamming(window)
    if noverlap is None:    noverlap = window.size // 2
    if nfft     is None:    nfft     = window.size
    x = framing(x, window.size, window.size-noverlap)
    if ZMEANSOURCE:
        x = x - x.mean(axis=1)[:,np.newaxis]
    if PREEMCOEF is not None:
        x = preemphasis(x, PREEMCOEF)
    x = scipy.fftpack.fft(x*window, nfft)
    return x[:,:x.shape[1]//2+1]


def specgram2sig(sg, window, noverlap=None, nfft=None):
    """Reconstruct signal from spectrogram SG. WINDOW and NOVERLAP should be the
    same as the parameters passed to the function specgram while computating SG.

    Example:
    load handel
    sg = specgram(y, 256, 8000, hanning(256), 128);
    x = specgram2sig(sg, 256, 128);
    The signal x should be almost identical scaled version of y.
  
    See also spectrogram.
    """
    window = np.array(window).ravel()
    winlen = len(window)
    if winlen == 1:
        winlen = window;
        window = 1.0;
    if noverlap is None: noverlap = winlen // 2
    if nfft     is None: nfft     = winlen
    sig=np.zeros(len(sg)*(winlen-noverlap)+noverlap)
    for i, spec in enumerate(sg):
        frmSig = np.real(scipy.fftpack.ifft(np.r_[spec, np.conjugate(spec[(nfft-1)//2:0:-1])], nfft))
        frmStrt = i * (winlen-noverlap);
        sig[frmStrt:frmStrt+winlen] = sig[frmStrt:frmStrt+winlen] + frmSig[:winlen] * window;
    return sig


def spectrogram_old(x, WINDOW, NOVERLAP=None, NFFT=None):
    if np.isscalar(WINDOW):
        WINDOW = np.hamming(WINDOW)
    if NOVERLAP is None:
        NOVERLAP = WINDOW.size // 2
    S = scipy.fftpack.fft(framing(x, WINDOW.size, WINDOW.size-NOVERLAP)*WINDOW, NFFT)
    return S[:,:S.shape[1]//2+1]


def dct_basis(nbasis, length):
    # the same DCT as in matlab
    return scipy.fftpack.idct(np.eye(nbasis, length), norm='ortho')


def preemphasis(x, coef=0.97):
    return x - np.c_[x[..., :1], x[..., :-1]] * coef


def mel_fbank_mx(winlen_nfft, fs, NUMCHANS=20, LOFREQ=0.0, HIFREQ=None, warp_fn=mel, inv_warp_fn=mel_inv, htk_bug=True):
    """Returns mel filterbank as an array (NFFT/2+1 x NUMCHANS)
    winlen_nfft - Typically the window length as used in mfcc_htk() call. It is
                  used to determine number of samples for FFT computation (NFFT).
                  If positive, the value (window lenght) is rounded up to the
                  next higher power of two to obtain HTK-compatible NFFT.
                  If negative, NFFT is set to -winlen_nfft. In such case, the 
                  parameter nfft in mfcc_htk() call should be set likewise.
    fs          - sampling frequency (Hz, i.e. 1e7/SOURCERATE)
    NUMCHANS    - number of filter bank bands
    LOFREQ      - frequency (Hz) where the first filter starts
    HIFREQ      - frequency (Hz) where the last filter ends (default fs/2)
    warp_fn     - function for frequency warping and its inverse
    inv_warp_fn - inverse function to warp_fn
    """
    if not HIFREQ: HIFREQ = 0.5 * fs
    nfft = 2**int(np.ceil(np.log2(winlen_nfft))) if winlen_nfft > 0 else -int(winlen_nfft)

    fbin_mel = warp_fn(np.arange(nfft / 2 + 1, dtype=float) * fs / nfft)
    cbin_mel = np.linspace(warp_fn(LOFREQ), warp_fn(HIFREQ), NUMCHANS + 2)
    cind = np.floor(inv_warp_fn(cbin_mel) / fs * nfft).astype(int) + 1
    mfb = np.zeros((len(fbin_mel), NUMCHANS))
    for i in range(NUMCHANS):
        mfb[cind[i]  :cind[i+1], i] = (cbin_mel[i]  -fbin_mel[cind[i]  :cind[i+1]]) / (cbin_mel[i]  -cbin_mel[i+1])
        mfb[cind[i+1]:cind[i+2], i] = (cbin_mel[i+2]-fbin_mel[cind[i+1]:cind[i+2]]) / (cbin_mel[i+2]-cbin_mel[i+1])
    if LOFREQ > 0.0 and float(LOFREQ)/fs*nfft+0.5 > cind[0] and htk_bug: mfb[cind[0],:] = 0.0 # Just to be HTK compatible
    return mfb


def fbank_htk(x, window, noverlap, fbank_mx, nfft=None, _E=None,
            USEPOWER=False, RAWENERGY=True, PREEMCOEF=0.97, ZMEANSOURCE=False,
            ENORMALISE=True, ESCALE=0.1, SILFLOOR=50.0, USEHAMMING=True):
    """Mel log Mel-filter bank channel outputs
    Returns NUMCHANS-by-M matrix of log Mel-filter bank outputs extracted from
    signal x, where M is the number of extracted frames, which can be computed
    as floor((length(x)-noverlap)/(window-noverlap)). Remaining parameters
    have the following meaning:
    x         - input signal
    window    - frame window length (in samples, i.e. WINDOWSIZE/SOURCERATE) 
                or vector of window weights override default windowing function
                (see option USEHAMMING)
    noverlap  - overlapping between frames (in samples, i.e window-TARGETRATE/SOURCERATE)
    fbank_mx  - array with (Mel) filter bank (as returned by function mel_fbank_mx()).
                Note that this must be compatible with the parameter 'nfft'.
    nfft      - number of samples for FFT computation. By default, it is set in the
                HTK-compatible way to the window length rounded up to the next higher
                power of two.
    _E        - include energy as the "first" or the "last" coefficient of each
                feature vector. The possible values are: "first", "last", None.

    Remaining options have exactly the same meaning as in HTK.

    See also:
      mel_fbank_mx:
          to obtain the matrix for the parameter fbank_mx
      add_deriv: 
          for adding delta, double delta, ... coefficients
      add_dither:
          for adding dithering in HTK-like fashion
    """
    from time import time
    tm = time()
    if type(USEPOWER) == bool:
        USEPOWER += 1
    if np.isscalar(window):
        window = np.hamming(window) if USEHAMMING else np.ones(window)
    if nfft is None:
        nfft = 2**int(np.ceil(np.log2(window.size)))
    x = framing(x.astype("float"), window.size, window.size-noverlap).copy()
    if ZMEANSOURCE:
        x -= x.mean(axis=1)[:,np.newaxis]
    if _E is not None and RAWENERGY:
        energy = np.log((x**2).sum(axis=1))
    if PREEMCOEF is not None:
        x = preemphasis(x, PREEMCOEF)
    x *= window
    if _E is not None and not RAWENERGY:
        energy = np.log((x**2).sum(axis=1))
    #x = np.abs(scipy.fftpack.fft(x, nfft))
    #x = x[:,:x.shape[1]//2+1]
    x = np.fft.rfft(x, nfft)
    #x = np.abs(x)
    x = x.real**2 + x.imag**2
    if USEPOWER != 2:
        x **= 0.5 * USEPOWER
    x = np.log(np.maximum(1.0, np.dot(x, fbank_mx)))
    if _E is not None and ENORMALISE:
        energy = (energy - energy.max())       * ESCALE + 1.0
        min_val  = -np.log(10**(SILFLOOR/10.)) * ESCALE + 1.0
        energy[energy < min_val] = min_val

    return np.hstack(([energy[:,np.newaxis]] if _E == "first" else []) + [x] +
                     ([energy[:,np.newaxis]] if (_E in ["last", True])  else []))
     

def powerspectrum_htk(x, window, noverlap, nfft=None, _E=None,
             USEPOWER=False, RAWENERGY=True, PREEMCOEF=0.97, ZMEANSOURCE=False,
             ENORMALISE=True, ESCALE=0.1, SILFLOOR=50.0, USEHAMMING=True):
    """
    """
    from time import time
    tm = time()
    if type(USEPOWER) == bool:
        USEPOWER += 1
    if np.isscalar(window):
        window = np.hamming(window) if USEHAMMING else np.ones(window)
    if nfft is None:
        nfft = 2**int(np.ceil(np.log2(window.size)))
    x = framing(x.astype("float"), window.size, window.size-noverlap).copy()
    if ZMEANSOURCE:
        x -= x.mean(axis=1)[:,np.newaxis]
    if _E is not None and RAWENERGY:
        energy = np.log((x**2).sum(axis=1))
    if PREEMCOEF is not None:
        x = preemphasis(x, PREEMCOEF)
    x *= window
    if _E is not None and not RAWENERGY:
        energy = np.log((x**2).sum(axis=1))
    x = np.fft.rfft(x, nfft)
    x = x.real**2 + x.imag**2
    if USEPOWER != 2:
        x **= 0.5 * USEPOWER
    return np.hstack(([energy[:,np.newaxis]] if _E == "first" else []) + [x] +
                     ([energy[:,np.newaxis]] if (_E in ["last", True])  else []))


def mfcc_htk(x, window, noverlap, fbank_mx, nfft=None,
            _0="last", _E=None, NUMCEPS=12,
            USEPOWER=False, RAWENERGY=True, PREEMCOEF=0.97, CEPLIFTER=22.0, ZMEANSOURCE=False,
            ENORMALISE=True, ESCALE=0.1, SILFLOOR=50.0, USEHAMMING=True):
    """MFCC Mel Frequency Cepstral Coefficients
    Returns NUMCEPS-by-M matrix of MFCC coeficients extracted form signal x,
    where M is the number of extracted frames, which can be computed as
    floor((length(x)-noverlap)/(window-noverlap)). Remaining parameters
    have the following meaning:
    x         - input signal
    window    - frame window lentgth (in samples, i.e. WINDOWSIZE/SOURCERATE) 
                or vector of widow weights override default windowing function
                (see option USEHAMMING)
    noverlap  - overlapping between frames (in samples, i.e window-TARGETRATE/SOURCERATE)
    fbank_mx  - array with (Mel) filter bank (as returned by function mel_fbank_mx()).
                Note that this must be compatible with the parameter 'nfft'.
    nfft      - number of samples for FFT computation. By default, it is  set in the
                HTK-compatible way to the window length rounded up to the next higher
                pover of two.
    _0, _E    - include C0 or/and energy as the "first" or the "last" coefficient(s)
                of each feature vector. The possible values are: "first", "last", None.
                If both C0 and energy are used, energy will be the very first or the
                very last coefficient.

    Remaining options have exactly the same meaning as in HTK.

    See also:
      mel_fbank_mx:
          to obtain the matrix for the parameter fbank_mx
      add_deriv: 
          for adding delta, double delta, ... coefficients
      add_dither:
          for adding dithering in HTK-like fashion
    """

    dct_mx = dct_basis(NUMCEPS+1,fbank_mx.shape[1]).T
    dct_mx[:,0] = np.sqrt(2.0/fbank_mx.shape[1])
    if type(USEPOWER) == bool:
        USEPOWER += 1
    if np.isscalar(window):
        window = np.hamming(window) if USEHAMMING else np.ones(window)
    if nfft is None:
        nfft = 2**int(np.ceil(np.log2(window.size)))
    x = framing(x.astype("float"), window.size, window.size-noverlap).copy()
    if ZMEANSOURCE:
        x -= x.mean(axis=1)[:,np.newaxis]
    if _E is not None and RAWENERGY:
        energy = np.log((x**2).sum(axis=1))
    if PREEMCOEF is not None:
        x = preemphasis(x, PREEMCOEF)
    x *= window
    if _E is not None and not RAWENERGY:
        energy = np.log((x**2).sum(axis=1))
    #x = np.abs(scipy.fftpack.fft(x, nfft))
    #x = x[:,:x.shape[1]//2+1]
    x = np.abs(np.fft.rfft(x, nfft))
    x = np.log(np.maximum(1.0, (x**USEPOWER).dot(fbank_mx))).dot(dct_mx)
    if CEPLIFTER is not None and CEPLIFTER > 0:
        x *= 1.0 + 0.5 * CEPLIFTER * np.sin(np.pi * np.arange(NUMCEPS+1) / CEPLIFTER)
    if _E is not None and ENORMALISE:
        energy = (energy - energy.max())       * ESCALE + 1.0
        min_val  = -np.log(10**(SILFLOOR/10.)) * ESCALE + 1.0
        energy[energy < min_val] = min_val

    return np.hstack(([energy[:,np.newaxis]] if _E == "first" else []) +
                     ([x[:,:1]]              if _0 == "first" else []) +
                      [x[:,1:]] +
                     ([x[:,:1]]              if (_0 in ["last", True])  else []) +
                     ([energy[:,np.newaxis]] if (_E in ["last", True])  else []))


def add_white_noise_at_SNR(x, SNRdB=40.0):
    noise = np.random.randn(x.size)
    return x + noise * np.linalg.norm(x) / np.linalg.norm(noise) / 10.**(SNRdB/20.)


def add_white_noise_at_quantization_level(x, scale=0.25):
    """ Zero mean white gaussian noise with standard deviation q_step * scale is
    added to the input signal x, where q_step is quantization step calculated as
    the smallest difference bettween any tw0 samples in the signal.
    """
    q_step = np.min(np.diff(np.unique(x)))
    return x + np.random.randn(x.size) * q_step * scale


def add_dither(x, level=8):
    return x + level * (np.random.rand(*x.shape)*2-1) 


def povey_window(winlen):
    return np.power(0.5 - 0.5*np.cos(np.linspace(0,2*np.pi, winlen)), 0.85)


def add_dither_at_quantization_level(x, level=0.5):
    """ +/- q_step*level uniform distributed noise is added to the input signal x,
    where q_step is quantization step calculated as the smallest difference
    bettween any tw0 samples in the signal.
    """
    q_step = np.min(np.diff(np.unique(x)))
    return x + level * (np.random.rand(*x.shape)*q_step*2-1) 

    
def temporal_dct(fea_mx, window, ncoef, shift=1):
    """ Compute 2D DCT features on a window
    Input:
    fea_mx: frames x dim
    window: context in # of frames
    ncoef: # of coefficients to retain from temporal dct
    Output:
    features: frames x out_dim
    """
    out = dct_basis(ncoef, window).dot(framing(fea_mx, window, shift)).transpose(1,0,2)
    zigzag_idx = np.argsort(np.add.outer(np.linspace(0,1,out.shape[1]), np.linspace(0,1,out.shape[2])).flat)
    return out.reshape(out.shape[0], -1)[:,zigzag_idx]


def add_deriv(fea, winlens=(2,2)):
    """ Add derivatives to features (deltas, double deltas, triple_delas, ...)
    Input:
    fea: feature array (frames in rows)
    winlens: tuple with window lengths for deltas, double deltas, ... default is (2,2)
    Output:
    feature array augmented with derivatives
    """
    import scipy.signal
    fea_list=[fea]
    for wlen in winlens:
        dfilter = -np.arange(-wlen,wlen+1, dtype=fea_list[0].dtype)
        dfilter = dfilter / dfilter.dot(dfilter)
        fea = np.r_[fea[[0]].repeat(wlen,0), fea, fea[[-1]].repeat(wlen,0)]
        fea = scipy.signal.lfilter(dfilter, 1, fea, 0)[2*wlen:].astype(fea_list[0].dtype)
        fea_list.append(fea)
    return np.hstack(fea_list)


def cmvn_floating(m,LC,RC, unbiased=False):
    """Mean and variance normalization over a floating window.
    m is the feature matrix (nframes x dim)
    LC, RC is the nmber of frames to the left and right defining the floating
    window around the current frame 
    """
    nframes, dim = m.shape
    LC = min(LC,nframes-1)
    RC = min(RC,nframes-1)
    n = (np.r_[np.arange(RC+1,nframes), np.ones(RC+1)*nframes] - np.r_[np.zeros(LC), np.arange(nframes-LC)])[:,np.newaxis]
    f = np.cumsum(m, 0)
    s = np.cumsum(m**2, 0)
    f = (np.r_[f[RC:], np.repeat(f[[-1]],RC, axis=0)] - np.r_[np.zeros((LC+1,dim)), f[:-LC-1]]) / n
    s = (np.r_[s[RC:], np.repeat(s[[-1]],RC, axis=0)] - np.r_[np.zeros((LC+1,dim)), s[:-LC-1]]
        ) / (n-1 if unbiased else n) - f**2 * (n/(n-1) if unbiased else 1)
    #print(s)
    print(m.shape)
    print(f.shape)
    print(s.shape)
    m = (m - f) / np.sqrt(s)
    m[s <= 0] = 0
    return m


def cmvn_floating_kaldi(x, LC,RC, norm_vars=True):
    """Mean and variance normalization over a floating window.
    x is the feature matrix (nframes x dim)
    LC, RC are the number of frames to the left and right defining the floating
    window around the current frame. This function uses Kaldi-like treatment of
    the initial and final frames: Floating windows stay of the same size and
    for the initial and final frames are not centered around the current frame
    but shifted to fit in at the beginning or the end of the feature segment.
    Global normalization is used if nframes is less than LC+RC+1.
    """
    N, dim = x.shape
    win_len = min(len(x),  LC+RC+1)
    win_start = np.maximum(np.minimum(np.arange(-LC,N-LC), N-win_len), 0)
    f = np.r_[np.zeros((1, dim)), np.cumsum(x, 0)]
    x = x - (f[win_start+win_len]-f[win_start])/win_len
    if norm_vars:
      f = np.r_[np.zeros((1, dim)), np.cumsum(x**2, 0)]
      x /= np.sqrt((f[win_start+win_len]-f[win_start])/win_len)
    return x



def cm_floating(m,LC,RC, unbiased=False):
    """Mean and variance normalization over a floating window.
    m is the feature matrix (nframes x dim)
    LC, RC is the nmber of frames to the left and right defining the floating
    window around the current frame 
    """
    nframes, dim = m.shape
    LC = min(LC,nframes-1)
    RC = min(RC,nframes-1)
    n = (np.r_[np.arange(RC+1,nframes), np.ones(RC+1)*nframes] - np.r_[np.zeros(LC), np.arange(nframes-LC)])[:,np.newaxis]
    f = np.cumsum(m, 0)
    f = (np.r_[f[RC:], np.repeat(f[[-1]],RC, axis=0)] - np.r_[np.zeros((LC+1,dim)), f[:-LC-1]]) / n
    #print(s)
    m = (m - f)
    return m


psophometric_fir = np.array([
   4.45651e-01,  -6.71610e-02,  -3.99779e-01, -2.04052e-01,  -1.16012e-01,   3.72950e-02,
   8.73900e-02,   1.04088e-01,   7.78700e-02,  5.31500e-02,   2.32560e-02,   8.22700e-03,
   3.16000e-04,   1.94000e-03,   4.63100e-03, -7.13000e-04,  -3.86600e-03,  -7.45300e-03,
  -1.12060e-02,  -1.03200e-02,  -1.13440e-02, -5.34500e-03,  -3.36400e-03,  -1.53300e-03,
  -3.26400e-03,  -2.36500e-03,  -2.63000e-03, -1.81300e-03,  -1.70800e-03,  -4.50000e-05,
   1.98500e-03,   1.36500e-03,   4.07000e-04,  6.40000e-05,  -7.00000e-06,   1.17200e-03,
   4.54000e-04,   1.75400e-03,   1.95900e-03,  2.14800e-03,   4.46000e-04,   8.00000e-05,
  -1.31000e-04,   3.25000e-04,  -1.18000e-04, -1.41000e-04,   6.67000e-04,   4.80000e-04,
  -4.50000e-04,  -1.11600e-03,  -1.29900e-03, -2.08000e-04,  -5.98000e-04,  -2.41000e-04,
  -1.17000e-04,   5.88000e-04,  -1.17000e-04, -4.50000e-04,  -4.75000e-04,   4.93000e-04,
   6.98000e-04,   3.37000e-04,   4.96000e-04,  6.13000e-04,   2.90000e-04,  -1.52000e-04,
  -5.21000e-04,   5.81000e-04,   6.35000e-04,  5.93000e-04,  -2.40000e-05,   2.94000e-04,
  -1.00000e-04,  -4.60000e-04,  -9.22000e-04, -2.59000e-04,   1.40000e-04,  -4.72000e-04,
   5.88000e-04,   1.61100e-03,   1.40000e-04, -1.72100e-03,  -2.36300e-03,  -2.23800e-03,
  -1.45300e-03,  -5.86000e-04,   1.37000e-04,  5.76000e-04,   8.04000e-04,   8.50000e-04,
   8.19000e-04,   7.32000e-04,   6.82000e-04,  6.00000e-04,   5.12000e-04,   3.82000e-04,
   2.61000e-04,   1.43000e-04,   5.10000e-05, -2.80000e-05,  -6.90000e-05,  -9.20000e-05,
  -1.04000e-04,  -1.29000e-04,  -1.38000e-04, -1.42000e-04,  -1.29000e-04,  -1.25000e-04,
  -1.04000e-04,  -8.40000e-05,  -6.00000e-05, -5.10000e-05,  -4.20000e-05,  -3.50000e-05,
  -1.40000e-05,  -2.00000e-06,   1.30000e-05,  2.40000e-05,   3.30000e-05,   3.20000e-05,
   2.60000e-05,   2.30000e-05,   2.30000e-05,  1.80000e-05,   1.80000e-05,   1.10000e-05,
   1.20000e-05,   3.00000e-06,  -5.00000e-06, -1.10000e-05,  -1.00000e-05,  -8.00000e-06,
  -1.00000e-05,  -1.00000e-05,  -6.00000e-06, -3.00000e-06,  -1.00000e-06,   0.00000e+00,
   7.00000e-06,   1.30000e-05,   1.40000e-05,  1.40000e-05,   1.80000e-05,   1.80000e-05,
   1.40000e-05,   1.50000e-05,   1.60000e-05,  1.60000e-05,   1.30000e-05,   1.20000e-05,
   1.00000e-05,   5.00000e-06,  -3.00000e-06, -8.00000e-06,  -1.10000e-05,  -1.10000e-05])


def A_weighting(f):
    """ A-weighting function for weighting magnitude spectrum to account for the 
    relative loudness perceived by the human ear.
    Input:
        f: frequency in Hz
    Output:
        w: weight correspondig to the frequency 'f'
    Example of calculating A-weighted per-frame energy of 8kHz signal 'x':
        a_weights = pytel.features.A_weighting(np.linspace(0,4000, 129))
        absX = np.abs(pytel.features.spectrogram(sf, 200, 200-80, 256, ZMEANSOURCE=True))
        frame_energy =np.power(absX * a_weights, 2).sum(1)
    """
    c1 = 3.5041384e16
    c2 = 20.598997**2
    c3 = 107.65265**2
    c4 = 737.86223**2
    c5 = 12194.217**2
    return c1*f**8 / (((c2+f**2)**2) * (c3+f**2) * (c4+f**2) * ((c5+f**2)**2))


if(__name__=="__main__"):
    pass
