#!/usr/bin/env python

import os
import numpy as np
import hashlib
import h5py
import sys
import os.path


################################################################################
################################################################################
def compute_list_hash(lst, prefix='', suffix='', alg='sha256'):
    """ Computes the hash of the list

    Input:
        lst  - the list of files
        prefix - prefix for each item of the list
        suffix - prefix for each item of the list
        alg    - the hash algorithm (see hashlib module documentation). 
                 DEFAULT is 'sha256'

    Output:
        str    - the hash string 

    The list will be expanded by prefix and suffix so that list with absolute
    paths, and a list with relative paths and prefix and suffix return the same 
    hash.
    """

    m = hashlib.new(alg)

    for ii, segname in enumerate(lst):
        m.update(prefix + segname + suffix)

    return m.hexdigest()


################################################################################
################################################################################
def load_gzvectors_into_ndarray(lst, prefix='', suffix='', allow_missing=False, 
    dtype=np.float64, msgfn=None):
    """ Loads the scp list into ndarray
    """
    n_data      = lst.shape[0]
    v_dim       = None
    missing     = []

    def default_msgfn(ind, n, segname):
        print("Loading [{}/{}] {}".format(ind, n, segname))
        
    def dummy_msgfn(ind, n, segname):
        pass
        
    if msgfn == 'default':
        msgfn = default_msgfn
    elif msgfn == None:
        msgfn = dummy_msgfn

    for ii, segname in enumerate(lst):
        msgfn(ii, n_data, segname)

        try:
            tmp_vec = np.loadtxt(prefix + segname + suffix, dtype=dtype)

        except IOError as e:
            if allow_missing: 
                print(segname, ' missing')
                missing.append(ii)
                continue
            else:
                raise

        if v_dim == None:
            v_dim   = len(tmp_vec)
            out     = np.zeros((n_data, v_dim), dtype=dtype)
        elif v_dim != len(tmp_vec):
            raise ValueError(str.format("Vector {} is of wrong size ({} instead of {})",
                segname, len(tmp_vec), v_dim))
            
        out[ii,:] = tmp_vec

    return out, missing


################################################################################
################################################################################
def load_gzvectors_into_ndarray_h(lst, prefix='', suffix='', allow_missing=False, 
    dtype=np.float64, msgfn=None, cache_root='/tmp', alg='sha256'):
    
    hash_string = compute_list_hash(lst, prefix, suffix, alg)
    cache_file  = cache_root + '/' + hash_string + '.npz'

    if os.path.isfile(cache_file):
        data = np.load(cache_file)
        return data['out'], data['missing']

    out, missing = load_gzvectors_into_ndarray(lst, prefix, suffix, allow_missing,
         dtype, msgfn)

    np.savez(cache_file, out, missing)

    return out, missing


def hash_gzload_ivecs(ivec_dir, segnames, tmp_dir='/tmp', packed=False, reread=False):
    test_ids=np.array(segnames)
    if isinstance(segnames, str):
        test_ids=np.array(np.loadtxt(segnames, dtype=object))
    print(test_ids.dtype)    
    h=hashlib.sha256((ivec_dir+''.join(test_ids)).encode('utf-8')).hexdigest()
    print(h)
    hashfile=tmp_dir+'/'+h+'.h5'
    if os.path.isfile(hashfile) and not reread:
        print('loading hashfile '+hashfile)
        with h5py.File(hashfile, 'r') as f:
            data=np.array(list(f['ivecs']))
            mask=np.array(list(f['mask']))
            test_ids=np.array(list(f['test_ids']))

    else:
        print('loading ivecs from disk: ' + ivec_dir)
        # retrieve dimensionality
        ndim=0
        for n in range(len(test_ids)):
            try:
                iv=np.loadtxt(ivec_dir+'/'+test_ids[n]+'.i.gz')
                ndim=iv.shape[0]
                break
            except IOError:
                pass
        if ndim==0:
            raise Exception("Could not find any .i.gz file")

        assert(ndim < 2000)
        data=np.zeros((len(test_ids), ndim))
        mask=np.ones((len(test_ids),), dtype=bool)

        # read out ivectors
        print("loading {:d} dimensional ivecs, one dot means 10k file loads ({:d} files to load):".format(ndim,len(segnames)))
        for n,line in enumerate(test_ids):
            try:
                data[n,:]=np.loadtxt(ivec_dir+'/'+line+'.i.gz')
            except IOError:
                mask[n]=False
            if (n+1) % 10000 == 0:
                sys.stdout.write('.')
                sys.stdout.flush()

        # save them to hash-named file
        with h5py.File(hashfile, 'w') as f:
            f.create_dataset('ivecs', data=data)
            f.create_dataset('mask', data=mask)
            f.create_dataset('test_ids', data=test_ids.astype("S21"))


    nmissing=len(mask)-np.sum(mask)
    print("Missing {:d}/{:d} files".format(nmissing, len(test_ids)))
    assert nmissing != len(test_ids), 'No data was loaded'

    if packed:
        data=data[mask]
        test_ids=test_ids[mask]
        return (data,test_ids,np.ones(len(test_ids), dtype=np.bool))

    return (data,test_ids,mask)


def hash_gzload_ivecs_cuts(ivec_dir, segnames, tmp_dir='/tmp', packed=False, reread=False):

    test_ids=np.array(segnames)
    if isinstance(segnames, str):
        test_ids=np.array(np.loadtxt(segnames, dtype=str))

    h=hashlib.sha256('gzload_h5_cuts'+ivec_dir+''.join(test_ids)).hexdigest()
    
    hashfile=tmp_dir+'/'+h+'.npz'
    print(hashfile)
    if os.path.isfile(hashfile) and not reread:
        print('loading hashfile '+hashfile)
        D=np.load(hashfile)
        data=D['data']
        mask=D['mask']
        test_ids=D['test_ids']
        try:
            cuts=D['cuts']
        except:
            cuts=None

    else:
        print('loading ivecs from disk: ' + ivec_dir)
        ndim=0

        mask=np.ones((len(test_ids),), dtype=bool)
        cuts=np.zeros((len(test_ids),), dtype=np.ndarray)


        # read out ivectors
        print("loading ivecs, one dot means 10k file loads ({:d} files to load):".format(len(segnames)))
        h5found=False
        for n,line in enumerate(test_ids):
            w=0
            if os.path.exists(ivec_dir+'/'+line+'.h5'):
                with h5py.File(ivec_dir+'/'+line+'.h5',  "r") as f: 
                    w=np.array(f["w"][...],dtype=float)
                    cuts[n]=np.array(f["cuts"][...],dtype=float)
                if not h5found:
                    print('H5 found, louding')
                    h5found=True
            elif os.path.exists(ivec_dir+'/'+line+'.i.gz'):
                assert(h5found is not True)
                w = np.loadtxt(ivec_dir+'/'+line+'.i.gz')
                w=w[np.newaxis,:]
                cuts[n]=np.array([])
            else:
                mask[n]=False
                cuts[n]=np.array([])
            w_dim=0
            try:
                w_dim = w.shape[1]
            except:
                pass

            if w_dim > 0:
                if ndim == 0:
                    assert(w_dim > 1 and w_dim < 2000)
                    ndim = w_dim
                    print("found {:d} dimensional ivectors".format(ndim))
                    data=np.zeros((len(test_ids), ndim))
                data[n,:]=w

            if (n+1) % 10000 == 0:
                sys.stdout.write('.')
                sys.stdout.flush()
    
        if len(filter(lambda x: len(x) > 0, cuts)) == 0:
            # if there are no cuts, set this to None
            cuts=None

        # save them to hash-named file
        np.savez(hashfile, data=data, mask=mask, test_ids=test_ids, cuts=cuts)


    nmissing=len(mask)-np.sum(mask)
    print("Missing {:d}/{:d} files".format(nmissing, len(test_ids)))
    assert nmissing != len(test_ids), 'No data was loaded'

    if packed:
        data=data[mask]
        test_ids=test_ids[mask]
        if cuts is not None:
            try:
                cuts=np.array(cuts[mask])
            except:
                cuts=None
        return (data,test_ids,np.ones(len(test_ids), dtype=np.bool), cuts)



    return (np.array(data),test_ids,mask,cuts)

################################################################################
################################################################################
if(__name__=="__main__"):
    pass
