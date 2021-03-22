#!/usr/bin/env python
"""
Probabilstic Linear Discriminant analysis.
All variants of Gaussian PLDA are implemented here

Follows Agnitio's implementation by Brummer
"""


import os,sys
import numpy as np
import scipy as sp
import h5py
from utils import *
from scipy.sparse import coo_matrix

def logdet(M):
    """Return the log determinant of a matrix"""
    return logdet_chol(spl.cholesky(M))

def logdet_chol(cholM):
    """Return the log determinant of a matrix given the cholesky decomposition"""
    return 2*np.sum(np.log(np.diagonal(cholM)))

def invhandle(M, func_only=True):
    """Return a function handle on multiplying a matrix
       by the inverse of the marix provided in this function
       if func_only=False then return: [logdet,chol,handle]
       (Niko's style)
       This is used when scoring the PLDA system
       """
    Cm = np.linalg.cholesky(M)
    def h(A):
        return np.linalg.solve(Cm.T, np.linalg.solve(Cm, A.T))
    if(func_only):
        return h
    else:
        logdet = logdet_chol(Cm)
        return [logdet, Cm, h]

def symmetrize(a):
    a = a + a.T
    a *= 0.5
    return a

class PLDA(object):
    def __init__(self,vdim,srank,crank,diag="isotropic"):
        """
            PLDA model
            Inputs: ivector dim, speaker and channel ranks
        """
        
        self.vdim=vdim
        self.srank=srank
        self.crank=crank
        self.noise_type=diag

        np.random.seed(7)
        self.V=np.array(np.random.randn(vdim,srank),np.float)*1e0
        self.U=np.array(np.random.randn(vdim,crank),np.float)*1e0
        if(diag=="zero"):
            self.D=np.zeros((vdim,1),np.float)
        else:
            self.D=np.ones((vdim,1),np.float)

        self.mu=np.zeros((vdim,1),np.float)

        # expectations
        self._R=np.eye(self.srank+self.crank,self.srank+self.crank)
        self._T=np.zeros((self.srank+self.crank,self.vdim))
        self.Syy=np.zeros((self.srank,self.srank))
        # point estimates
        self._y=None
        self._x=None

    def __str__(self):
        return "= PLDA model d=%d,s=%d,c=%d,type=%s"%(self.vdim,self.srank,self.crank,self.noise_type)

    @classmethod
    def from_params(cls, V, U, D, noise_type, mu):
        """Create object given the parameters
        """
        vdim, srank = V.shape
        crank = U.shape[1]
        plda = PLDA(vdim, srank, crank, noise_type)
        plda.V = np.array(V)
        plda.U = np.array(U)
        plda.D = np.array(D)
        plda.mu = np.array(mu)
        plda.cache_statistics()
        return plda

    def save_to_dict(self):
        return {'V': self.V, 'D': self.D, 'noise_type': np.array(self.noise_type), 'mu': self.mu, 'U': self.U}

    @classmethod
    def load_from_dict(cls, d):
        return PLDA.from_params(d['V'], d['U'], d['D'], d['noise_type'], d['mu'])

    def save(self, outfile, format = "hdf5"):
        """Save the PLDA parameters, use hdf5 format as default"""
        if format == "hdf5":
            with h5py.File(outfile, 'w') as f:
                f.create_dataset('V', data = self.V)
                f.create_dataset('D', data = self.D)
                f.create_dataset('noise_type', data = np.array(self.noise_type))
                f.create_dataset('mu', data = self.mu)
                f.create_dataset('U', data = self.U)
        else:
            np.savez(outfile,self.V,self.U,self.D,np.array(self.noise_type),self.mu)

    @classmethod
    def load(cls, infile, format = "hdf5"):
        """ Load the PLDA param from file"""
        plda = None
        if format == "hdf5":
            with h5py.File(infile, 'r') as f:
                plda = PLDA.from_params(f['V'], f['U'], f['D'], f['noise_type'], f['mu'])
        else: # npz format
            modelfile=np.load(infile)
            plda=PLDA.from_params(modelfile['arr_0'],modelfile['arr_1'],modelfile['arr_2'],modelfile['arr_3'],modelfile['arr_4'])
        return plda

    @property
    def W(self):
        """Stacked spk and channel matrices"""
        return np.hstack((self.U,self.V))

    @property
    def z(self):
        """Stacked spk and channel factors"""
        return np.hstack((self._x,self._y))

    #def reset(self):
    #    """Reset statistics computed using data"""
    #    self._R=np.eye(self.srank+self.crank,self.srank+self.crank)
    #    self._T=np.zeros((self.srank+self.crank,self.vdim))


    def cache_statistics(self):
        """
            various values independant of the data in the cache object
        """
        # values independant of the data
        mc=AttrDict()
        mc.UD=self.U*self.D
        mc.VD=self.V*self.D
        mc.J=np.dot(self.U.T,mc.VD)
        #mc.K=symmetrize(np.dot(self.U.T,mc.UD)+np.eye(self.crank))
        mc.K=np.sqrt(self.D) * self.U
        mc.K=symmetrize(np.dot(mc.K.T,mc.K)+np.eye(self.crank))

        mc.iK=np.linalg.inv(mc.K)
        mc.P0=mc.VD.T-np.dot(np.dot(mc.J.T,mc.iK),mc.UD.T)
        self.mc=mc
        return mc

    def estep(self,stats):
        """
            Compute expectations
            Input: stats object from sufficient_stats()
            Output:
                self._R: second order Expectation
                self._T: first order Expectation
        """
        #print(stats)
        self.nb_spk=np.shape(stats.N)[0]
        self.nb_ex=sum(stats.N)
        # Compute some stats
        mc=self.cache_statistics()

        # Covariance and first order stats preparation
        Py=symmetrize(np.dot(mc.P0,self.V)) #(DV-iKJDU)V=VDV-JiKJ
        Py_yhat=np.dot(mc.P0,stats.F.T) #(DV=iKJDU)F=DVF-iKJDUF=DVF-Jxtilda

        # spk related: y, Ryy and Syy
        self._y=np.zeros((self.srank,self.nb_spk))
        Ryy=np.zeros((self.srank,self.srank))
        self.Syy=np.zeros((self.srank,self.srank))
        mc.obj_y=0.0
        for u in np.unique(stats.N):
            idxs=np.where(stats.N==u)[0]
            nspk=len(idxs)

            # Posterior of y
            Py_u=u*Py+np.eye(self.srank)
            iPy_u=np.linalg.inv(Py_u)
            self._y[:,idxs]=np.dot(iPy_u,Py_yhat[:,idxs]) # made it slower by not using solve so that it's faster later by storing the inv

            # Update Ryy and Syy
            Syyu=nspk*iPy_u+np.dot(self._y[:,idxs],self._y[:,idxs].T)
            Ryy+=u*Syyu
            self.Syy+=Syyu

            # Objective function
            mc.obj_y-=0.5*(nspk*logdet(Py_u)-np.sum(np.sum(self._y[:,idxs]*Py_yhat[:,idxs]))) #obj value for y posterior # HALF TIME THERE in logdet

        # T matrix
        Ty=np.dot(self._y,stats.F)
        JTy=np.dot(mc.J,Ty)
        _Ttemp=np.dot(mc.UD.T,stats.S)-JTy
        Tx=np.linalg.solve(mc.K,_Ttemp)
        self._T=np.vstack((Tx,Ty))

        ## R matrix
        # Ryy
        # Done before

        # Rxy
        _Rxytemp=np.dot(Ty,mc.UD)
        Rxy=np.dot(mc.iK,_Rxytemp.T-np.dot(mc.J,Ryy))

        # Rxx
        Rxx=self.nb_ex*mc.iK
        # infamous equation 59
        _udsdu=symmetrize(np.dot(np.dot(mc.UD.T,stats.S),mc.UD))
        _udtj=np.dot(JTy,mc.UD)
        _jrj=symmetrize(np.dot(np.dot(mc.J,Ryy),mc.J.T))
        #Rxx=np.dot(mc.iK,np.dot(Rxx+_udsdu-_udtj-_udtj.T+_jrj,mc.iK))
        Rxx+=symmetrize(np.dot(mc.iK,np.dot(_udsdu-_udtj-_udtj.T+_jrj,mc.iK)))
        self._R = np.vstack((np.hstack((Rxx,Rxy)),np.hstack((Rxy.T,Ryy))))
        print("[Estep] Traces [R:{:.2f}] [T:{:.2f}] ([Rxx (cov x):{:.2f}] [Rxy:{:.2f}] [(~cov y) Ryy:{:.2f}])".format(np.trace(self._R),np.trace(self._T),np.trace(Rxx/self.nb_ex),np.trace(Rxy),np.trace(Ryy/self.nb_spk)))

    def mstep(self,stats,U=True,V=True,D=False):
        """
            Maximize the auxiliary function
        W=[V U]=inv(R)*T
        D=
        """
        if(U and V):
            W=np.linalg.solve(self._R,self._T).transpose()
            self.U=W[:,0:self.crank]
            self.V=W[:,self.crank:self.crank+self.srank]

        if(U and not V):
            Rxx=self._R[0:self.crank,0:self.crank]
            Tx=self._T[0:self.crank,:]
            Rxy=self._R[0:self.crank,self.crank:self.crank+self.srank]
            self.U=np.linalg.solve(Rxx,Tx-np.dot(Rxy,self.V.T)).T

        if(V and not U):
            Ryy=self._R[self.crank:self.crank+self.srank,self.crank:self.crank+self.srank]
            Ty=self._T[self.crank:self.crank+self.srank,:]
            Rxy=self._R[0:self.crank,self.crank:self.crank+self.srank]
            self.V=np.linalg.solve(Ryy,Ty-np.dot(Rxy.T,self.U.T)).T

        if(D):
            if(self.noise_type=="isotropic"):
                d=(self.nb_ex*self.vdim)/np.trace(stats.S-np.dot(self.W,self._T))
                if(d<=0):
                    raise Exception("Noise model negative {:.2f}".format(d))
                self.D=d*np.ones_like(self.D)
            elif(self.noise_type=="diagonal"):
                D=(self.nb_ex)/(np.diagonal(stats.S)-np.diagonal(np.dot(self.W,self._T)))
                self.D=D[:,np.newaxis] #remove diag for fullcov noise or diagonalize, and normalize the data using SVD decomposition
            else:
                raise Exception("Unknown noise type {:s}".format(self.noise_type))
        print("[Mstep] Traces [V:{:.3f}] [U:{:.3f}] [D:{:.3f}]".format(np.trace(np.dot(self.V.T,self.V)),np.trace(np.dot(self.U.T,self.U)),np.sum(self.D)))

    def mdstep(self,joint=True):
        """ Perform minimum divergence. Fix the lower bound work on the KL of the auxiliary
            Faster convergence properties
            Will just do the normalization w.r.t expectations
        """
        Rxx=self._R[0:self.crank,0:self.crank]
        Ryy=self._R[self.crank:self.crank+self.srank,self.crank:self.crank+self.srank]
        Rxy=self._R[0:self.crank,self.crank:self.crank+self.srank]

        corrY=self.Syy/float(self.nb_spk)

        self.V=np.dot(self.V,sp.linalg.cholesky(corrY, lower=True))

        if(joint==True): # joint mindiv
            G = sp.linalg.solve(Ryy,Rxy.T).T
            C = (1.0/float(self.nb_ex))*(Rxx-symmetrize(np.dot(G,Rxy.T)))
            self.U=np.dot(self.U,sp.linalg.cholesky(C, lower=True))
            self.V+=np.dot(self.U,G)
        else:
            corrX=Rxx/float(self.nb_ex)
            self.U=np.dot(self.U,np.linalg.cholesky(corrX))
        print("[MDstep] Traces [V:{:.3f}] [U:{:.3f}] [D:{:.3f}]".format(np.trace(np.dot(self.V.T,self.V)),np.trace(np.dot(self.U.T,self.U)),np.sum(self.D)))

    @classmethod
    def train_from_stats(cls, stats,srank,crank,diag="diagonal",nb_em_it=10):
        """
            Train a PLDA model based on Aginitio's implementation
            This follows Prince's method for Face recognition, using an EM algorithm.
            The algorithm is much faster by using sufficient statistics
            Input:
                stats object from sufficient_stats()
                    stats.mu: mean of ivectors
                    stats.N: vector of counts for each speaker
                    stats.F: vector of sum of ivector for each speaker
                    stats.S: global second order stat
                srank, crank: rank of the speaker and channel subspaces
                diag: diagonal model type: "isotropic", "diagonal", "full"
                nb_em_it: number of iterations
            Ouput:
                PLDA object. PLDA.V and PLDA.U gives the subspaces
        """
        ivect_dim=np.shape(stats.F)[1]
        if srank is None:
            srank = ivect_dim
        if crank is None:
            crank = ivect_dim
        
        plda=PLDA(ivect_dim,srank,crank,diag=diag)
        plda.mu=stats.mu
        if(diag=="isotropic"):
            plda.D=np.sum(stats.N)/np.mean(np.diag(stats.S))*np.ones((ivect_dim,1),np.float)
        else:
            plda.D=np.reshape(np.sum(stats.N)/np.diag(stats.S),[ivect_dim,1])
        obj = []
        for it in range(nb_em_it):
            plda.estep(stats)
            plda.mstep(stats,U=True,V=True,D=True)
            plda.mdstep()
            o = plda.likelihood(stats)
            print("**[EM IT {:d}] Obj. function value {:.5f}".format(it, o))
            obj.append(o)
        plda.cache_statistics()
        return plda, obj

    @classmethod
    def train(cls, data, class_ids, srank, crank, diag="diagonal", nb_em_it=10):
        spk2seg_mx = coo_matrix((np.ones(len(class_ids)), (class_ids, range(len(class_ids)))), dtype=int)
        stats = AttrDict()
        stats.mu =data.mean(0)
        data_mu = data-stats.mu
        stats.S = data_mu.T.dot(data_mu)
        stats.F = spk2seg_mx.dot(data_mu)
        stats.N = np.array(spk2seg_mx.sum(1))
        return PLDA.train_from_stats(stats, srank, crank, diag=diag, nb_em_it=nb_em_it)


    def likelihood(self,stats):
        """ Compute EM auxiliary """
        llk = 0
        N = self.nb_ex
        llkD = 0.5 * N * np.log(self.D).sum() - 0.5 * np.dot(self.D.T,np.diagonal(stats.S))
        llkY = self.mc.obj_y
        llkX = -0.5 * (self.nb_ex * logdet(self.mc.K) - np.trace(np.dot(self.mc.iK, self.mc.UD.T.dot(stats.S).dot(self.mc.UD))))
        print("{:.2f} {:.2f} {:.2f}".format(llkD,llkY,llkX))
        llk = llkD+llkY+llkX
        return llk

    def score_with_constant_N(self,nT,fT,nt,ft):
        """  Compute verification score
             This changes for every pair nT nt and can be done only once for 1 session training
             fT = P*y_head = (V'D-J'iK*U'D)*f, where f is the first order stats
        """
        # Covariance and first order stats preparation
        Py=symmetrize(np.dot(self.mc.P0,self.V)) #(DV-iKJDU)V=VDV-JiKJ

        # P in eq. 31, EM for PLDA by Niko
        Py_Train=nT*Py+np.eye(self.srank)
        Py_Test=nt*Py+np.eye(self.srank)
        Py_SameSpk=(nt+nT)*Py+np.eye(self.srank)

        [logdetQ1,cholQ1,hQ1]=invhandle(Py_Train,func_only=False)
        [logdetQ2,cholQ2,hQ2]=invhandle(Py_Test,func_only=False)
        [logdetQ12,cholQ12,hQ12]=invhandle(Py_SameSpk,func_only=False)
        # fT*hQ1(fT.T) is terms y_head*P*y_head in eq 30, EM for PLDA by Niko
        Q1=0.5*(logdetQ1+sum(fT*hQ12(fT.T))-sum(fT*hQ1(fT.T)))
        Q2=0.5*(logdetQ2+sum(ft*hQ12(ft.T))-sum(ft*hQ2(ft.T)))
        #Q2=Q2[np.newaxis,:] (can be dangerous)
        Q1=Q1[:,np.newaxis]
        scores=np.dot(fT.T,hQ12(ft.T))
        Q2-=0.5*logdetQ12
        scores+=Q1
        scores+=Q2
        return scores

    def prepare_stats(self, data, seg2model=None):
        """ Convert data (e.g. i-vectors) to statistics useful for scoring
            Input:
                data:      ivect-dim x nb_ex matrix
                seg2model: defines of multisession enrollment (or test). It is
                           2 column array of integer indices mapping rows of data
                           (1st column) to rows of output statistics (2nd column).
                           By default (seg2model=None) each vector in data has its
                           own output vector of statistics (i.e. single session
                           enrollment or test). seg2model can be also represented
                           by coo_matrix or by 1D array of labels if each input
                           vector belongs to exactly one enrollment.
            Output: a AttrDict object with
                    N: vector of counts for each speaker
                    F: array of sum of ivector for each speaker transformed by P0
                       to srank dimensionality
        """
        if type(seg2model) is not coo_matrix:
            if seg2model is None:
                seg2model = np.arange(len(data),dtype=np.int32)
            if seg2model.ndim == 1:
                seg2model = np.c_[np.arange(len(data),dtype=np.int32), seg2model]
            print(seg2model)
            seg2model = coo_matrix((np.ones(len(seg2model)), (seg2model[:,0], seg2model[:,1])))

        stats = AttrDict()
        stats.N = np.array(seg2model.T.sum(1))
        stats.F = np.dot(self.mc.P0, seg2model.T.dot(data-self.mu).T).T
        return stats


    def score(self, Tstats,tstats):
        """
            Score ivectors based on the PLDA model
            Input:
                PLDA object. PLDA.V and PLDA.U gives the subspaces
                stats objects: enroll and test
            Output:
                2D array of scores of all possibilities
        """
        # Create scores
        scores=np.zeros((len(Tstats.N),len(tstats.N)),'f')
        (a,b)=scores.shape
        print("Scoring {:d} models against {:d} tests for {:d} trials".format(a,b,a*b))

        # Score for each uniq combination of N enroll and M test sessions (only enroll for now)
        for n_enroll_sessions in np.unique(Tstats.N):
            idxs_T = np.where(Tstats.N == n_enroll_sessions)[0]
            print("{:d} session(s) enrollement, nb speakers {:d}".format(n_enroll_sessions,len(idxs_T)))
            for n_test_sessions in np.unique(tstats.N):
                idxs_t = np.where(tstats.N == n_test_sessions)[0]
                print("{:d} session(s) tests, nb speakers {:d}".format(n_test_sessions,len(idxs_t)))
                scores[np.ix_(idxs_T, idxs_t)]=self.score_with_constant_N(n_enroll_sessions, Tstats.F[idxs_T,:].T,
                                                                          n_test_sessions,   tstats.F[idxs_t,:].T)
        return scores

    #def score_from_data(self, Tdata, tdata, Tseg2model=None, tseg2model=None):
    #    self.cache_statistics()
    #    Tstats = self.plda_sufficient_stats(Tdata, Tseg2model)
    #    tstats = self.plda_sufficient_stats(tdata, tseg2model)
    #    return self.score(Tstats,tstats):

    def posterior(self, data, seg2model=None):
        """ Calculate p(data|model)
            F is the first order stats
        """
        if type(seg2model) is not coo_matrix:
            if seg2model is None:
                seg2model = np.arange(len(data))
            if seg2model.ndim == 1:
                seg2model = np.c_[np.arange(len(data)), seg2model]
            seg2model = coo_matrix((np.ones(len(seg2model)), (seg2model[:,0], seg2model[:,1])))

        stats = AttrDict()
        stats.N = np.array(seg2model.T.sum(1))
        stats.F = np.dot(self.mc.P0, seg2model.T.dot(data - self.mu).T).T
        # Covariance and first order stats preparation
        Py = np.dot(self.mc.P0, self.V)  # (DV-iKJDU)V=VDV-iKJK
        Q = np.zeros(stats.F.shape[0])
        if self.U is not None:
            W = spl.inv(self.U.T.dot(self.U) + spl.inv(self.D))
        else:
            W = self.D
        data_mn = data - self.mu
        for n_sessions in np.unique(stats.N):
            idxs = np.where(stats.N == n_sessions)[0]
            #print("{:d} session(s), nb speakers {:d}".format(n_sessions,len(idxs)))
            Py_Test = n_sessions * Py + np.eye(self.srank)
            [logdetQ, cholQ, hQ] = invhandle(Py_Test, func_only=False)
            Q[idxs] = -0.5 * (logdetQ - sum(stats.F[idxs, :].T * hQ(stats.F[idxs, :])))
        Q = Q + 0.5 * (stats.N.T[0] * logdet(W) - \
            seg2model.T.dot(sum(data_mn.T * W.dot(data_mn.T))[:, np.newaxis]).T[0])
        return Q

# End of the class

#
# Main training function
#

def sufficient_stats(data, idxs, mu=None):
    """ Speaker-dep zero and first order, global second order
        Input:
            data: ivect-dim x nb_ex matrix
            idxs: array of ints with the speaker index
            mu the mean of the model
        Output: a AttrDict object with
                N: vector of counts for each speaker
                F: vector of sum of ivector for each speaker
                S: global second order stat
                stats.mu
        if mu is none, compute it from the data (training) otherwise use it.
        mu is used to center the first and second order stats
    """
    idxs = np.array(idxs, np.uint32)
    nb_spk = max(idxs) + 1
    vdim, nb_ex = data.shape
    data = data.astype(np.float32)
    print("== Compute stats: #spk [{:d}] #sent [{:d}] dim [{:d}] ==".format(nb_spk, nb_ex, vdim))

    stats = AttrDict()
    stats.N = np.zeros((nb_spk,1),dtype=np.uint32)
    stats.F = np.zeros((nb_spk,vdim),dtype=np.float32)
    stats.S = np.zeros((vdim,vdim),dtype=np.float32)

    # Centering
    #stats.mu=None
    if mu is None: #This is in training when the mean is not known
        # Compute data mean and scatter matrix (to be affected to the model during training, if not, discarded)
        stats.mu = np.mean(data,1)
        data = (data.T-stats.mu).T
        stats.S = data.dot(data.T)
        print("Data loaded. Trace 2nd order [{:.2f}]".format(np.sqrt(np.trace(stats.S))))

        # Compute sufficient stats
        for iS in range(nb_spk):
            j = np.where(idxs==iS)[0]
            stats.N[iS] = len(j)
            stats.F[iS] = data[:,j].sum(axis=1) # mean already removed

    else: # This is in testing when mu is known (given to the function)
        stats.mu = mu.squeeze()
        print("Using model mean square {:f}".format(stats.mu**2).sum())
        # Compute sufficient stats
        for iS in range(nb_spk):
            j=np.where(idxs==iS)[0]
            stats.N[iS] = len(j)
            stats.F[iS] = data[:,j].sum(axis=1) - stats.N[iS] * stats.mu #remove mean from model
    return stats

def test_plda():
    # Generate synthetic data
    ivect_dim=100
    srank=100
    crank=100

    np.set_printoptions(precision=3,suppress=True)
    nb_spk=200
    nb_rep=10
    np.random.seed(7)
    V=np.array(np.random.randn(ivect_dim,srank),'f')
    U=np.array(np.random.randn(ivect_dim,crank),'f')
    D=np.ones((ivect_dim,1))

    y=np.random.randn(srank,nb_spk)
    idxs=np.zeros((nb_spk*nb_rep),np.uint32)

    M=np.zeros((nb_spk*nb_rep, ivect_dim),'f')
    for spk in range(nb_spk):
        x=np.random.randn(nb_rep,crank)
        z=np.random.randn(nb_rep,ivect_dim)
        for sent in range(nb_rep):
            M[spk*nb_rep+sent,:]=np.dot(V,y[:,spk])+np.dot(U,x[sent].T)+z[sent].T
            idxs[spk*nb_rep+sent]=spk

    Mtest=M.copy()

    # Training
    plda, obj  = PLDA.train(M, idxs, srank, crank, diag="diagonal", nb_em_it=10)
    print(str(PLDA))
    nplda = PLDA.load_from_dict(plda.save_to_dict())

    # Scoring
    Tstats = nplda.prepare_stats(Mtest, idxs)
    tstats = nplda.prepare_stats(Mtest)
    scores = nplda.score(Tstats, tstats)
    return M, nplda, scores



if(__name__=="__main__"):
    #pass
    (M,nplda,scores)=test_plda()
    
    print(scores)
