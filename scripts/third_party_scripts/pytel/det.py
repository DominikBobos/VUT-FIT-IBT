#!/usr/bin/env python
"""
Det curve related tools
"""

import sys,re
import numpy as np

class Det:
    """
    An object to store the ROC curve info
    Pmiss, Pfa: Prob of Miss and FA. Size = sorted_scores.size + 1, Pfa[0]=1-Pmiss[0]=1
    sorted_scores
    ntrue,nfalse: # target and non-target scores
    """
    def __init__(self, tar,non, fa_norm=None, miss_norm=None):
        """
        Compute the DET measures Pfa, PMiss, based on the target and impostor scores.
        Normalizers fa_norm, miss_norm can be used to normalize Pfa, PMiss probabilities
        instead of nfalse, ntrue, respectively. Can be used, for example, for keyword
        spotting, where Pfa can be normalized by # of words per keyword.
        """
        ntrue=tar.size
        nfalse=non.size
        ntotal=ntrue+nfalse
        if(ntrue==0):
            raise Exception("No target trials found")
        if(nfalse==0):
            raise Exception("No impostor trials found")

        Pmiss=np.zeros(ntotal+1,np.float32) # 1 more for the boundaries
        Pfa=np.zeros_like(Pmiss)

        scores=np.zeros((ntotal,2),np.float32)
        scores[0:nfalse,0]=non
        scores[0:nfalse,1]=0
        scores[nfalse:ntotal,0]=tar
        scores[nfalse:ntotal,1]=1

        ## Sort DET scores.
        # Scores are sorted along the first row while keeping the second row fix (same as MATLAB sortrows)
        # Todo: make sure same scores are not sorted
        scores=scores[scores[:,0].argsort(),]

        sumtrue=np.cumsum(scores[:,1])
        sumfalse=nfalse - (np.arange(1,ntotal+1)-sumtrue)

        if miss_norm is None: miss_norm = ntrue
        if fa_norm is None: fa_norm = nfalse

        Pmiss[0]=float(miss_norm-ntrue) / miss_norm
        Pfa[0]=float(nfalse) / fa_norm
        Pmiss[1:]=(sumtrue+miss_norm-ntrue) / miss_norm
        Pfa[1:]=sumfalse / fa_norm
        #print Pfa
        #print Pmiss


        self.Pmiss, self.Pfa, self.ntrue, self.nfalse =  Pmiss, Pfa, ntrue, nfalse
        #TODO: remove sorted_scores, keep only threshold
        self.sorted_scores = scores[:,0]
        # Array of thresholds is extended by -/+ Inf values so that indices
        # returned by scoring methods (e.g. min_dcf) do not run out of the array
        # range. This way Pfa[ind] and Pmiss[ind] can be seen as valid values
        # for any threshod between threshold[ind] and threshold[ind+1]
        self.threshold = np.r_[-np.inf, self.sorted_scores, np.inf]

    ## TODO: All calls of this function should be replaced by Det()
    @classmethod
    def compute_det(self,tar,non):
        return Det(tar,non)

    def cdet(self, p_tar, p_non, threshold, Cmiss=1.0, Cfa=1.0):
        """
        Return the Cdet function as defined by NIST.
        Based on hard decisions given the threshold
        """
        idx = self.sorted_scores.searchsorted(threshold)
        cdet = Cmiss * p_tar * self.Pmiss[idx] + Cfa * p_non * self.Pfa[idx]
        return cdet, idx

    def min_nonlin_dcf(self, p_tar, normalize=True, exp_on_pmiss=2, exp_on_pfa=1):
        """ Non linear DCF defined for the BEST eval
        dcf=Cm0 * Pmiss * Ptar + Cm1 * Pmiss^2 * Ptar +\
            Cfa0 * Pfa * (1-Ptar) + Cfa1 * Pfa^2 * (1-Ptar)
        Ptar=0.01, Cm0=Cfa1=0, Cm1=100, Cfa0=10
        p_tar is computed outside using the above values
        """
        import copy
        det_mod = copy.deepcopy(self)
        det_mod.Pmiss=self.Pmiss**exp_on_pmiss
        det_mod.Pfa=self.Pfa**exp_on_pfa
        return det_mod.min_dcf(p_tar, normalize=normalize)

    def min_dcf(self, p_tar, normalize=True):
        """
        input:
          p_tar: a vector of target priors
          normalize: normalize act DCFs
        output:
          Values of minDCF, one for each value of p_tar, using the optimal
          prior assuming well callibrated scores that can be interpreted as
          log-likelihood ratios
        """
        p_non=1-p_tar
        cdet=np.dot(np.vstack((p_tar, p_non)).T, np.vstack((self.Pmiss,self.Pfa)))
        idxdcfs=np.argmin(cdet, 1)
        dcfs=cdet[np.arange(len(idxdcfs)),idxdcfs]

        if(normalize):
            mins=np.amin(np.vstack((p_tar, p_non)), axis=0)
            dcfs/=mins

        return dcfs.squeeze(), idxdcfs.squeeze()

    def eer(self):
        """
        Compute EER
        det: a Det() object
        """
        # EER
        idxeer=np.argmin(np.abs(self.Pfa-self.Pmiss))
        return 0.5*(self.Pfa[idxeer]+self.Pmiss[idxeer]), idxeer

    def Pfa_at_Pmiss(self, Pmiss):
        """
        Get the Pfa at defined Pmiss
        output:
            threshold, exact value of Pmiss (close to the one defined by parameter), value of Pmiss at Pfa
        """
        idx = np.argmin(np.abs(self.Pmiss-Pmiss))
        return self.Pfa[idx], idx

    def Pmiss_at_Pfa(self, Pfa):
        """
        Get the Pmiss at defined Pfa
        output:
            threshold, exact value of Pfa (close to the one defined by parameter), value of Pmiss at Pfa
        """
        idx=np.argmin(np.abs(self.Pfa-Pfa))
        return self.Pmiss[idx], idx

    def filter_det(self):
        """
        Remove "redundant" points from DET curve, which lives on the
        horizopntal/vertical lines and which do not change shape of the ploted
        DET curve.
        output:
            tuple of filtered Pfa and Pmiss numpy array
        """
        Pmiss_changes = np.r_[True, self.Pmiss[1:] != self.Pmiss[:-1], True]
        Pfa_changes   = np.r_[True, self.Pfa[1:]   != self.Pfa[:-1],   True]
        keep = (Pmiss_changes[1:] | Pmiss_changes[:-1]) & (Pfa_changes[1:] | Pfa_changes[:-1])
        return self.Pfa[keep], self.Pmiss[keep]


    # Ploting functions
    def plot_roc(self, axes=None, *args, **kwargs):
        """  Plot ROC curve """
        from matplotlib.pyplot import gca, draw_if_interactive
        Pfa, Pmiss = self.filter_det()
        if axes is None: axes = gca()
        axes.plot(Pfa*100, Pmiss*100, **kwargs)
        axes.set_xlabel("FA [%]", fontsize = 12)
        axes.set_ylabel("Miss [%]", fontsize = 12)
        axes.grid(True)
        draw_if_interactive()

    def plot_det(self, axes=None, *args, **kwargs):
        """  Plot DET curve """
        from matplotlib.pyplot import gca
        import probit_scale
        if axes is None: axes = gca()
        axes.set_xscale('probit', unit_scale=100)
        axes.set_yscale('probit', unit_scale=100)
        self.plot_roc(axes=axes, **kwargs)

    def plot_dr30(self, axes=None, label='DR30', **kwargs):
        """  Plots a indicating the Doddington 30 point """
        from matplotlib.pyplot import gca, draw_if_interactive
        if axes is None: axes = gca()
        axes.plot(axes.xaxis.get_view_interval(), (30.0/self.ntrue*100,)*2, 'c--', label=label, **kwargs)
        axes.plot((30.0/self.nfalse*100,)*2, axes.yaxis.get_view_interval(), 'c--', **kwargs)
        draw_if_interactive()

    def plot_min_dcf(self, target_prior, axes=None, label=None, **kwargs):
        """ Plot minDCF point """
        from matplotlib.pyplot import gca, draw_if_interactive
        dcf, dcf_ind = self.min_dcf(target_prior)
        if not label: label='minDCF = %g' % dcf
        if axes is None: axes = gca()
        axes.plot(self.Pfa[dcf_ind]*100, self.Pmiss[dcf_ind]*100, 'o', label=label, **kwargs)
        draw_if_interactive()

    def plot_Pfa_at_Pmiss(self, Pmiss, axes=None, label=None, **kwargs):
        """ Plot Pfa point at specific Pmiss"""
        from matplotlib.pyplot import gca, draw_if_interactive
        Pfa, idx = self.Pfa_at_Pmiss(Pmiss)
        if not label: label = 'FA@miss%g%% = %g%%' % (Pmiss*100, Pfa*100)
        if axes is None: axes = gca()
        axes.plot((self.Pfa[idx]*100,)*2, (axes.yaxis.get_view_interval()[0], self.Pmiss[idx]*100), 'k', label=label, **kwargs)
        axes.plot((axes.xaxis.get_view_interval()[0], self.Pfa[idx]*100), (self.Pmiss[idx]*100,)*2, 'k', **kwargs)
        draw_if_interactive()

    def plot_Pmiss_at_Pfa(self, Pfa, axes=None, label=None, **kwargs):
        """ Plot Pmiss point at specific Pfa"""
        from matplotlib.pyplot import gca, draw_if_interactive
        Pmiss, idx = self.Pmiss_at_Pfa(Pfa)
        if not label: label = 'miss@FA%g%% = %g%%' % (Pfa*100, Pmiss*100)
        if axes is None: axes = gca()
        axes.plot((self.Pfa[idx]*100,)*2, (axes.yaxis.get_view_interval()[0], self.Pmiss[idx]*100), 'k', label=label, **kwargs)
        axes.plot((axes.xaxis.get_view_interval()[0], self.Pfa[idx]*100), (self.Pmiss[idx]*100,)*2, 'k', **kwargs)
        draw_if_interactive()

def test_this():
    n=np.random.randn(1001)
    t=np.random.randn(1001)+1
    d = Det(t,n)
    d.plot_det()
    d.plot_dr30()
    d.plot_min_dcf(0.5)
    d.plot_Pfa_at_Pmiss(0.1)
    d.plot_Pmiss_at_Pfa(0.1)
    print("EER", d.eer())
    print("Pmiss_at_Pfa0.1", d.Pmiss_at_Pfa(0.1))
    t=np.random.randn(1001)+2
    d2 = Det(t,n)
    d2.plot_det()
    legend()


if(__name__=="__main__"):
    pass
