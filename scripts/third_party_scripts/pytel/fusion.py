#!/usr/bin/env python

"""
Code for logistic regression fusion
"""

import numpy as np
import scipy.optimize
import utils
from utils import profileit
import time
import numexpr as ne
import scoring
from IPython import embed

def load_scores(scorelist,in_format):

    i = 1
    scoredict = dict()
    for f in scorelist:
        fields = f.split(':')
        if len(fields) == 2: sysname = fields[1]
        else: sysname = 'system'+str(i)
        i += 1
        #idlog.info("Loading {}".format(fields[0]))
        print("Loading {}".format(fields[0]))
        scoredict[sysname] = scoring.Scores.load(fields[0],in_format)
    return scoredict


def apply_calibration_and_fusion(scoredict,key,models,sideinfo=None,use_si_in_fusion=False):
    """
    Apply first the calibration model for each individual system and then the fusion model
    """

    valid_scores = key.mask != 0
    num_scores = len(valid_scores.ravel())

    if sideinfo: 
        # The sideinfo is input as a Scores class, where the values of the scores are the sideinfo 
        si = sideinfo.align(key)
    else:
        # Fake si with a single sideinfo value equal to 0
        si = scoring.Scores(key.train_ids, key.test_ids, np.zeros_like(key.mask))
    si_values = list(np.unique(si.score_mat.ravel()))
    if np.NINF in si_values: si_values.remove(np.NINF)

    # Apply calibration and replace missing values with 0s
    # Accumulate the resulting scores for fusion
    scores = np.empty((num_scores, 1, len(scoredict)))
    sys = 0

    for sysname in sorted(scoredict.keys()):
        scr = scoredict[sysname].align(key)
        cal_scores = np.zeros_like(scr.score_mat)

        for siv in si_values:

            ind_valid_scores  = valid_scores & (scr.mask != 0) & (scr.score_mat != np.NINF) & (si.score_mat == siv)
            ind_scores        = np.empty((num_scores, 1, 1))
            ind_scores[:,0,0] = scr.score_mat.ravel()
            ind_scores        = ind_scores[ind_valid_scores.ravel()]

            if sysname+','+str(siv) in models:
                model = models[sysname+','+str(siv)]
            else:
                raise Exception("Model {},{} missing".format(sysname,str(siv)))
            
            ind_calibrated_scores = apply_nary_lin_fusion(ind_scores, model[1:], model[0])

            # Replace valid scores with their calibrated values, leaving all invalid values as zero.
            cal_scores[ind_valid_scores] = ind_calibrated_scores[:,0]

        scores[:,0,sys] = cal_scores.ravel()
        sys += 1

    if not use_si_in_fusion:
        si_values = [0]
        si.score_mat[:,:] = 0

    fused_scores = np.zeros_like(scr.score_mat)
    for siv in si_values:
        si_valid_scores  = valid_scores & (si.score_mat == siv)
        si_scores = scores[si_valid_scores.ravel()]
        model = models['fusion,'+str(siv)]
        fused_scores[si_valid_scores] = apply_nary_lin_fusion(si_scores,  model[1:], model[0])[:,0]

    # Align with key again to make sure that all scores for trials not valid in key are turned to NINF
    return scoring.Scores(scr.train_ids,scr.test_ids,fused_scores).align(key)


def train_calibration_and_fusion(scoredict,key,priors,outfile,sideinfo=None,use_si_in_fusion=False):
    """
    Train calibration models for scores in the scoredict, first aligning them to key.
    If sideinfo is defined, a separate calibration model for each sideinfo value is created and applied for each
    system. The fusion model is then trained with the calibrated individual systems.
    """

    f = open(outfile,'w')

    valid_scores = key.mask != 0
    classf = (key.mask < 0).astype(int).ravel()
    num_scores = len(valid_scores.ravel())
    if sideinfo: 
        # The sideinfo is input as a Scores class, where the values of the scores are the sideinfo 
        si = sideinfo.align(key)
    else:
        # Fake si with a single sideinfo value equal to 0
        si = scoring.Scores(key.train_ids, key.test_ids, np.zeros_like(key.mask))
    si_values = list(np.unique(si.score_mat.ravel()))
    if np.NINF in si_values: si_values.remove(np.NINF)

    # Models will contain the calibration models and the fusion model.
    models = dict()

    # Train calibration for each individual system, apply calibration and replace missing values with 0s
    # Accumulate the resulting scores for fusion                                                                         
    scores = np.empty((num_scores, 1, len(scoredict)))
    sys = 0

    for sysname in sorted(scoredict.keys()):
        scr = scoredict[sysname].align(key)
        cal_scores = np.zeros_like(scr.score_mat)

        for siv in si_values:

            ind_valid_scores  = valid_scores & (scr.mask != 0) & (scr.score_mat != np.NINF) & (si.score_mat == siv)
            ind_scores        = np.empty((num_scores, 1, 1))
            ind_scores[:,0,0] = scr.score_mat.ravel()
            ind_scores        = ind_scores[ind_valid_scores.ravel()]
            ind_classf        = classf[ind_valid_scores.ravel()]

            #idlog.info("Training calibration model for system {} with %d impostor samples and %d target samples for sideinfo value {}".format(sysname,len(np.where(ind_classf==1)[0]),len(np.where(ind_classf==0)[0]),siv))
            print("Training calibration model for system {} with {:d} impostor samples and {:d} target samples for sideinfo value {}".format(sysname,len(np.where(ind_classf==1)[0]),len(np.where(ind_classf==0)[0]),siv))
            tgts = ind_scores[ind_classf==0]
            imps = ind_scores[ind_classf==1]
            #idlog.info("Stats before calibration: meantgt = %5.2f, sdevtgt = %5.2f, meanimp = %5.2f, sdevimp = %5.2f" % (np.mean(tgts),np.std(tgts),np.mean(imps),np.std(imps)))
            print("Stats before calibration: meantgt = {:5.2f}, sdevtgt = {:5.2f}, meanimp = {5.2f}, sdevimp = {5.2f}".format(np.mean(tgts),np.std(tgts),np.mean(imps),np.std(imps)))
            ind_alpha, ind_beta = train_nm1ary_llr_fusion(ind_scores, ind_classf, priors=priors)
            sysnameiv = sysname+','+str(siv)
            models[sysnameiv] = np.concatenate((ind_beta, ind_alpha))
            f.write('%-20s {}\n'.format(sysnameiv,' '.join(map(str, models[sysnameiv]))))

            ind_calibrated_scores = apply_nary_lin_fusion(ind_scores, ind_alpha, ind_beta)
            tgts = ind_calibrated_scores[ind_classf==0]
            imps = ind_calibrated_scores[ind_classf==1]
            #idlog.info("Stats after calibration:  meantgt = %5.2f, sdevtgt = %5.2f, meanimp = %5.2f, sdevimp = %5.2f" % (np.mean(tgts),np.std(tgts),np.mean(imps),np.std(imps)))
            print("Stats after calibration:  meantgt = {:5.2f}, sdevtgt = {:5.2f}, meanimp = {:5.2f}, sdevimp = {:5.2f}".format(np.mean(tgts),np.std(tgts),np.mean(imps),np.std(imps)))

            # Replace valid scores with their calibrated values
            cal_scores[ind_valid_scores] = ind_calibrated_scores[:,0]

        scores[:,0,sys] = cal_scores.ravel()
        sys += 1

    if not use_si_in_fusion:
        si_values = [0]
        si.score_mat[:,:] = 0

    for siv in si_values:
        si_valid_scores  = valid_scores & (si.score_mat == siv)
        si_scores = scores[si_valid_scores.ravel()]
        si_classf = classf[si_valid_scores.ravel()]

        #idlog.info("Training fusion model with {:d} impostor samples and {:d} target samples for sideinfo value {}".format(len(np.where(si_classf==1)[0]),len(np.where(si_classf==0)[0]),siv))
        print("Training fusion model with {:d} impostor samples and {:d} target samples for sideinfo value {}".format(len(np.where(si_classf==1)[0]),len(np.where(si_classf==0)[0]),siv))
        alpha, beta = train_nm1ary_llr_fusion(si_scores, si_classf, priors=priors)
        fusionname = 'fusion,'+str(siv)
        models[fusionname] = np.concatenate((beta, alpha))
        f.write('%-20s {}\n'.format(fusionname,' '.join(map(str, models[fusionname]))))
        
    f.close()

    return models
    
def sigmoid(lodds):
    """Inverse of logit function"""
    return 1 / (1 + np.exp(-lodds))


def loglh2posterior(loglh, prior=1):
    """
    Converts log-likelihoods to class log-posteriors

    Input:
    loglh - matrix log-likelihoods (trial, trial)
    prior - vector of class priors (default: equal priors)

    Output:
    logp  - matrix log-posteriors (trial, class)
    """
    loglh = loglh + np.log(np.array(prior))
    return loglh - utils.logsumexp(loglh, axis=1)



def loglh2detection_llr(loglh, prior=None):
    """
    Converts log-likelihoods to log-likelihood ratios returned by detectors
    of the individual classes as defined in NIST LRE evaluations (i.e. for each
    detector calculates ratio between 1) likelihood of the corresponding class
    and 2) sum of likelihoods if all other classes). When summing likelihoods
    of the compeeting clases, these can be weighted by vector of priors.

    Input:
    loglh - matrix log-likelihoods (trial, class)
    prior - vector of class priors (default: equal priors)

    Output:
    logllr - matrix of log-likelihood ratios (trial, class)
    """
    logllr = np.empty_like(loglh)
    nc = loglh.shape[1]
    prior = np.ones(nc) if prior is None else np.array(prior, np.float).ravel()
    
    for d in range(nc):
        competitors = np.r_[:d, d+1:nc]
        logweights = np.log(prior[competitors] / sum(prior[competitors]))
        logllr[:,d] = loglh[:,d] - utils.logsumexp(loglh[:,competitors] + logweights, axis=1)
        
    return logllr



def apply_nary_lin_fusion(scores, alpha, beta, out=None):
    """ Apply nary linear fusion

    Input:
    scores - 3D matrix of scores (trial, class, system)
    alpha  - vector of system combination weights
    beta   - vector of score offsets

    Output:
    loglh  - log-likelihood scores of the fused system
    """
    if out is None:
      out = np.empty(scores.shape[:-1])

    np.dot(scores, alpha, out=out)
    out += beta
    return out



def nary_llr_obj_grad(params, scores, trial_wght, y, targets_f, grad_const):
    """ Gradient w.r.t nary logistic regression parameters
    Input:
      params     - vector of fusion parameters to be optimized (S alphas, where
                   S is system s in fusion and C-1 betas, where C is number of
                   clases)
      scores     - 3D array of scores (trial, class, system).
      trial_wght - per trial weight (to compensate for unbalanced
                   representation of classes in training data)
      y          - pre-allocated matrix (#trials,C) serving as a workspace
      targets_f  - vector of true class indeces in flattened score matrix
      grad_const - constant term in gradient corresponding to subtracting 1
                   #from posteriors of target class

    Output:
      fobj       - value of objective function at alpha_beta
      g          - gradient vector at alpha_beta
    """

    tic = time.time()
    # apply fusion to get log likelohhos and normalize to log class posteriors
    nsys = scores.shape[2]
    apply_nary_lin_fusion(scores, params[:nsys], np.r_[params[nsys:], 0], out=y)
    y -= utils.logsumexp(y, axis=1)[:, np.newaxis]

    # evaluate cross entropy objective
    obj = -trial_wght[:,0].dot(y.take(targets_f))

    # Let t be 1-to-N encoding of class labels
    # grad_alpha = sum_n trial_weight[n] * (y[n] - t[n]) * scores[n]
    # grad_beta  = sum_n trial_weight[n] * (y[n] - t[n])
    # t[n] is removed in the calculations below and the cooresponding
    # constant terms are precomputed in grad_const
    ne.evaluate("exp(y)*trial_wght", out=y)
    grad_alpha = y.ravel().dot(scores.reshape(-1, nsys))
    #print("Obj:", obj,  time.time() - tic)
    return obj, np.r_[grad_alpha, y.sum(0)[:-1]] + grad_const


#@profileit
def train_nary_llr_fusion(scores, classf, priors=None, alpha0=None, beta0=None):
    """ Train nary logistic regression fusion parameters
    Input:
      scores - 3D array of scores (trial, class, system)
      classf - true class labels (0..C-1)
      priors - prior probability of classes
      alpha0 - optional: initial values for combination weights
      beta0  - optional: initial values for of score offsets parameters

    Output:
      alpha  - vector of system combination weights
      beta   - vector of score offsets
    """
    ntrials, nclasses, nsystems = scores.shape
    if priors is None:
        priors = np.ones(nclasses) / nclasses
    else:
        priors = np.array(priors,float)

    if alpha0 is None:
        alpha0 = np.ones(nsystems)
    if beta0 is None:
        beta0 = np.zeros(nclasses)

    beta0 += np.log(priors) - np.log(priors[-1])

    class_freq = np.sum(np.arange(nclasses)[:,np.newaxis] == classf, axis=1)
    trial_wght = (priors * ntrials / class_freq)[classf]

    # pre-allocate workspace for nary_llr_obj_grad function
    y = np.empty((ntrials, nclasses))

    # create vector of true class indeces in flattened score matrix
    targets_f = np.ravel_multi_index((np.arange(ntrials), classf), y.shape)

    # precompute a constant term in gradient callculation corresponding
    # to subtracting 1 from posteriors of target class
    grad_const = -np.r_[trial_wght.dot(scores[np.arange(ntrials), classf,:]), priors[:-1] * ntrials]

    xopt, fobj, d = scipy.optimize.fmin_l_bfgs_b(nary_llr_obj_grad,
                      np.r_[alpha0, beta0[:-1] - beta0[-1]],
                      args=(scores, trial_wght[:,np.newaxis], y, targets_f, grad_const))

    # return fusion weights and bias; log prior, which is learned in the bias,
    # is removed; bias is normalized to zero the last element
    return (xopt[:nsystems],
            np.r_[xopt[nsystems:], 0] - np.log(priors) + np.log(priors[-1]))

# The same routines as the two abowe just dolved as NOT overdefined problem,
# where we have only C-1 log-likelihood from each system, where C is number of
# classes (see help of these functions). The two routines above should be 
# implemented using these two routines by subtracting log-likelihood of the
# last class from log-likelihoodsall other likelihoods (e.g. 2-class problem
# is naturally represented by only log-likelihood ratios).
def nm1ary_llr_obj_grad(alpha_beta, scores, trial_wght, y, targets_f):
    """ Gradient w.r.t nary logistic regression parameters
  Input:
      alpha_beta - vector of fusion parameters to be optimized (S alphas, where
                   S is system s in fusion and C-1 betas, where C is number of 
                   clases)
      scores     - 3D array of scores (trial, class, system). Scores for only
                   C-1 classes are expected (i.e. uncalibrated log likelihood
                   ratio between each class and the ommited last class)
      classf     - true class lavels (0..C-1)
      priors     - prior probability of classes
      trial_wght - per trial weight (to compensate for unbalanced
                   representation of classes in training data)
      y          - pre-allocated matrix (#trials,C) as a workspace for calculations

    Output:
      fobj       - value of objective function at alpha_beta
      g          - gradient vector at alpha_beta
    """
    ntrials, nclasses, nsystems = scores.shape
    nclasses += 1
    alpha = alpha_beta[:nsystems]
    beta = alpha_beta[nsystems:]
    # apply fusion to get log likelohhos and normalize to log class posteriors
    y[:,:-1] = scores.dot(alpha) + beta
    y[:,-1] = 0.0
    y -= utils.logsumexp(y, axis=1)[:, np.newaxis]

    # evaluate cross entropy oblective
    obj = -trial_wght.dot(y.take(targets_f))

    # Let t be 1-to-N encoding of classf
    # grad_alpha = sum_n trial_weight[n] * (y[n] - t[n]) * scores[n]
    # grad_beta  = sum_n trial_weight[n] * (y[n] - t[n])
    ne.evaluate("exp(y)", out=y)
    y.flat[targets_f] -= 1

    y *= trial_wght[:, np.newaxis]
    grad_alpha = y[:,:-1].ravel().dot(scores.reshape(-1, nsystems))
    grad_beta = y.sum(0)
    return obj, np.r_[grad_alpha, grad_beta[:-1]]

def train_nm1ary_llr_fusion(scores, classf, priors=None, alpha0=None, beta0=None):
    ntrials, nclasses, nsystems = scores.shape
    nclasses += 1
    if priors is None:
        priors = np.ones(nclasses) / nclasses
    else:
        priors = np.array(priors,np.float).ravel()
        if len(priors) == nclasses-1:
            priors=np.r_[priors, 1.0-priors.sum()]
    if alpha0 is None:
        alpha0 = np.ones(nsystems)
    if beta0 is None:
        beta0 = np.zeros(nclasses-1)
    beta0 += np.log(priors[:-1]) - np.log(priors[-1])

    class_freq = np.sum(np.arange(nclasses) == classf[:,np.newaxis], axis=0)
    trial_wght = (priors * np.sum(class_freq) / class_freq)[classf]
    #print(trial_wght[[0,610]])
    # pre-allocate workspace for nm1ary_llr_obj_grad function
    y = np.empty((ntrials, nclasses)) 
    #idlog.info("Prealocating of %d trials and %d classes finished"%(ntrials, nclasses))
    print("Prealocating of {:d} trials and {:d} classes finished".format(ntrials, nclasses))
    xopt, fobj, d = scipy.optimize.fmin_l_bfgs_b(nm1ary_llr_obj_grad,
                      np.r_[alpha0, beta0], args=(scores, trial_wght, y,
                            np.ravel_multi_index((np.arange(ntrials), classf), y.shape)),
                      factr=1e7, m=10)
    #idlog.info("INFO5")
    # return fusion weights and bias; log prior, which is learned in the bias,
    # is removed; bias is normalized to zero the last element
    return (xopt[:nsystems],
            xopt[nsystems:] - np.log(priors[:-1]) + np.log(priors[-1]))



def binary_llr_obj_grad(alpha_beta, scr_tar, scr_non, wght_tar, wght_non):
    """ Gradient w.r.t nary logistic regression parameters
  Input:
      alpha_beta - vector of fusion parameters to be optimized (S alphas, where
                   S is systems in fusion and scalar beta
                   clases)
      scr_tar    - array (trial, system) of uncalibrated log likelihood ratios for target trials
      scr_tar    - array (trial, system) of uncalibrated log likelihood ratios for non-target trials
      trial_wght - per trial weight (to compensate for unbalanced
      wght_tar
      wght_non   - weights to compensate for unbalanced number of target/non-target trials

    Output:
      fobj       - value of objective function at alpha_beta
      g          - gradient vector at alpha_beta
    """

    alpha = alpha_beta[:-1]
    beta = alpha_beta[-1]

    # get true class posteriors in vector y
    y_tar = np.dot(scr_tar, alpha)
    y_non = np.dot(scr_non, alpha)

    # evaluate cross entropy oblective

    np.logaddexp(0, -y_tar-beta, out=y_tar) # more stable, but possibly slower and more memory consuming (thanks the minus operator)
    np.logaddexp(0, y_non+beta,  out=y_non)
    #ne.evaluate("log1p(exp(-y_tar-beta))", out=y_tar)
    #ne.evaluate("log1p(exp(y_non+beta))", out=y_non)

    obj = ne.evaluate("sum(y_tar,0)") * wght_tar + ne.evaluate("sum(y_non,0)") * wght_non
    #obj = np.sum(y_tar,0) * wght_tar + np.sum(y_non,0) * wght_non


    # evaluate gradient
    #ne.evaluate("(exp(-y_tar)-1.0)", out=y_tar)
    #ne.evaluate("(1.0-exp(-y_non))", out=y_non)
    y_tar = np.exp(-y_tar) - 1.0
    y_non = 1.0 - np.exp(-y_non)
    dalpha = y_tar.dot(scr_tar) * wght_tar + y_non.dot(scr_non) * wght_non
    dbeta  = y_tar.sum()        * wght_tar + y_non.sum()        * wght_non

    print("Obj:", obj)
    return obj, np.r_[dalpha, dbeta]


def train_binary_llr_fusion(scr_tar, scr_non, p_tar=0.5, alpha=None, beta=0.0, maxfun=15000):
    """ Train nary logistic regression fusion parameters
    Input:
      scr_tar - array (trial, system) of uncalibrated log likelihood ratios for target trials
      scr_non - array (trial, system) of uncalibrated log likelihood ratios for non-target trials
      p_tar   - prior probability of target class

    Output:
      alpha  - vector of system combination weights
      beta   - score offsets
    """
    wght_tar =    p_tar  * (len(scr_tar)+len(scr_non)) / len(scr_tar)
    wght_non = (1-p_tar) * (len(scr_tar)+len(scr_non)) / len(scr_non)

    cllr_norm=-(p_tar*np.log(p_tar)+(1-p_tar)*np.log(1-p_tar))*(len(scr_tar)+len(scr_non))
    wght_tar /= cllr_norm
    wght_non /= cllr_norm

    if scr_tar.ndim == 1:
      scr_tar = scr_tar[:,np.newaxis]
      scr_non = scr_non[:,np.newaxis]

    if alpha is None:
        alpha = np.ones(scr_tar.shape[1])
    
    beta +=  np.log(p_tar) - np.log(1-p_tar)

    # pre-allocate workspace for nm1ary_llr_obj_grad function
    #y = np.empty(ntrials)
    xopt, fobj, d = scipy.optimize.fmin_l_bfgs_b(binary_llr_obj_grad,
                      np.r_[alpha, beta], args=(scr_tar, scr_non, wght_tar, wght_non), maxfun=maxfun)

    # return fusion weights and bias; log prior, which is learned in the bias,
    # is removed; bias is normalized to zero the last element
    return (xopt[:-1], xopt[-1] - np.log(p_tar) + np.log(1-p_tar))


# Alternative implementation of binary logistic regression fusion with +1/-1 labels
def binary_llr_obj_grad2(alpha_beta, scores, targets, trial_wght, y):
    """ Gradient w.r.t nary logistic regression parameters
  Input:
      alpha_beta - vector of fusion parameters to be optimized (S alphas, where
                   S is systems in fusion and scalar beta
      scores     - array (trial, system) of uncalibrated log likelihood ratios
      targets    - true labels using -1/1 coding for not-target/target trials
      trial_wght - per trial weight (to compensate for unbalanced
                   representation of classes in training data)
      y          - pre-allocated vector as a workspace for calculations

    Output:
      fobj       - value of objective function at alpha_beta
      g          - gradient vector at alpha_beta
    """
    alpha = alpha_beta[:-1]
    beta = alpha_beta[-1]

    # get true class posteriors in vector y
    np.dot(scores, -alpha, out=y)
    ne.evaluate("1.0/(1.0+exp(targets*(y-beta)))", out=y)

    # evaluate cross entropy oblective
    obj = -ne.evaluate("sum(log(y)*trial_wght,0)")
    print("Obj:", obj)

    # evaluate gradient
    ne.evaluate("(y-1.0)*targets*trial_wght", out=y)
    return obj, np.r_[y.dot(scores), y.sum(0)]



def train_binary_llr_fusion2(scores, targets, p_tar=0.5):
    """ Train nary logistic regression fusion parameters
    Input:
      scores  - array (trial, system) of uncalibrated log likelihood ratios
      targets - true labels using -1/1 coding for not-target/target trials
      p_tar   - prior probability of target class

    Output:
      alpha  - vector of system combination weights
      beta   - score offsets
    """
    ntrials, nsystems = scores.shape
    trial_wght = np.zeros_like(targets, dtype=float)
    trial_wght[targets ==  1] = p_tar     * len(targets) / np.sum(targets ==  1)
    trial_wght[targets == -1] = (1-p_tar) * len(targets) / np.sum(targets == -1)

    # pre-allocate workspace for nm1ary_llr_obj_grad function
    y = np.empty(ntrials)
    xopt, fobj, d = scipy.optimize.fmin_l_bfgs_b(binary_llr_obj_grad2,
                      np.r_[np.ones(nsystems), np.log(p_tar) - np.log(1-p_tar)],
                      args=(scores, targets.astype(float), trial_wght, y))

    # return fusion weights and bias; log prior, which is learned in the bias,
    # is removed; bias is normalized to zero the last element
    return (xopt[:-1], xopt[-1] - np.log(p_tar) + np.log(1-p_tar))



# The math below is based on paper:
# Burget, L., et al.: Discriminatively Trained Probabilistic Linear Discriminant Analysis for Speaker Verification,
# In: Proc. of ICASSP 2011, Prague, 2011

def PLDA_params_to_bilinear_form(WC, AC, mu):
    iTC,    ldTC    = utils.inv_posdef_and_logdet(WC + AC)
    iWC,    ldWC    = utils.inv_posdef_and_logdet(WC)
    iWC2AC, ldWC2AC = utils.inv_posdef_and_logdet(WC + 2*AC)

    Gamma = -0.25*(iWC2AC + iWC - 2*iTC)
    Lambda= -0.5*(iWC2AC - iWC)
    c = (iWC2AC - iTC).dot(mu)
    k = - 0.5*(ldWC2AC + ldWC - 2*ldTC) - mu.T.dot(c)
    return Lambda, Gamma, c, k

def bilinear_scoring(Lambda, Gamma, c, k, Fe, Ft, out=None):
    if out is None:
        out = np.empty((Fe.shape[0], Ft.shape[0]))

    np.dot(Fe.dot(Lambda), Ft.T, out=out)
    tmp = (np.sum(Fe.dot(Gamma) * Fe, 1) + Fe.dot(c))[:,np.newaxis]
    ne.evaluate("out+tmp", out=out) # faster (multithread) version of +=
    tmp = (np.sum(Ft.dot(Gamma) * Ft, 1) + Ft.dot(c))[np.newaxis,:] + k
    ne.evaluate("out+tmp", out=out)
    return out

def DPLDA_unpack_params(params_trained, params_all, params_mask, vdim):
    params_all[params_mask] = params_trained
    Lambda, Gamma, c, alpha, beta = np.split(params_all, [vdim**2, vdim**2*2, vdim**2*2+vdim, -1])
    Lambda = Lambda.reshape(vdim,vdim)
    Gamma  = Gamma.reshape( vdim,vdim)
    Lambda = np.triu(Lambda)+np.triu(Lambda,k=1).T
    Gamma  = np.triu(Gamma) +np.triu(Gamma, k=1).T
    return Lambda, Gamma, c, alpha, beta



def DPLDA_obj_grad(params_trained, params_all, params_mask, params_reference, params_reg, scr_tar,
                   scr_non, Fe, Ft, ids_tar, ids_non, wght_tar, wght_non):
    tic = time.time()
    #print("==", time.time()-tic)
    Lambda, Gamma, c, alpha, beta = DPLDA_unpack_params(params_trained, params_all, params_mask, Fe.shape[1])

    G = bilinear_scoring(Lambda, Gamma, c, beta, Fe, Ft)
    y_tar = G.take(ids_tar)
    y_non = G.take(ids_non)

    if scr_tar is not None:
        y_tar += np.dot(scr_tar, alpha)
        y_non += np.dot(scr_non, alpha)

    #ne.evaluate("log1p(exp(-y_tar))", out=y_tar)
    #ne.evaluate("log1p(exp(y_non))", out=y_non)
    np.logaddexp(0, -y_tar, out=y_tar) # more stable, but possibly slower and more memory consuming (thanks the minus operator)
    np.logaddexp(0, y_non,  out=y_non)

    xent = ne.evaluate("sum(y_tar,0)") * wght_tar + ne.evaluate("sum(y_non,0)") * wght_non
    obj = xent + 0.5*params_reg.dot((params_trained-params_reference)**2)

    #print("Obj:", obj, obj-xent, xent, time.time()-tic, "s")
    #return obj

    ne.evaluate("(exp(-y_tar)-1.0) * wght_tar", out=y_tar)
    ne.evaluate("(1.0-exp(-y_non)) * wght_non", out=y_non)

    # Gradient for standard fusion weights and bias
    if scr_tar is not None:
        dalpha = y_tar.dot(scr_tar) + y_non.dot(scr_non)
    else:
        dalpha = []
    dbeta  = y_tar.sum() + y_non.sum()

    # Gradient of the bilinear form
    #G.fill(0.0)
    ne.evaluate("G*0.0", out=G) # faster (mutithread) way of filling matrix with zeros
    G.flat[ids_tar] = y_tar
    G.flat[ids_non] = y_non

    # dot product with vector of ones is MUCH (7x) faster then sum
    #dataEsumG = Fe * G.sum(1)[:,np.newaxis]
    #dataTsumG = Ft * G.sum(0)[:,np.newaxis]
    dataEsumG = Fe * G.dot(np.ones(G.shape[1]))[:,np.newaxis]
    dataTsumG = Ft * np.ones(G.shape[0]).dot(G)[:,np.newaxis]
    dG = 2 * (Fe.T.dot(dataEsumG) + dataTsumG.T.dot(Ft))
    dG -= np.diag(np.diag(dG)) * 0.5
    dL = Fe.T.dot(G).dot(Ft)
    dL =  (dL + dL.T)
    dL -= np.diag(np.diag(dL)) * 0.5
    dc = dataEsumG.sum(0) + dataTsumG.sum(0)


    print("Obj:", obj, obj-xent, xent, time.time()-tic, "s")
    return obj, np.r_[dL.flat, dG.flat, dc, dalpha, dbeta][params_mask]+(params_trained-params_reference)*params_reg


def DPLDA_obj_grad_with_check(params_trained, params_all, params_mask, params_reference, params_reg, scr_tar,
                   scr_non, Fe, Ft, ids_tar, ids_non, wght_tar, wght_non):
    approx = scipy.optimize.approx_fprime(
                 params_trained,
                 lambda x: DPLDA_obj_grad(x, params_all, params_mask, params_reference, params_reg, scr_tar,scr_non, Fe, Ft, ids_tar, ids_non, wght_tar, wght_non)[0],
                 1e-9)

    obj, grad = DPLDA_obj_grad(params_trained, params_all, params_mask, params_reference, params_reg, scr_tar,
                   scr_non, Fe, Ft, ids_tar, ids_non, wght_tar, wght_non)
    print("approximate grad.", approx)
    print("exact grad.", grad)
    print("diff", approx - grad)
    return obj, grad


#ids_tar=np.nonzero(((spk_ids[:,None]==spk_ids[None,:]) & np.tri(len(spk_ids), dtype=bool)).flat)[0]
#ids_non=np.nonzero(((spk_ids[:,None]!=spk_ids[None,:]) & np.tri(len(spk_ids), dtype=bool)).flat)[0]
#ids_tar=np.nonzero((spk_ids[:,None]==spk_ids[None,:]).flat)[0]
#ids_non=np.nonzero((spk_ids[:,None]!=spk_ids[None,:]).flat)[0]

def train_DPLDA(Lambda, Gamma, c, beta, Fe, Ft, ids_tar, ids_non, p_tar=0.5,
                alpha=None, scr_tar=None, scr_non=None,
                Lambda_mask=None, Gamma_mask=None, c_mask=None,
                Lambda_reg=0.0,   Gamma_reg=0.0,  c_reg=0.0, relative_reg=False,
                m=10, maxfun=15000):
    wght_tar =    p_tar  * (len(ids_tar)+len(ids_non)) / len(ids_tar)
    wght_non = (1-p_tar) * (len(ids_tar)+len(ids_non)) / len(ids_non)

    cllr_norm=-(p_tar*np.log(p_tar)+(1-p_tar)*np.log(1-p_tar))*(len(ids_tar)+len(ids_non))
    wght_tar /= cllr_norm
    wght_non /= cllr_norm
    
    if Lambda_mask is None:
        Lambda_mask = np.ones_like(Lambda, dtype=bool)
    if Gamma_mask is None:
        Gamma_mask = np.ones_like(Gamma, dtype=bool)
    if c_mask is None:
        c_mask = np.ones_like(c, dtype=bool)

    if scr_tar is not None and scr_tar.ndim == 1:
      scr_tar = scr_tar.reshape(-1,1)
      scr_non = scr_non.reshape(-1,1)

    if scr_tar is None:
        alpha = []
    elif alpha is None:
      alpha = np.ones(scr_tar.shape[1])

    beta = beta +  np.log(p_tar) - np.log(1-p_tar)

    Lambda_mask = np.triu(Lambda_mask).astype(bool)
    Gamma_mask  = np.triu(Gamma_mask).astype(bool)
    Lambda_reg = np.zeros_like(Lambda)+Lambda_reg
    Gamma_reg  = np.zeros_like(Gamma) +Gamma_reg
    c_reg      = np.zeros_like(c)     +c_reg

    params_mask = np.r_[Lambda_mask.flat, Gamma_mask.flat, c_mask, np.ones_like(alpha, dtype=bool), True]
    params_all  = np.r_[Lambda.flat, Gamma.flat, c, alpha, beta]
    params_trained = params_all[params_mask]
    params_reg  = np.r_[Lambda_reg.flat, Gamma_reg.flat, c_reg, np.zeros_like(alpha), 0.0][params_mask]
    params_reference = params_trained.copy() if relative_reg else np.zeros_like(params_trained)

    # pre-allocate workspace for nm1ary_llr_obj_grad function
    #y = np.empty(ntrials)
    xopt, fobj, d = scipy.optimize.fmin_l_bfgs_b(DPLDA_obj_grad,
                      params_trained,
                      args=(params_all, params_mask,
                      params_reference, params_reg, scr_tar, scr_non,
                      Fe, Ft, ids_tar, ids_non, wght_tar, wght_non),
                      m=m, maxfun=maxfun, factr=1e7)
    print("Obj:", fobj)

    # return fusion weights and bias; log prior, which is learned in the bias, is removed; bias is normalized to zero the last element
    Lambda, Gamma, c, alpha, beta = DPLDA_unpack_params(xopt, params_all, params_mask, len(c))
    beta -= np.log(p_tar) - np.log(1-p_tar)
    return Lambda, Gamma, c, alpha, beta

def bilinear_scoring_sideinfo(Lambda, Gamma, c, k, M, nsigmoids, Fe, Ft, out=None):
    if out is None:
        out = np.empty((Fe.shape[0], Ft.shape[0]))
        
    if len(M):
        Se = Fe.dot(M)
        St = Ft.dot(M)
        Se[:,:nsigmoids] = sigmoid(Se[:,:nsigmoids])
        St[:,:nsigmoids] = sigmoid(St[:,:nsigmoids])
    else:
        Se = Fe
        St = Ft
        
    np.dot(Se.dot(Lambda), St.T, out=out)
    tmp = (np.sum(Se.dot(Gamma) * Se, 1) + Se.dot(c))[:,np.newaxis]
    ne.evaluate("out+tmp", out=out) # faster (multithread) version of +=
    tmp = (np.sum(St.dot(Gamma) * St, 1) + St.dot(c))[np.newaxis,:] + k
    ne.evaluate("out+tmp", out=out)
    return out

def DPLDA_sideinfo_unpack_params(params_trained, params_all, params_mask, idim, mdim):
    params_all[params_mask] = params_trained
    Lambda, Gamma, c, M, alpha, beta = np.split(params_all, [idim**2, idim**2*2, idim**2*2+idim,idim**2*2+idim+idim*mdim, -1])
    Lambda = Lambda.reshape(idim,idim)
    Gamma  = Gamma.reshape( idim,idim)
    if mdim: M = M.reshape((mdim,idim))
    Lambda = np.triu(Lambda)+np.triu(Lambda,k=1).T
    Gamma  = np.triu(Gamma) +np.triu(Gamma, k=1).T
    return Lambda, Gamma, c, M, alpha, beta

def DPLDA_sideinfo_obj_grad(params_trained, params_all, params_mask, params_reference, params_reg, scr_tar,
                   scr_non, Fe, Ft, ids_tar, ids_non, wght_tar, wght_non, idim, mdim, nsigmoids):
    tic = time.time()
    Lambda, Gamma, c, M, alpha, beta = DPLDA_sideinfo_unpack_params(params_trained, params_all, params_mask, idim, mdim)

    if mdim:
        Se = Fe.dot(M)
        St = Ft.dot(M)
        Se[:,:nsigmoids] = sigmoid(Se[:,:nsigmoids])
        St[:,:nsigmoids] = sigmoid(St[:,:nsigmoids])
    else:
        Se = Fe
        St = Ft
    
    G = bilinear_scoring(Lambda, Gamma, c, beta, Se, St)
    y_tar = G.take(ids_tar)
    y_non = G.take(ids_non)
    if scr_tar is not None:
        y_tar += np.dot(scr_tar, alpha)
        y_non += np.dot(scr_non, alpha)

    #ne.evaluate("log1p(exp(-y_tar))", out=y_tar)
    #ne.evaluate("log1p(exp(y_non))", out=y_non)
    np.logaddexp(0, -y_tar, out=y_tar) # more stable, but possibly slower and more memory consuming (thanks the minus operator)
    np.logaddexp(0, y_non,  out=y_non)

    #xent = np.sum(y_tar,0) * wght_tar + np.sum(y_non,0) * wght_non
    #xent = ne.evaluate("sum(y_tar,0)") * wght_tar + ne.evaluate("sum(y_non,0)") * wght_non
    xent = np.r_[np.atleast_1d(ne.evaluate("sum(y_tar,0)")), 0.0].sum() * wght_tar + \
           np.r_[np.atleast_1d(ne.evaluate("sum(y_non,0)")), 0.0].sum() * wght_non
    xent = np.sum(y_tar,0) * wght_tar + np.sum(y_non,0) * wght_non
    
    obj = xent + 0.5*params_reg.dot((params_trained-params_reference)**2)

    #print("Obj:", obj, obj-xent, xent, time.time()-tic, "s")
    #return obj

    ne.evaluate("(exp(-y_tar)-1.0) * wght_tar", out=y_tar)
    ne.evaluate("(1.0-exp(-y_non)) * wght_non", out=y_non)

    # Gradient for standard fusion weights and bias
    if scr_tar is not None:
        dalpha = y_tar.dot(scr_tar) + y_non.dot(scr_non)
    else:
        dalpha = []
    dbeta  = y_tar.sum() + y_non.sum()

    # Gradient of the bilinear form
    #G.fill(0.0)
    ne.evaluate("G*0.0", out=G) # faster (mutithread) way of filling matrix with zeros
    G.flat[ids_tar] = y_tar
    G.flat[ids_non] = y_non
    del y_tar, y_non

    # dot product with vector of ones is MUCH (7x) faster then sum
    #Gesum = G.sum(1)[:,np.newaxis]
    #Gtsum = G.sum(0)[:,np.newaxis]
    Gesum = G.dot(np.ones(G.shape[1]))[:,np.newaxis]
    Gtsum = np.ones(G.shape[0]).dot(G)[:,np.newaxis]
    
    if mdim:
        # derivatives of xent objective w.r.t all side-info (Se, St) vectors 
        omegaE = G.dot(St.dot(Lambda))   + Gesum * (Se.dot(2*Gamma)+c)
        omegaT = G.T.dot(Se.dot(Lambda)) + Gtsum * (St.dot(2*Gamma)+c)
        #dM = Fe.T.dot(omegaE*Se*(1.0-Se)) + Ft.T.dot(omegaT*St*(1.0-St))
        #print("Eb", omegaE[0])
        #print("Tb", omegaT[0])
        omegaE[:,:nsigmoids] *= Se[:,:nsigmoids]*(1.0-Se[:,:nsigmoids])
        omegaT[:,:nsigmoids] *= St[:,:nsigmoids]*(1.0-St[:,:nsigmoids])
        #print("Ea", omegaE[0])
        #print("Ta", omegaT[0])
        #print("Se", Se[0], Se[0].shape)
        #print("St", St[0], St[0].shape)
        
        dM = Fe.T.dot(omegaE) + Ft.T.dot(omegaT)
        del omegaE, omegaT
    else:
        dM = np.array([])

    dataEsumG = Se * Gesum
    dataTsumG = St * Gtsum
    dG = 2 * (Se.T.dot(dataEsumG) + dataTsumG.T.dot(St))
    dG -= np.diag(np.diag(dG)) * 0.5
    dL = Se.T.dot(G).dot(St) # expensive G.dot(St) already calculated for omegaE -> possible to optimize
    dL =  (dL + dL.T)
    dL -= np.diag(np.diag(dL)) * 0.5
    dc = dataEsumG.sum(0) + dataTsumG.sum(0)

    print("Obj:", obj, obj-xent, xent, time.time()-tic, "s")
    return obj, np.r_[dL.flat, dG.flat, dc, dM.flat, dalpha, dbeta][params_mask]+(params_trained-params_reference)*params_reg
    
def DPLDA_sideinfo_obj_grad_with_check(params_trained, params_all, params_mask, params_reference, params_reg, scr_tar,
                   scr_non, Fe, Ft, ids_tar, ids_non, wght_tar, wght_non, idim, mdim, nsigmoids):
    approx = scipy.optimize.approx_fprime(
                 params_trained,
                 lambda x: DPLDA_sideinfo_obj_grad(x, params_all, params_mask, params_reference, params_reg, scr_tar, scr_non, Fe, Ft, ids_tar, ids_non, wght_tar, wght_non, idim, mdim, nsigmoids)[0],
                 1e-7)

    obj, grad = DPLDA_sideinfo_obj_grad(params_trained, params_all, params_mask, params_reference, params_reg, scr_tar, scr_non, Fe, Ft, ids_tar, ids_non, wght_tar, wght_non, idim, mdim, nsigmoids)
    print("approximate grad.", approx)
    print("exact grad.", grad)
    print("diff", approx - grad)
    print("obj", obj, grad.dot(grad), np.sum((approx-grad)**2))
    return obj, grad

def train_DPLDA_sideinfo(Lambda, Gamma, c, beta, Fe, Ft, ids_tar, ids_non, 
                p_tar=0.5,
                M=np.array([]), 
                alpha=None, 
                scr_tar=None, 
                scr_non=None,
                Lambda_mask=None, 
                Gamma_mask=None, 
                c_mask=None, 
                M_mask=None,
                Lambda_reg=0.0,
                Gamma_reg=0.0,
                c_reg=0.0,
                M_reg=0.0,
                relative_reg=False,
                nsigmoids=0,
                m=10, 
                maxfun=15000):
    wght_tar =    p_tar  / len(ids_tar)
    wght_non = (1-p_tar) / len(ids_non)

    cllr_norm=-(p_tar*np.log(p_tar)+(1-p_tar)*np.log(1-p_tar))
    wght_tar /= cllr_norm
    wght_non /= cllr_norm
    
    print(wght_tar, wght_non)

    if Lambda_mask is None:
        Lambda_mask = np.ones_like(Lambda, dtype=bool)
    if Gamma_mask is None:
        Gamma_mask = np.ones_like(Gamma, dtype=bool)
    if c_mask is None:
        c_mask = np.ones_like(c, dtype=bool)
    if M_mask is None:
        M_mask = np.ones_like(M, dtype=bool)

    if scr_tar is not None and scr_tar.ndim == 1:
      scr_tar = scr_tar.reshape(-1,1)
      scr_non = scr_non.reshape(-1,1)

    if scr_tar is None:
        alpha = []
    elif alpha is None:
      alpha = np.ones(scr_tar.shape[1])

    beta +=  np.log(p_tar) - np.log(1-p_tar)

    Lambda_mask = np.triu(Lambda_mask).astype(bool)
    Gamma_mask  = np.triu(Gamma_mask).astype(bool)
    Lambda_reg = np.zeros_like(Lambda)+Lambda_reg
    Gamma_reg  = np.zeros_like(Gamma) +Gamma_reg
    c_reg      = np.zeros_like(c)     +c_reg
    M_reg      = np.zeros_like(M)     +M_reg

    params_mask = np.r_[Lambda_mask.flat, Gamma_mask.flat, c_mask, M_mask.flat, np.ones_like(alpha, dtype=bool), True]
    params_all  = np.r_[Lambda.flat, Gamma.flat, c, M.flat, alpha, beta]
    params_trained = params_all[params_mask]
    params_reg  = np.r_[Lambda_reg.flat, Gamma_reg.flat, c_reg, M_reg.flat,  np.zeros_like(alpha), 0.0][params_mask]
    params_reference = params_trained.copy() if relative_reg else np.zeros_like(params_trained)

    # pre-allocate workspace for 
    #y = np.empty(ntrials)
    #embed()
    xopt, fobj, d = scipy.optimize.fmin_l_bfgs_b(DPLDA_sideinfo_obj_grad,
                      params_trained,
                      args=(params_all, params_mask,
                      params_reference, params_reg, scr_tar, scr_non,
                      Fe, Ft, ids_tar, ids_non, wght_tar, wght_non, len(c), len(M), nsigmoids),
                      m=m, maxfun=maxfun, factr=1e1)
    print("Obj:", fobj)

    # return fusion weights and bias; log prior, which is learned in the bias, is removed; bias is normalized to zero the last element
    Lambda, Gamma, c, M, alpha, beta = DPLDA_sideinfo_unpack_params(xopt, params_all, params_mask, len(c), len(M))
    beta -= np.log(p_tar) - np.log(1-p_tar)
    return Lambda, Gamma, c, M, alpha, beta

def get_sideinfo(ids,sideinfo,usecol):
    cls = np.unique([sideinfo[e][usecol] for e in sideinfo])
    convertor = dict(zip(cls,np.arange(len(cls))))
    data = np.zeros((len(ids),len(cls)+1))
    data[:,-1] = 1
    for idx,item in enumerate(ids):
      data[idx,convertor[sideinfo[item][usecol]]] = 1
    return data

# tarids=find(bsxfun(@eq, spk_ids, spk_ids'));
# nonids=find(bsxfun(@ne, spk_ids, spk_ids'));
#
# Z=zeros(size(llrs));
# Z(tarids) =  sigmoid(-llrs(tarids)-logit(tar_prior)) / length(tarids) *    tar_prior;
# Z(nonids) = -sigmoid( llrs(nonids)+logit(tar_prior)) / length(nonids) * (1-tar_prior);
#
# dataEsumZ = bsxfun(@times, dataE', sum(Z, 2));
# dataTsumZ = bsxfun(@times, dataT,  sum(Z, 1));
# dR = dataE * dataEsumZ + dataTsumZ * dataT';
# dS = dataE * Z * dataT';
# dS = dS + dS';
# dt = -2*(sum(dataEsumZ, 1)' + sum(dataTsumZ, 2));
# du = sum(sum(Z));
#
#
# learnrate = 0.05;
# S = S + learnrate * dS;
# R = R + learnrate * dR;
# t = t + learnrate * dt;
# u = u + learnrate * du;
#

# y = mean(neglogsigmoid( llrs(tarids)+logit(tar_prior))) *   tar_prior+...
#     mean(neglogsigmoid(-llrs(nonids)-logit(tar_prior))) * (1-tar_prior);
#
#
# deriv_this(dy, tarids, nonids, dataE, dataT, llrs, tar_prior, param_dim)
#   Z = zeros(size(llrs));
#   Z(tarids) =  sigmoid(-llrs(tarids)-logit(tar_prior)) / length(tarids) *    tar_prior;
#   Z(nonids) = -sigmoid( llrs(nonids)+logit(tar_prior)) / length(nonids) * (1-tar_prior);
#   dataEsumZ = bsxfun(@times, dataE', sum(Z, 2));
#   dataTsumZ = bsxfun(@times, dataT,  sum(Z, 1));
#   dR = dataE * dataEsumZ + dataTsumZ * dataT';
#   dS = dataE * Z * dataT';
#   dS = dS + dS';
#   dt = -2*(sum(dataEsumZ, 1)' + sum(dataTsumZ, 2));
#   du = sum(sum(Z));
#   g0 = -[dR(:); dS(:); dt(:); du];
#   g  = dy*g0;
#
# function [h, Jv] = hessianprod(d, dy, g0, tarids, nonids, dataE, dataT, llrs, tar_prior, param_dim)
#   H=zeros(size(llrs));
#   H(tarids) = -sigmoid(llrs(tarids)+logit(tar_prior)).*sigmoid(-llrs(tarids)-logit(tar_prior))/length(tarids)*tar_prior;
#   H(nonids) = -sigmoid(llrs(nonids)+logit(tar_prior)).*sigmoid(-llrs(nonids)-logit(tar_prior))/length(nonids)*(1-tar_prior);
#   H = H.*scoring(d, param_dim, dataE, dataT);
#   dataEsumH = bsxfun(@times, dataE', sum(H, 2));
#   dataTsumH = bsxfun(@times, dataT,  sum(H, 1));
#   ddR = dataE * dataEsumH + dataTsumH * dataT';
#   ddS = dataE * H * dataT';
#   ddS = ddS + ddS';
#   ddt = -2*(sum(dataEsumH, 1)' + sum(dataTsumH, 2));
#   ddu = sum(sum(H));
#   h0 = -[ddR(:); ddS(:); ddt(:); ddu];
#   h  = dy*h0;
#
#   if nargout>1
#       Jv = d.'*g0;
#   end

##
# TESTS
##


#def nary_nary_logistic_obj_grad(params, data, trial_wght, y, targets_f, grad_const):
#    """ Gradient w.r.t nary logistic regression parameters
#    Input:
#      params     - vector of fusion parameters to be optimized (S alphas, where
#                   S is system s in fusion and C-1 betas, where C is number of
#                   clases)
#      scores     - 3D array of scores (trial, class, system).
#      trial_wght - per trial weight (to compensate for unbalanced
#                   representation of classes in training data)
#      y          - pre-allocated matrix (#trials,C) serving as a workspace
#      targets_f  - vector of true class indeces in flattened score matrix
#      grad_const - constant term in gradient corresponding to subtracting 1
#                   #from posteriors of target class
#
#    Output:
#      fobj       - value of objective function at alpha_beta
#      g          - gradient vector at alpha_beta
#    """
#
#
#    # apply fusion to get log likelohhos and normalize to log class posteriors
#    ndim = scores.shape[2]
#    nclasses = y.shape[2]
#    A = params[:ndim*nclasses].shape(ndim*nclasses)
#    b = params[A.size:]
#
#
#    np.dot(data, A, out=y)
#    y += b
#
#    y -= utils.logsumexp(y, axis=1)[:, np.newaxis]
#
#    # evaluate cross entropy objective
#    obj = -trial_wght[:,0].dot(y.take(targets_f))
#
#    # Let t be 1-to-N encoding of class labels
#    # grad_A = sum_n trial_weight[n] * (y[n] - t[n])' * data[n]
#    # grad_b = sum_n trial_weight[n] * (y[n] - t[n])
#    # t[n] is removed in the calculations below and the cooresponding
#    # constant terms are precomputed in grad_const
#    ne.evaluate("exp(y)*trial_wght", out=y)
#    grad_A = data.dot.y.T
#    return obj, np.r_[grad_A.ravel(), y.sum(0)[:-1]] + grad_const
#
#
##@profileit
#def train_nary_logistic_regression(data, classf, priors=None, A0=None, b0=None):
#    """ Train nary logistic regression fusion parameters
#    Input:
#      scores - 3D array of scores (trial, class, system)
#      classf - true class labels (0..C-1)
#      priors - prior probability of classes
#      alpha0 - optional: initial values for combination weights
#      beta0  - optional: initial values for of score offsets parameters
#
#    Output:
#      alpha  - vector of system combination weights
#      beta   - vector of score offsets
#    """
#    nclasses = np.max(classf)+1
#    ntrials, ndim = scores.shape
#    priors = np.ones(nclasses) if priors is  None else np.array(priors,float)
#    priors /= priors.sum()
#
#    if A0 is None:
#        A0 = np.ones((ndim,nclasses))
#    if beta0 is None:
#        beta0 = np.zeros(nclasses)
#
#    beta0 += np.log(priors) - np.log(priors[-1])
#
#    class_freq = np.sum(np.arange(nclasses)[:,np.newaxis] == classf, axis=1)
#    trial_wght = (priors * ntrials / class_freq)[classf]
#
#    # pre-allocate workspace for nary_llr_obj_grad function
#    y = np.empty((ntrials, nclasses))
#
#    # create vector of true class indeces in flattened score matrix
#    targets_f = np.ravel_multi_index((np.arange(ntrials), classf), y.shape)
#
#    # precompute a constant term in gradient callculation corresponding
#    # to subtracting 1 from posteriors of target class
##    grad_const = -np.r_[trial_wght.dot(data[np.arange(ntrials), classf,:]), priors[:-1] * ntrials]
#
#    xopt, fobj, d = scipy.optimize.fmin_l_bfgs_b(nary_logistic_regression_obj_grad,
#                      np.r_[A0.ravel(), beta0[:-1] - beta0[-1]],
#                      args=(data, trial_wght[:,np.newaxis], y, targets_f, grad_const))
#
#    # return fusion weights and bias; log prior, which is learned in the bias,
#    # is removed; bias is normalized to zero the last element
#    return (xopt[:A0.size].shape(A0.shape),
#            np.r_[xopt[A0.size:], 0] - np.log(priors) + np.log(priors[-1]))


def test_nary_llr_fusion():
    """ Test multiclass logistic regression based fusion"""

    ntrials, nclasses, nsystems = 100000, 3, 2
    alpha = np.array([-3, 8])
    beta = np.arange(nclasses - 1, -1, -1)
    classf = np.repeat(range(nclasses), ntrials)

    # Generate data from 3 models and 2 system (gaussian distributiohs with
    # unity variance different means) and evaluated the data using the same
    # models to obtain well calibrated scores
    class_means = np.arange(nclasses)
    d = (np.random.randn(ntrials * nsystems, nclasses) + class_means).T
    scores = -0.5 * (np.subtract.outer(d.reshape(1, -1), class_means) ** 2
                    ).reshape(-1, nsystems, nclasses)

    # Uncalibrate the scores by inverse application of fusion weights and bias
    print("Expected weights:", alpha, beta)
    scores = np.diag(1.0 / alpha).dot(scores - beta / 2.0).transpose(1,2,0)

    # Learn the fusion weights and bias from the uncalibrated scores
    alpha, beta = train_nary_llr_fusion(scores, classf)
    print("Obtained weights:", alpha, beta)

    alpha, beta = train_nm1ary_llr_fusion((scores-scores[:,[-1],:])[:,:-1,:], classf)
    print("Obtained weights:", alpha, beta)

if(__name__ == "__main__"):
    test_nary_llr_fusion()
