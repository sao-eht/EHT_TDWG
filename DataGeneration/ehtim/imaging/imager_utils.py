# imager_utils.py
# Andrew Chael, 3/11/2017
# General imager for total intensity VLBI data

#TODO
# add more general linearized energy functions
# closure amplitude and phase covariance

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import range

import string
import time
import numpy as np
import scipy.optimize as opt
import scipy.ndimage as nd
import scipy.ndimage.filters as filt
import matplotlib.pyplot as plt

import ehtim.image as image
from . import linearize_energy as le

from ehtim.const_def import *
from ehtim.observing.obs_helpers import *

from IPython import display

##################################################################################################
# Constants & Definitions
##################################################################################################


NHIST = 50 # number of steps to store for hessian approx
MAXIT = 100

DATATERMS = ['vis', 'bs', 'amp', 'cphase', 'camp', 'logcamp']
REGULARIZERS = ['gs', 'tv', 'tv2','l1', 'patch', 'simple']

GRIDDER_P_RAD_DEFAULT = 2.0
GRIDDER_CONV_FUNC_DEFAULT = 'gaussian'
FFT_PAD_DEFAULT = 2
FFT_INTERP_DEFAULT = 3

nit = 0 # global variable to track the iteration number in the plotting callback

##################################################################################################
# Total Intensity Imager
##################################################################################################
def imager_func(Obsdata, InitIm, Prior, flux,
           d1='vis', d2=False, s1='simple', s2=False,
           alpha_s1=1, alpha_s2=1,
           alpha_d1=100, alpha_d2=100,
           alpha_flux=500, alpha_cm=500,
           ttype='direct', 
           fft_pad_frac=FFT_PAD_DEFAULT, fft_interp=FFT_INTERP_DEFAULT,
           grid_prad=GRIDDER_P_RAD_DEFAULT, grid_conv=GRIDDER_CONV_FUNC_DEFAULT,
           clipfloor=0., datamin="gd", grads=True, logim=True,
           maxit=MAXIT, stop=1e-10, ipynb=False, show_updates=True):

    """Run a general interferometric imager.

       Args:
           Obsdata (Obsdata): The Obsdata object with VLBI data
           Prior (Image): The Image object with the prior image
           InitIm (Image): The Image object with the initial image for the minimization
           flux (float): The total flux of the output image in Jy
           d1 (str): The first data term; options are 'vis', 'bs', 'amp', 'cphase', 'camp', 'logcamp'
           d2 (str): The second data term; options are 'vis', 'bs', 'amp', 'cphase', 'camp', 'logcamp'
           s1 (str): The first regularizer; options are 'simple', 'gs', 'tv', 'tv2', 'l1', 'patch'
           s2 (str): The second regularizer; options are 'simple', 'gs', 'tv', 'tv2','l1', 'patch'
           alpha_d1 (float): The first data term weighting
           alpha_d2 (float): The second data term weighting
           alpha_s1 (float): The first regularizer term weighting
           alpha_s2 (float): The second regularizer term weighting
           alpha_flux (float): The weighting for the total flux constraint
           alpha_cm (float): The weighting for the center of mass constraint

           ttype (str): The Fourier transform type; options are 'fast' and 'direct'
           fft_pad_frac (float): The FFT will pre-pad the image by this factor x the original size
           fft_interp (int): Interpolation order for sampling the FFT
           grid_conv_func (str): The convolving function for gridding; options are 'gaussian', 'pill', and 'spheroidal'
           grid_prad (float): The pixel radius for the convolving function in gridding for FFTs

           clipfloor (float): The Jy/pixel level above which prior image pixels are varied
           datamin (str): If 'lin', linearized energy is used (currently only compatible with CHIRP)
           grads (bool): If True, analytic gradients are used
           logim (bool): If True, uses I = exp(I') change of variables

           maxit (int): Maximum number of minimizer iterations
           stop (float): The convergence criterion
           show_updates (bool): If True, displays the progress of the minimizer
           ipynb (bool): If True, adjusts the plotting for the ipython/jupyter notebook

       Returns:
           Image: Image object with result
    """

    print ("Imaging observation with %s Fourier transform" % ttype)
    print ("Data terms: %s , %s" %  (d1,d2))
    print ("Regularizer terms: %s, %s\n" % (s1,s2))

    # Make sure data and regularizer options are ok
    if not d1 and not d2:
        raise Exception("Must have at least one data term!")

    if not s1 and not s2:
        raise Exception("Must have at least one regularizer term!")

    if (not ((d1 in DATATERMS) or d1==False)) or (not ((d2 in DATATERMS) or d2==False)):
        raise Exception("Invalid data term: valid data terms are: " + ' '.join(DATATERMS))

    if (not ((s1 in REGULARIZERS) or s1==False)) or (not ((s2 in REGULARIZERS) or s2==False)):
        raise Exception("Invalid regularizer: valid regularizers are: " + ' '.join(REGULARIZERS))

    if (Prior.psize != InitIm.psize) or (Prior.xdim != InitIm.xdim) or (Prior.ydim != InitIm.ydim):
        raise Exception("Initial image does not match dimensions of the prior image!")

    # Catch scale and dimension problems
    imsize = np.max([Prior.xdim, Prior.ydim]) * Prior.psize
    uvmax = 1.0/Prior.psize
    uvmin = 1.0/imsize
    uvdists = Obsdata.unpack('uvdist')['uvdist']
    maxbl = np.max(uvdists)
    minbl = np.max(uvdists[uvdists > 0])
    maxamp = np.max(np.abs(Obsdata.unpack('amp')['amp']))

    if uvmax < maxbl:
        print("Warning! Pixel Spacing is larger than smallest spatial wavelength!")
    if uvmin > minbl:
        print("Warning! Field of View is smaller than largest nonzero spatial wavelength!")
    if flux > 1.2*maxamp:
        print("Warning! Specified flux is > 120% of maximum visibility amplitude!")
    if flux < .8*maxamp:
        print("Warning! Specified flux is < 80% of maximum visibility amplitude!")

    # Normalize prior image to total flux and limit imager range to prior values > clipfloor
    embed_mask = Prior.imvec > clipfloor
    nprior = (flux * Prior.imvec / np.sum((Prior.imvec)[embed_mask]))[embed_mask]
    ninit = (flux * InitIm.imvec / np.sum((InitIm.imvec)[embed_mask]))[embed_mask]

    # Get data and fourier matrices for the data terms
    (data1, sigma1, A1) = chisqdata(Obsdata, Prior, embed_mask, d1, ttype=ttype, fft_pad_frac=fft_pad_frac)
    (data2, sigma2, A2) = chisqdata(Obsdata, Prior, embed_mask, d2, ttype=ttype, fft_pad_frac=fft_pad_frac)

    # Coordinate matrix for center-of-mass constraint
    coord = Prior.psize * np.array([[[x,y] for x in np.arange(Prior.xdim//2,-Prior.xdim//2,-1)]
                                           for y in np.arange(Prior.ydim//2,-Prior.ydim//2,-1)])
    coord = coord.reshape(Prior.ydim*Prior.xdim, 2)
    coord = coord[embed_mask]

    # Katie - if you are using the linearized energy then compute the A and b of your
    # linearized equation given your current image
    # TODO generalize this linearization for the other data terms!
    if d1=='bs' and datamin=='lin':
        (Alin1, blin1) = le.computeLinTerms_bi(ninit, A1, data1, sigma1,
                               len(ninit), alpha=alpha_d1)
    if d2=='bs' and datamin=='lin':
        (Alin2, blin2) = le.computeLinTerms_bi(ninit, A2, data2, sigma2,
                               len(ninit), alpha=alpha_d2)


    # Define the chi^2 and chi^2 gradient
    def chisq1(imvec):
        return chisq(imvec, A1, data1, sigma1, d1, ttype=ttype, mask=embed_mask, fft_interp=fft_interp)

    def chisq1grad(imvec):
        if d1=='bs' and datamin=='lin':
            c = 2.0/(2.0*len(sigma1)) * np.dot(Alin1.T, np.dot(Alin1, imvec) - blin1)
        else:
            c = chisqgrad(imvec, A1, data1, sigma1, d1, ttype=ttype, mask=embed_mask,
                                fft_interp=fft_interp, grid_prad=grid_prad, grid_conv=grid_conv)
        return c

    def chisq2(imvec):
        return chisq(imvec, A2, data2, sigma2, d2, ttype=ttype, mask=embed_mask, fft_interp=fft_interp)


    def chisq2grad(imvec):
        if d2=='bs' and datamin=='lin':
            c = 2.0/(2.0*len(sigma2)) * np.dot(Alin2.T, np.dot(Alin2, imvec) - blin2)
        else:
            c = chisqgrad(imvec, A2, data2, sigma2, d2, ttype=ttype, mask=embed_mask,
                          fft_interp=fft_interp, grid_prad=grid_prad, grid_conv=grid_conv)
        return c

    # Define the regularizer and regularizer gradient
    def reg1(imvec):
        return regularizer(imvec, nprior, embed_mask, flux, Prior.xdim, Prior.ydim, Prior.psize, s1)

    def reg1grad(imvec):
        return regularizergrad(imvec, nprior, embed_mask, flux, Prior.xdim, Prior.ydim, Prior.psize, s1)

    def reg2(imvec):
        return regularizer(imvec, nprior, embed_mask, flux, Prior.xdim, Prior.ydim, Prior.psize, s2)

    def reg2grad(imvec):
        return regularizergrad(imvec, nprior, embed_mask, flux, Prior.xdim, Prior.ydim, Prior.psize, s2)

    # Define constraint functions
    def flux_constraint(imvec):
        #norm = flux**2
        norm = 1
        return (np.sum(imvec) - flux)**2/norm

    def flux_constraint_grad(imvec):
        #norm = flux**2
        norm = 1
        return 2*(np.sum(imvec) - flux) / norm

    def cm_constraint(imvec):
        #norm = flux**2 * Prior.psize**2
        norm = 1
        return (np.sum(imvec*coord[:,0])**2 + np.sum(imvec*coord[:,1])**2)/norm

    def cm_constraint_grad(imvec):
        #norm = flux**2 * Prior.psize**2
        norm = 1
        return 2*(np.sum(imvec*coord[:,0])*coord[:,0] + np.sum(imvec*coord[:,1])*coord[:,1]) / norm

    # Define the objective function and gradient
    def objfunc(imvec):
        if logim: imvec = np.exp(imvec)

        datterm  = alpha_d1 * (chisq1(imvec) - 1) + alpha_d2 * (chisq2(imvec) - 1)
        regterm  = alpha_s1 * reg1(imvec) + alpha_s2 * reg2(imvec)
        conterm  = alpha_flux * flux_constraint(imvec)  + alpha_cm * cm_constraint(imvec)

        return datterm + regterm + conterm

    def objgrad(imvec):
        if logim: imvec = np.exp(imvec)

        datterm  = alpha_d1 * chisq1grad(imvec) + alpha_d2 * chisq2grad(imvec)
        regterm  = alpha_s1 * reg1grad(imvec) + alpha_s2 * reg2grad(imvec)
        conterm  = alpha_flux * flux_constraint_grad(imvec)  + alpha_cm * cm_constraint_grad(imvec)

        grad = datterm + regterm + conterm

        # chain rule term for change of variables
        if logim: grad *= imvec

        return grad

    # Define plotting function for each iteration
    global nit
    nit = 0
    def plotcur(im_step):
        global nit
        if logim: im_step = np.exp(im_step)
        if show_updates:
            chi2_1 = chisq1(im_step)
            chi2_2 = chisq2(im_step)
            s_1 = reg1(im_step)
            s_2 = reg2(im_step)
            #fluxreg = flux_constraint(im_step)
            #cmreg = cm_constraint(im_step)
            if np.any(np.invert(embed_mask)): im_step = embed(im_step, embed_mask)
            plot_i(im_step, Prior, nit, chi2_1, chi2_2, ipynb=ipynb)
            print("i: %d chi2_1-1: %0.2f chi2_2-1: %0.2f s_1: %0.2f s_2: %0.2f" % (nit, chi2_1-1, chi2_2-1,s_1,s_2))
        nit += 1

    # Generate and the initial image
    if logim:
        xinit = np.log(ninit)
    else:
        xinit = ninit


    # Print stats
    print("Initial Chi^2_1: %f Chi^2_2: %f" % (chisq1(ninit), chisq2(ninit)))
    print("Total Pixel #: ",(len(Prior.imvec)))
    print("Clipped Pixel #: ",(len(ninit)))
    print()
    plotcur(xinit)

    # Minimize
    optdict = {'maxiter':maxit, 'ftol':stop, 'maxcor':NHIST} # minimizer dict params
    tstart = time.time()
    if grads:
        res = opt.minimize(objfunc, xinit, method='L-BFGS-B', jac=objgrad,
                       options=optdict, callback=plotcur)
    else:
        res = opt.minimize(objfunc, xinit, method='L-BFGS-B',
                       options=optdict, callback=plotcur)

    tstop = time.time()

    # Format output
    out = res.x
    if logim: out = np.exp(res.x)
    if np.any(np.invert(embed_mask)): out = embed(out, embed_mask)

    outim = image.Image(out.reshape(Prior.ydim, Prior.xdim), Prior.psize,
                     Prior.ra, Prior.dec, rf=Prior.rf, source=Prior.source,
                     mjd=Prior.mjd, pulse=Prior.pulse)

    if len(Prior.qvec):
        print("Preserving image complex polarization fractions!")
        qvec = Prior.qvec * out / Prior.imvec
        uvec = Prior.uvec * out / Prior.imvec
        outim.add_qu(qvec.reshape(Prior.ydim, Prior.xdim), uvec.reshape(Prior.ydim, Prior.xdim))

    # Print stats
    print("time: %f s" % (tstop - tstart))
    print("J: %f" % res.fun)
    print("Final Chi^2_1: %f Chi^2_2: %f" % (chisq1(out[embed_mask]), chisq2(out[embed_mask])))
    print(res.message)

    # Return Image object
    return outim

##################################################################################################
# Wrapper Functions
##################################################################################################

def chisq(imvec, A, data, sigma, dtype, ttype='direct', mask=[], fft_interp=FFT_INTERP_DEFAULT):
    """return the chi^2 for the appropriate dtype
    """

    chisq = 1 
    if not dtype in DATATERMS:
        return chisq

    if ttype == 'direct':
        if dtype == 'vis':
            chisq = chisq_vis(imvec, A, data, sigma)

        elif dtype == 'amp':
            chisq = chisq_amp(imvec, A, data, sigma)

        elif dtype == 'bs':
            chisq = chisq_bs(imvec, A, data, sigma)

        elif dtype == 'cphase':
            chisq = chisq_cphase(imvec, A, data, sigma)

        elif dtype == 'camp':
            chisq = chisq_camp(imvec, A, data, sigma)

        elif dtype == 'logcamp':
            chisq = chisq_logcamp(imvec, A, data, sigma)
    
    elif ttype== 'fast':
        if len(mask)>0 and np.any(np.invert(mask)):
            imvec = embed(imvec, mask, randomfloor=True)
        vis_arr = fft_imvec(imvec, A[0])

        vis_arr = fft_imvec(imvec, A[0])
        if dtype == 'vis':            
            chisq = chisq_vis_fft(vis_arr, A, data, sigma, order=fft_interp)
        elif dtype == 'amp':            
            chisq = chisq_amp_fft(vis_arr, A, data, sigma, order=fft_interp)
        elif dtype == 'bs':            
            chisq = chisq_bs_fft(vis_arr, A, data, sigma, order=fft_interp)
        elif dtype == 'cphase':            
            chisq = chisq_cphase_fft(vis_arr, A, data, sigma, order=fft_interp)
        elif dtype == 'camp':            
            chisq = chisq_camp_fft(vis_arr, A, data, sigma, order=fft_interp)
        elif dtype == 'logcamp':            
            chisq = chisq_logcamp_fft(vis_arr, A, data, sigma, order=fft_interp)

    return chisq

def chisqgrad(imvec, A, data, sigma, dtype, ttype='direct', mask=[],
              fft_interp=FFT_INTERP_DEFAULT, grid_prad=GRIDDER_P_RAD_DEFAULT, grid_conv=GRIDDER_CONV_FUNC_DEFAULT):
    
    """return the chi^2 gradient for the appropriate dtype
    """

    chisqgrad = np.zeros(len(imvec))
    if not dtype in DATATERMS:
        return chisqgrad

    if ttype == 'direct':
        if dtype == 'vis':
            chisqgrad = chisqgrad_vis(imvec, A, data, sigma)

        elif dtype == 'amp':
            chisqgrad = chisqgrad_amp(imvec, A, data, sigma)

        elif dtype == 'bs':
            chisqgrad = chisqgrad_bs(imvec, A, data, sigma)

        elif dtype == 'cphase':
            chisqgrad = chisqgrad_cphase(imvec, A, data, sigma)

        elif dtype == 'camp':
            chisqgrad = chisqgrad_camp(imvec, A, data, sigma)

        elif dtype == 'logcamp':
            chisqgrad = chisqgrad_logcamp(imvec, A, data, sigma)

    elif ttype== 'fast':
        if len(mask)>0 and np.any(np.invert(mask)):
            imvec = embed(imvec, mask, randomfloor=True)
        vis_arr = fft_imvec(imvec, A[0])

        if dtype == 'vis':                        
            chisqgrad = chisqgrad_vis_fft(vis_arr, A, data, sigma, order=fft_interp, conv_func=grid_conv, p_rad=grid_prad)
        elif dtype == 'amp':            
            chisqgrad = chisqgrad_amp_fft(vis_arr, A, data, sigma, order=fft_interp, conv_func=grid_conv, p_rad=grid_prad)
        elif dtype == 'bs':            
            chisqgrad = chisqgrad_bs_fft(vis_arr, A, data, sigma, order=fft_interp, conv_func=grid_conv, p_rad=grid_prad)
        elif dtype == 'cphase':            
            chisqgrad = chisqgrad_cphase_fft(vis_arr, A, data, sigma, order=fft_interp, conv_func=grid_conv, p_rad=grid_prad)
        elif dtype == 'camp':            
            chisqgrad = chisqgrad_camp_fft(vis_arr, A, data, sigma, order=fft_interp, conv_func=grid_conv, p_rad=grid_prad)
        elif dtype == 'logcamp':            
            chisqgrad = chisqgrad_logcamp_fft(vis_arr, A, data, sigma, order=fft_interp, conv_func=grid_conv, p_rad=grid_prad)
        
        if len(mask)>0 and np.any(np.invert(mask)):
            chisqgrad = chisqgrad[mask]

    return chisqgrad


def regularizer(imvec, nprior, mask, flux, xdim, ydim, psize, stype):
    if stype == "simple":
        s = -ssimple(imvec, nprior, flux)
    elif stype == "l1":
        s = -sl1(imvec, nprior, flux)
    elif stype == "gs":
        s = -sgs(imvec, nprior, flux)
    elif stype == "patch":
        s = -spatch(imvec, nprior, flux)

    elif stype == "tv":
        if np.any(np.invert(mask)):
            imvec = embed(imvec, mask, randomfloor=True)
        s = -stv(imvec, xdim, ydim, flux)
    elif stype == "tv2":
        if np.any(np.invert(mask)):
            imvec = embed(imvec, mask, randomfloor=True)
        s = -stv2(imvec, xdim, ydim, flux)
    else:
        s = 0

    return s

def regularizergrad(imvec, nprior, mask, flux, xdim, ydim, psize, stype):

    if stype == "simple":
        s = -ssimplegrad(imvec, nprior, flux)
    elif stype == "l1":
        s = -sl1grad(imvec, nprior, flux)
    elif stype == "gs":
        s = -sgsgrad(imvec, nprior, flux)
    elif stype == "patch":
        s = -spatchgrad(imvec, nprior, flux)

    elif stype == "tv":
        if np.any(np.invert(mask)):
            imvec = embed(imvec, mask, randomfloor=True)
        s = -stvgrad(imvec, xdim, ydim, flux)[mask]
    elif stype == "tv2":
        if np.any(np.invert(mask)):
            imvec = embed(imvec, mask, randomfloor=True)
        s = -stv2grad(imvec, xdim, ydim, flux)[mask]
    else:
        s = np.zeros(len(imvec))

    return s

def chisqdata(Obsdata, Prior, mask, dtype, ttype='direct', fft_pad_frac=1):
    """Return the data, sigma, and matrices for the appropriate dtype
    """

    (data, sigma, A) = (False, False, False)

    if ttype=='direct':
        if dtype == 'vis':
            (data, sigma, A) = chisqdata_vis(Obsdata, Prior, mask)
        elif dtype == 'amp':
            (data, sigma, A) = chisqdata_amp(Obsdata, Prior, mask)
        elif dtype == 'bs':
            (data, sigma, A) = chisqdata_bs(Obsdata, Prior, mask)
        elif dtype == 'cphase':
            (data, sigma, A) = chisqdata_cphase(Obsdata, Prior, mask)
        elif dtype == 'camp':
            (data, sigma, A) = chisqdata_camp(Obsdata, Prior, mask)
        elif dtype == 'logcamp':
            (data, sigma, A) = chisqdata_logcamp(Obsdata, Prior, mask)

    elif ttype=='fast':
        if dtype=='vis':
            (data, sigma, A) = chisqdata_vis_fft(Obsdata, Prior, fft_pad_frac=fft_pad_frac)
        elif dtype == 'amp':
            (data, sigma, A) = chisqdata_amp_fft(Obsdata, Prior, fft_pad_frac=fft_pad_frac)
        elif dtype == 'bs':
            (data, sigma, A) = chisqdata_bs_fft(Obsdata, Prior, fft_pad_frac=fft_pad_frac)
        elif dtype == 'cphase':
            (data, sigma, A) = chisqdata_cphase_fft(Obsdata, Prior, fft_pad_frac=fft_pad_frac)
        elif dtype == 'camp':
            (data, sigma, A) = chisqdata_camp_fft(Obsdata, Prior, fft_pad_frac=fft_pad_frac)
        elif dtype == 'logcamp':
            (data, sigma, A) = chisqdata_logcamp_fft(Obsdata, Prior, fft_pad_frac=fft_pad_frac)

    return (data, sigma, A)

##################################################################################################
# DFT Chi-squared and Gradient Functions
##################################################################################################

def chisq_vis(imvec, Amatrix, vis, sigma):
    """Visibility chi-squared"""

    samples = np.dot(Amatrix, imvec)
    return np.sum(np.abs((samples-vis)/sigma)**2)/(2*len(vis))

def chisqgrad_vis(imvec, Amatrix, vis, sigma):
    """The gradient of the visibility chi-squared"""

    samples = np.dot(Amatrix, imvec)
    wdiff = (vis - samples)/(sigma**2)

    out = -np.real(np.dot(Amatrix.conj().T, wdiff))/len(vis)
    return out

def chisq_amp(imvec, A, amp, sigma):
    """Visibility Amplitudes (normalized) chi-squared"""

    amp_samples = np.abs(np.dot(A, imvec))
    return np.sum(np.abs((amp - amp_samples)/sigma)**2)/len(amp)

def chisqgrad_amp(imvec, A, amp, sigma):
    """The gradient of the amplitude chi-squared"""

    i1 = np.dot(A, imvec)
    amp_samples = np.abs(i1)

    pp = ((amp - amp_samples) * amp_samples) / (sigma**2) / i1
    out = (-2.0/len(amp)) * np.real(np.dot(pp, A))
    return out

def chisq_bs(imvec, Amatrices, bis, sigma):
    """Bispectrum chi-squared"""

    bisamples = np.dot(Amatrices[0], imvec) * np.dot(Amatrices[1], imvec) * np.dot(Amatrices[2], imvec)
    return np.sum(np.abs(((bis - bisamples)/sigma))**2)/(2.*len(bis))

def chisqgrad_bs(imvec, Amatrices, bis, sigma):
    """The gradient of the bispectrum chi-squared"""

    bisamples = np.dot(Amatrices[0], imvec) * np.dot(Amatrices[1], imvec) * np.dot(Amatrices[2], imvec)
    wdiff = ((bis - bisamples).conj())/(sigma**2)
    pt1 = wdiff * np.dot(Amatrices[1],imvec) * np.dot(Amatrices[2],imvec)
    pt2 = wdiff * np.dot(Amatrices[0],imvec) * np.dot(Amatrices[2],imvec)
    pt3 = wdiff * np.dot(Amatrices[0],imvec) * np.dot(Amatrices[1],imvec)
    out = -np.real(np.dot(pt1, Amatrices[0]) + np.dot(pt2, Amatrices[1]) + np.dot(pt3, Amatrices[2]))/len(bis)
    return out

def chisq_cphase(imvec, Amatrices, clphase, sigma):
    """Closure Phases (normalized) chi-squared"""
    clphase = clphase * DEGREE
    sigma = sigma * DEGREE
    clphase_samples = np.angle(np.dot(Amatrices[0], imvec) * np.dot(Amatrices[1], imvec) * np.dot(Amatrices[2], imvec))
    return (2.0/len(clphase)) * np.sum((1.0 - np.cos(clphase-clphase_samples))/(sigma**2))


def chisqgrad_cphase(imvec, Amatrices, clphase, sigma):
    """The gradient of the closure phase chi-squared"""
    clphase = clphase * DEGREE
    sigma = sigma * DEGREE

    i1 = np.dot(Amatrices[0], imvec)
    i2 = np.dot(Amatrices[1], imvec)
    i3 = np.dot(Amatrices[2], imvec)
    clphase_samples = np.angle(i1 * i2 * i3)

    pref = np.sin(clphase - clphase_samples)/(sigma**2)
    pt1  = pref/i1
    pt2  = pref/i2
    pt3  = pref/i3
    out  = -(2.0/len(clphase)) * np.imag(np.dot(pt1, Amatrices[0]) + np.dot(pt2, Amatrices[1]) + np.dot(pt3, Amatrices[2]))
    return out

def chisq_camp(imvec, Amatrices, clamp, sigma):
    """Closure Amplitudes (normalized) chi-squared"""

    clamp_samples = np.abs(np.dot(Amatrices[0], imvec) * np.dot(Amatrices[1], imvec) / (np.dot(Amatrices[2], imvec) * np.dot(Amatrices[3], imvec)))
    return np.sum(np.abs((clamp - clamp_samples)/sigma)**2)/len(clamp)

def chisqgrad_camp(imvec, Amatrices, clamp, sigma):
    """The gradient of the closure amplitude chi-squared"""

    i1 = np.dot(Amatrices[0], imvec)
    i2 = np.dot(Amatrices[1], imvec)
    i3 = np.dot(Amatrices[2], imvec)
    i4 = np.dot(Amatrices[3], imvec)
    clamp_samples = np.abs((i1 * i2)/(i3 * i4))

    pp = ((clamp - clamp_samples) * clamp_samples)/(sigma**2)
    pt1 =  pp/i1
    pt2 =  pp/i2
    pt3 = -pp/i3
    pt4 = -pp/i4
    out = (-2.0/len(clamp)) * np.real(np.dot(pt1, Amatrices[0]) + np.dot(pt2, Amatrices[1]) + np.dot(pt3, Amatrices[2]) + np.dot(pt4, Amatrices[3]))
    return out

def chisq_logcamp(imvec, Amatrices, log_clamp, sigma):
    """Log Closure Amplitudes (normalized) chi-squared"""

    a1 = np.abs(np.dot(Amatrices[0], imvec))
    a2 = np.abs(np.dot(Amatrices[1], imvec))
    a3 = np.abs(np.dot(Amatrices[2], imvec))
    a4 = np.abs(np.dot(Amatrices[3], imvec))
    samples = np.log(a1) + np.log(a2) - np.log(a3) - np.log(a4)

    return np.sum(np.abs((log_clamp - samples)/sigma)**2) / (len(log_clamp))

def chisqgrad_logcamp(imvec, Amatrices, log_clamp, sigma):
    """The gradient of the Log closure amplitude chi-squared"""

    i1 = np.dot(Amatrices[0], imvec)
    i2 = np.dot(Amatrices[1], imvec)
    i3 = np.dot(Amatrices[2], imvec)
    i4 = np.dot(Amatrices[3], imvec)
    log_clamp_samples = np.log(np.abs(i1)) + np.log(np.abs(i2)) - np.log(np.abs(i3)) - np.log(np.abs(i4))

    pp = (log_clamp - log_clamp_samples) / (sigma**2)
    pt1 = pp / i1
    pt2 = pp / i2
    pt3 = -pp / i3
    pt4 = -pp / i4
    out = (-2.0/len(log_clamp)) * np.real(np.dot(pt1, Amatrices[0]) + np.dot(pt2, Amatrices[1]) + np.dot(pt3, Amatrices[2]) + np.dot(pt4, Amatrices[3]))
    return out

##################################################################################################
# FFT Chi-squared and Gradient Functions
##################################################################################################

def chisq_vis_fft(vis_arr, A, vis, sigma, order=FFT_INTERP_DEFAULT):
    """Visibility chi-squared from fft
    """
    im_info, uv = A # to maintain compatibility with chisq()
                    # im_info = (im.xdim, im.ydim, npad, im.psize, im.pulse) 

    samples = sampler(vis_arr, im_info, [uv], order=order, sample_type="vis")

    chisq = np.sum(np.abs((samples-vis)/sigma)**2)/(2*len(vis))
    return chisq

def chisqgrad_vis_fft(vis_arr, A, vis, sigma, 
                      order=FFT_INTERP_DEFAULT, conv_func=GRIDDER_CONV_FUNC_DEFAULT, p_rad=GRIDDER_P_RAD_DEFAULT):

    """The gradient of the visibility chi-squared from fft
    """

    im_info, uv = A
    (xdim, ydim, npad, psize, pulse) = im_info

    # parameters to unpad the array  
    padvalx1 = padvalx2 = int(np.floor((npad - xdim)/2.0))
    if xdim % 2:
        padvalx2 += 1
    padvaly1 = padvaly2 = int(np.floor((npad - ydim)/2.0))
    if ydim % 2:
        padvaly2 += 1

    # grid phase and pulse factor
    phase = np.exp(-1j*np.pi*psize*(uv[:,0] + uv[:,1]))
    pulsefac = np.array([pulse(2*np.pi*uvpt[0], 2*np.pi*uvpt[1], psize, dom="F") for uvpt in uv])

    # samples and gradient FT
    samples = sampler(vis_arr, im_info, [uv], order=order, sample_type="vis")
    wdiff_vec = -1.0/len(vis)*(vis - samples)/(sigma**2)

    # Setup and perform the inverse FFT
    wdiff_arr = gridder(wdiff_vec * phase.conj() * pulsefac.conj(), im_info, uv, conv_func=conv_func, p_rad=p_rad)    
    grad_arr = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(wdiff_arr))) * npad * npad

    # extract relevant cells and flatten
    out = np.real(grad_arr[padvalx1:-padvalx2,padvaly1:-padvaly2].flatten()) # TODO or is x<-->y??

    return out

def chisq_amp_fft(vis_arr, A, amp, sigma, order=FFT_INTERP_DEFAULT):
    """Visibility amplitude chi-squared from fft
    """
   
    im_info, uv = A 
    amp_samples = np.abs(sampler(vis_arr, im_info, [uv], order=order, sample_type="vis"))
    chisq = np.sum(np.abs((amp_samples-amp)/sigma)**2)/(len(amp))
    return chisq

def chisqgrad_amp_fft(vis_arr, A, amp, sigma,
                      order=FFT_INTERP_DEFAULT, conv_func=GRIDDER_CONV_FUNC_DEFAULT, p_rad=GRIDDER_P_RAD_DEFAULT):

    """The gradient of the amplitude chi-squared
    """

    im_info, uv = A 
    (xdim, ydim, npad, psize, pulse) = im_info

    # parameters to unpad the array  
    padvalx1 = padvalx2 = int(np.floor((npad - xdim)/2.0))
    if xdim % 2:
        padvalx2 += 1
    padvaly1 = padvaly2 = int(np.floor((npad - ydim)/2.0))
    if ydim % 2:
        padvaly2 += 1

    # grid phase and pulse factors
    phase = np.exp(-1j*np.pi*psize*(uv[:,0] + uv[:,1]))
    pulsefac = np.array([pulse(2*np.pi*uvpt[0], 2*np.pi*uvpt[1], psize, dom="F") for uvpt in uv])

    # samples
    samples = sampler(vis_arr, im_info, [uv], order=order, sample_type="vis")
    amp_samples = np.abs(samples)

    # gradient FT
    wdiff_vec = -2.0/len(amp)*((amp - amp_samples) * amp_samples) / (sigma**2) / samples.conj()

    # Setup and perform the inverse FFT
    wdiff_arr = gridder(wdiff_vec * phase.conj() * pulsefac.conj(), im_info, uv, conv_func=conv_func, p_rad=p_rad)    
    grad_arr = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(wdiff_arr))) * npad * npad

    # extract relevent cells and flatten
    out = np.real(grad_arr[padvalx1:-padvalx2,padvaly1:-padvaly2].flatten()) # TODO or is x<-->y??

    return out

def chisq_bs_fft(vis_arr, A, bis, sigma, order=FFT_INTERP_DEFAULT):
    """Bispectrum chi-squared from fft"""
    im_info, (uv1, uv2, uv3) = A 
    
    bisamples = sampler(vis_arr, im_info, [uv1, uv2, uv3], order=order, sample_type="bs")

    return np.sum(np.abs(((bis - bisamples)/sigma))**2)/(2.*len(bis))

def chisqgrad_bs_fft(vis_arr, A, bis, sigma, 
                     order=FFT_INTERP_DEFAULT, conv_func=GRIDDER_CONV_FUNC_DEFAULT, p_rad=GRIDDER_P_RAD_DEFAULT):

    """The gradient of the amplitude chi-squared
    """
    im_info, (uv1, uv2, uv3) = A 
    (xdim, ydim, npad, psize, pulse) = im_info

    # grid phase and pulse factors
    phase1 = np.exp(-1j*np.pi*psize*(uv1[:,0] + uv1[:,1]))
    phase2 = np.exp(-1j*np.pi*psize*(uv2[:,0] + uv2[:,1]))
    phase3 = np.exp(-1j*np.pi*psize*(uv3[:,0] + uv3[:,1]))
    pulsefac1 = np.array([pulse(2*np.pi*uvpt[0], 2*np.pi*uvpt[1], psize, dom="F") for uvpt in uv1])
    pulsefac2 = np.array([pulse(2*np.pi*uvpt[0], 2*np.pi*uvpt[1], psize, dom="F") for uvpt in uv2])
    pulsefac3 = np.array([pulse(2*np.pi*uvpt[0], 2*np.pi*uvpt[1], psize, dom="F") for uvpt in uv3])

    # parameters to unpad the array  
    padvalx1 = padvalx2 = int(np.floor((npad - xdim)/2.0))
    if xdim % 2:
        padvalx2 += 1
    padvaly1 = padvaly2 = int(np.floor((npad - ydim)/2.0))
    if ydim % 2:
        padvaly2 += 1

    v1 = sampler(vis_arr, im_info, [uv1], order=order, sample_type="vis")
    v2 = sampler(vis_arr, im_info, [uv2], order=order, sample_type="vis")
    v3 = sampler(vis_arr, im_info, [uv3], order=order, sample_type="vis")
    bisamples = v1*v2*v3

    wdiff = -1.0/len(bis)*(bis - bisamples)/(sigma**2)

    pt1 = wdiff * (v2 * v3).conj()
    pt2 = wdiff * (v1 * v3).conj()
    pt3 = wdiff * (v1 * v2).conj()

    # Setup and perform the inverse FFT
    wdiff_arr1 = gridder(pt1 * phase1.conj() * pulsefac1.conj(), im_info, uv1, conv_func=conv_func, p_rad=p_rad)    
    wdiff_arr2 = gridder(pt2 * phase2.conj() * pulsefac2.conj(), im_info, uv2, conv_func=conv_func, p_rad=p_rad)    
    wdiff_arr3 = gridder(pt3 * phase3.conj() * pulsefac3.conj(), im_info, uv3, conv_func=conv_func, p_rad=p_rad)    
    grad_arr = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(wdiff_arr1 + wdiff_arr2 + wdiff_arr3))) * npad * npad

    # extract relevant cells and flatten
    out = np.real(grad_arr[padvalx1:-padvalx2,padvaly1:-padvaly2].flatten()) # TODO or is x<-->y??

    return out

def chisq_cphase_fft(vis_arr, A, clphase, sigma, order=FFT_INTERP_DEFAULT):
    """Closure Phases (normalized) chi-squared from fft
    """

    im_info, (uv1, uv2, uv3) = A
    clphase = clphase * DEGREE
    sigma = sigma * DEGREE
    clphase_samples = np.angle(sampler(vis_arr, im_info, [uv1, uv2, uv3], order=order, sample_type="bs"))
    chisq = (2.0/len(clphase)) * np.sum((1.0 - np.cos(clphase-clphase_samples))/(sigma**2))
    return chisq

def chisqgrad_cphase_fft(vis_arr, A, clphase, sigma,
                         order=FFT_INTERP_DEFAULT, conv_func=GRIDDER_CONV_FUNC_DEFAULT, p_rad=GRIDDER_P_RAD_DEFAULT):
    """The gradient of the closure phase chi-squared from fft"""
    """The gradient of the amplitude chi-squared"""
    im_info, (uv1, uv2, uv3) = A
    (xdim, ydim, npad, psize, pulse) = im_info
    phase1 = np.exp(-1j*np.pi*psize*(uv1[:,0] + uv1[:,1]))
    phase2 = np.exp(-1j*np.pi*psize*(uv2[:,0] + uv2[:,1]))
    phase3 = np.exp(-1j*np.pi*psize*(uv3[:,0] + uv3[:,1]))
    pulsefac1 = np.array([pulse(2*np.pi*uvpt[0], 2*np.pi*uvpt[1], psize, dom="F") for uvpt in uv1])
    pulsefac2 = np.array([pulse(2*np.pi*uvpt[0], 2*np.pi*uvpt[1], psize, dom="F") for uvpt in uv2])
    pulsefac3 = np.array([pulse(2*np.pi*uvpt[0], 2*np.pi*uvpt[1], psize, dom="F") for uvpt in uv3])
    # parameters to unpad the array  
    padvalx1 = padvalx2 = int(np.floor((npad - xdim)/2.0))
    if xdim % 2:
        padvalx2 += 1
    padvaly1 = padvaly2 = int(np.floor((npad - ydim)/2.0))
    if ydim % 2:
        padvaly2 += 1

    clphase = clphase * DEGREE
    sigma = sigma * DEGREE

    v1 = sampler(vis_arr, im_info, [uv1], order=order, sample_type="vis")
    v2 = sampler(vis_arr, im_info, [uv2], order=order, sample_type="vis")
    v3 = sampler(vis_arr, im_info, [uv3], order=order, sample_type="vis")
    clphase_samples = np.angle(v1*v2*v3)

    pref = (2.0/len(clphase)) * np.sin(clphase - clphase_samples)/(sigma**2)
    pt1  = pref/v1.conj()
    pt2  = pref/v2.conj()
    pt3  = pref/v3.conj()

    # Setup and perform the inverse FFT
    wdiff_arr1 = gridder(pt1 * phase1.conj() * pulsefac1.conj(), im_info, uv1, conv_func=conv_func, p_rad=p_rad)    
    wdiff_arr2 = gridder(pt2 * phase2.conj() * pulsefac2.conj(), im_info, uv2, conv_func=conv_func, p_rad=p_rad)    
    wdiff_arr3 = gridder(pt3 * phase3.conj() * pulsefac3.conj(), im_info, uv3, conv_func=conv_func, p_rad=p_rad)    
    grad_arr = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(wdiff_arr1 + wdiff_arr2 + wdiff_arr3))) * npad * npad

    # extract relevant cells and flatten
    out = np.imag(grad_arr[padvalx1:-padvalx2,padvaly1:-padvaly2].flatten()) # TODO or is x<-->y??

    return out

def chisq_camp_fft(vis_arr, A, clamp, sigma, order=FFT_INTERP_DEFAULT):
    """Closure Amplitudes (normalized) chi-squared from fft
    """

    im_info, (uv1, uv2, uv3, uv4) = A

    clamp_samples = sampler(vis_arr, im_info, [uv1, uv2, uv3, uv4], order=order, sample_type="camp")
    chisq = np.sum(np.abs((clamp - clamp_samples)/sigma)**2)/len(clamp)
    return chisq

def chisqgrad_camp_fft(vis_arr, A, clamp, sigma,
                       order=FFT_INTERP_DEFAULT, conv_func=GRIDDER_CONV_FUNC_DEFAULT, p_rad=GRIDDER_P_RAD_DEFAULT):

    """The gradient of the closure amplitude chi-squared from fft
    """

    im_info, (uv1, uv2, uv3, uv4) = A
    (xdim, ydim, npad, psize, pulse) = im_info

    # parameters to unpad the array  
    padvalx1 = padvalx2 = int(np.floor((npad - xdim)/2.0))
    if xdim % 2:
        padvalx2 += 1
    padvaly1 = padvaly2 = int(np.floor((npad - ydim)/2.0))
    if ydim % 2:
        padvaly2 += 1

    # grid phase and pulse factors
    phase1 = np.exp(-1j*np.pi*psize*(uv1[:,0] + uv1[:,1]))
    phase2 = np.exp(-1j*np.pi*psize*(uv2[:,0] + uv2[:,1]))
    phase3 = np.exp(-1j*np.pi*psize*(uv3[:,0] + uv3[:,1]))
    phase4 = np.exp(-1j*np.pi*psize*(uv4[:,0] + uv4[:,1]))
    pulsefac1 = np.array([pulse(2*np.pi*uvpt[0], 2*np.pi*uvpt[1], psize, dom="F") for uvpt in uv1])
    pulsefac2 = np.array([pulse(2*np.pi*uvpt[0], 2*np.pi*uvpt[1], psize, dom="F") for uvpt in uv2])
    pulsefac3 = np.array([pulse(2*np.pi*uvpt[0], 2*np.pi*uvpt[1], psize, dom="F") for uvpt in uv3])
    pulsefac4 = np.array([pulse(2*np.pi*uvpt[0], 2*np.pi*uvpt[1], psize, dom="F") for uvpt in uv4])

    # sampled visibility and closure amplitudes
    v1 = sampler(vis_arr, im_info, [uv1], order=order, sample_type="vis")
    v2 = sampler(vis_arr, im_info, [uv2], order=order, sample_type="vis")
    v3 = sampler(vis_arr, im_info, [uv3], order=order, sample_type="vis")
    v4 = sampler(vis_arr, im_info, [uv4], order=order, sample_type="vis")

    clamp_samples = np.abs((v1 * v2)/(v3 * v4))

    # gradient components
    pp = (-2.0/len(clamp)) * ((clamp - clamp_samples) * clamp_samples)/(sigma**2)
    pt1 =  pp/v1.conj()
    pt2 =  pp/v2.conj()
    pt3 = -pp/v3.conj()
    pt4 = -pp/v4.conj()

    # grid and perform the inverse FFT
    wdiff_arr1 = gridder(pt1 * phase1.conj() * pulsefac1.conj(), im_info, uv1, conv_func=conv_func, p_rad=p_rad)    
    wdiff_arr2 = gridder(pt2 * phase2.conj() * pulsefac2.conj(), im_info, uv2, conv_func=conv_func, p_rad=p_rad)    
    wdiff_arr3 = gridder(pt3 * phase3.conj() * pulsefac3.conj(), im_info, uv3, conv_func=conv_func, p_rad=p_rad)    
    wdiff_arr4 = gridder(pt4 * phase4.conj() * pulsefac4.conj(), im_info, uv4, conv_func=conv_func, p_rad=p_rad)    

    grad_arr = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(wdiff_arr1 + wdiff_arr2 + wdiff_arr3 + wdiff_arr4))) * npad * npad

    # extract relevant cells and flatten
    out = np.real(grad_arr[padvalx1:-padvalx2,padvaly1:-padvaly2].flatten()) # TODO or is x<-->y??

    return out

def chisq_logcamp_fft(vis_arr, A, log_clamp, sigma, order=FFT_INTERP_DEFAULT):
    """Closure Amplitudes (normalized) chi-squared from fft
    """
    
    im_info, (uv1, uv2, uv3, uv4) = A
    samples = np.log(sampler(vis_arr, im_info, [uv1, uv2, uv3, uv4], order=order, sample_type="camp"))
    chisq = np.sum(np.abs((log_clamp - samples)/sigma)**2) / (len(log_clamp))

    return chisq

def chisqgrad_logcamp_fft(vis_arr, A, log_clamp, sigma,
                          order=FFT_INTERP_DEFAULT, conv_func=GRIDDER_CONV_FUNC_DEFAULT, p_rad=GRIDDER_P_RAD_DEFAULT):

    """The gradient of the closure amplitude chi-squared from fft
    """

    im_info, (uv1, uv2, uv3, uv4) = A
    (xdim, ydim, npad, psize, pulse) = im_info

    # parameters to unpad the array  
    padvalx1 = padvalx2 = int(np.floor((npad - xdim)/2.0))
    if xdim % 2:
        padvalx2 += 1
    padvaly1 = padvaly2 = int(np.floor((npad - ydim)/2.0))
    if ydim % 2:
        padvaly2 += 1

    # extra gridding phases and pulse factors
    phase1 = np.exp(-1j*np.pi*psize*(uv1[:,0] + uv1[:,1]))
    phase2 = np.exp(-1j*np.pi*psize*(uv2[:,0] + uv2[:,1]))
    phase3 = np.exp(-1j*np.pi*psize*(uv3[:,0] + uv3[:,1]))
    phase4 = np.exp(-1j*np.pi*psize*(uv4[:,0] + uv4[:,1]))
    pulsefac1 = np.array([pulse(2*np.pi*uvpt[0], 2*np.pi*uvpt[1], psize, dom="F") for uvpt in uv1])
    pulsefac2 = np.array([pulse(2*np.pi*uvpt[0], 2*np.pi*uvpt[1], psize, dom="F") for uvpt in uv2])
    pulsefac3 = np.array([pulse(2*np.pi*uvpt[0], 2*np.pi*uvpt[1], psize, dom="F") for uvpt in uv3])
    pulsefac4 = np.array([pulse(2*np.pi*uvpt[0], 2*np.pi*uvpt[1], psize, dom="F") for uvpt in uv4])

    # sampled visibilities and closure amplitudes
    v1 = sampler(vis_arr, im_info, [uv1], order=order, sample_type="vis")
    v2 = sampler(vis_arr, im_info, [uv2], order=order, sample_type="vis")
    v3 = sampler(vis_arr, im_info, [uv3], order=order, sample_type="vis")
    v4 = sampler(vis_arr, im_info, [uv4], order=order, sample_type="vis")

    log_clamp_samples = np.log(np.abs((v1 * v2)/(v3 * v4)))

    # gradient components
    pp = (-2.0/len(log_clamp)) * (log_clamp - log_clamp_samples) / (sigma**2)
    pt1 = pp / v1.conj()
    pt2 = pp / v2.conj()
    pt3 = -pp / v3.conj()
    pt4 = -pp / v4.conj()

    # grid and perform the inverse FFT
    wdiff_arr1 = gridder(pt1 * phase1.conj() * pulsefac1.conj(), im_info, uv1, conv_func=conv_func, p_rad=p_rad)    
    wdiff_arr2 = gridder(pt2 * phase2.conj() * pulsefac2.conj(), im_info, uv2, conv_func=conv_func, p_rad=p_rad)    
    wdiff_arr3 = gridder(pt3 * phase3.conj() * pulsefac3.conj(), im_info, uv3, conv_func=conv_func, p_rad=p_rad)    
    wdiff_arr4 = gridder(pt4 * phase4.conj() * pulsefac4.conj(), im_info, uv4, conv_func=conv_func, p_rad=p_rad)    

    grad_arr = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(wdiff_arr1 + wdiff_arr2 + wdiff_arr3 + wdiff_arr4))) * npad * npad

    # extract relevant cells and flatten
    out = np.real(grad_arr[padvalx1:-padvalx2,padvaly1:-padvaly2].flatten()) # TODO or is x<-->y??

    return out


##################################################################################################
# Regularizer and Gradient Functions
##################################################################################################

def ssimple(imvec, priorvec, flux):
    """Simple entropy
    """
    #norm = flux
    norm = 1
    return -np.sum(imvec*np.log(imvec/priorvec))/norm

def ssimplegrad(imvec, priorvec, flux):
    """Simple entropy gradient
    """
    #norm = flux
    norm =1
    return (-np.log(imvec/priorvec) - 1)/norm

def sl1(imvec, priorvec, flux):
    """L1 norm regularizer
    """
    #norm = flux
    norm = 1
    return -np.sum(np.abs(imvec - priorvec))/norm

def sl1grad(imvec, priorvec, flux):
    """L1 norm gradient
    """
    #norm = flux
    norm = 1
    return -np.sign(imvec - priorvec)/norm

def sgs(imvec, priorvec, flux):
    """Gull-skilling entropy
    """
    #norm = flux
    norm =1
    return np.sum(imvec - priorvec - imvec*np.log(imvec/priorvec))/norm


def sgsgrad(imvec, priorvec, flux):
    """Gull-Skilling gradient
    """
    #norm = flux
    norm = 1
    return -np.log(imvec/priorvec)/norm


def stv(imvec, nx, ny, flux):
    """Total variation regularizer
    """
    #norm = flux
    norm = 1
    im = imvec.reshape(ny, nx)
    impad = np.pad(im, 1, mode='constant', constant_values=0)
    im_l1 = np.roll(impad, -1, axis=0)[1:ny+1, 1:nx+1]
    im_l2 = np.roll(impad, -1, axis=1)[1:ny+1, 1:nx+1]
    out = -np.sum(np.sqrt(np.abs(im_l1 - im)**2 + np.abs(im_l2 - im)**2))
    return out/norm

def stvgrad(imvec, nx, ny, flux):
    """Total variation gradient
    """
    #norm = flux
    norm = 1
    im = imvec.reshape(ny,nx)
    impad = np.pad(im, 1, mode='constant', constant_values=0)
    im_l1 = np.roll(impad, -1, axis=0)[1:ny+1, 1:nx+1]
    im_l2 = np.roll(impad, -1, axis=1)[1:ny+1, 1:nx+1]
    im_r1 = np.roll(impad, 1, axis=0)[1:ny+1, 1:nx+1]
    im_r2 = np.roll(impad, 1, axis=1)[1:ny+1, 1:nx+1]
    im_r1l2 = np.roll(np.roll(impad, 1, axis=0), -1, axis=1)[1:ny+1, 1:nx+1]
    im_l1r2 = np.roll(np.roll(impad, 1, axis=0), -1, axis=1)[1:ny+1, 1:nx+1]

    g1 = (2*im - im_l1 - im_l2)/np.sqrt((im_l1 - im)**2 + (im_l2 - im)**2)
    g2 = (im - im_r1)/np.sqrt((im_r1 - im)**2 + (im_r1l2 - im_r1)**2)
    g3 = (im - im_r2)/np.sqrt((im_r2 - im)**2 + (im_l1r2 - im_r2)**2)
    out= -(g1 + g2 + g3).flatten()
    return out/norm

def stv2(imvec, nx, ny, flux):
    """Squared Total variation regularizer
    """
    #norm = flux
    norm = 1
    im = imvec.reshape(ny, nx)
    impad = np.pad(im, 1, mode='constant', constant_values=0)
    im_l1 = np.roll(impad, -1, axis=0)[1:ny+1, 1:nx+1]
    im_l2 = np.roll(impad, -1, axis=1)[1:ny+1, 1:nx+1]
    out = -np.sum(np.abs(im_l1 - im)**2 + np.abs(im_l2 - im)**2)
    return out/norm

def stv2grad(imvec, nx, ny, flux):
    """Squared Total variation gradient
    """
    #norm = flux
    norm = 1
    im = imvec.reshape(ny,nx)
    impad = np.pad(im, 1, mode='constant', constant_values=0)
    im_l1 = np.roll(impad, -1, axis=0)[1:ny+1, 1:nx+1]
    im_l2 = np.roll(impad, -1, axis=1)[1:ny+1, 1:nx+1]
    im_r1 = np.roll(impad, 1, axis=0)[1:ny+1, 1:nx+1]
    im_r2 = np.roll(impad, 1, axis=1)[1:ny+1, 1:nx+1]
    im_r1l2 = np.roll(np.roll(impad, 1, axis=0), -1, axis=1)[1:ny+1, 1:nx+1]
    im_l1r2 = np.roll(np.roll(impad, 1, axis=0), -1, axis=1)[1:ny+1, 1:nx+1]

    g1 = (2*im - im_l1 - im_l2)
    g2 = (im - im_r1)
    g3 = (im - im_r2)
    out= -2*(g1 + g2 + g3).flatten()
    return out/norm

def spatch(imvec, priorvec, flux):
    """Patch prior regularizer
    """
    #norm = flux**2
    norm = 1
    out = -0.5*np.sum( ( imvec - priorvec) ** 2)
    return out/norm

def spatchgrad(imvec, priorvec, flux):
    """Patch prior gradient
    """
    #norm = flux**2
    norm = 1
    out = -(imvec  - priorvec)
    return out/norm

##################################################################################################
# Embedding and Chi^2 Data functions
##################################################################################################
def embed(im, mask, clipfloor=0., randomfloor=False):
    """Embeds a 1d image array into the size of boolean embed mask
    """
    j=0
    out=np.zeros(len(mask))
    for i in range(len(mask)):
        if mask[i]:
            out[i] = im[j]
            j += 1
        else:
            # prevent total variation gradient singularities
            if randomfloor: out[i] = clipfloor * np.random.normal()
            else: out[i] = clipfloor

    return out

def chisqdata_vis(Obsdata, Prior, mask):
    """Return the visibilities, sigmas, and fourier matrix for an observation, prior, mask
    """

    data_arr = Obsdata.unpack(['u','v','vis','sigma'])
    uv = np.hstack((data_arr['u'].reshape(-1,1), data_arr['v'].reshape(-1,1)))
    vis = data_arr['vis']
    sigma = data_arr['sigma']
    A = ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv, pulse=Prior.pulse, mask=mask)

    return (vis, sigma, A)

def chisqdata_amp(Obsdata, Prior, mask):
    """Return the amplitudes, sigmas, and fourier matrix for and observation, prior, mask
    """

    ampdata = Obsdata.unpack(['u','v','amp','sigma'], debias=True)
    uv = np.hstack((ampdata['u'].reshape(-1,1), ampdata['v'].reshape(-1,1)))
    amp = ampdata['amp']
    sigma = ampdata['sigma']
    A = ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv, pulse=Prior.pulse, mask=mask)

    return (amp, sigma, A)

def chisqdata_bs(Obsdata, Prior, mask):
    """return the bispectra, sigmas, and fourier matrices for and observation, prior, mask
    """

    biarr = Obsdata.bispectra(mode="all", count="min")
    uv1 = np.hstack((biarr['u1'].reshape(-1,1), biarr['v1'].reshape(-1,1)))
    uv2 = np.hstack((biarr['u2'].reshape(-1,1), biarr['v2'].reshape(-1,1)))
    uv3 = np.hstack((biarr['u3'].reshape(-1,1), biarr['v3'].reshape(-1,1)))
    bi = biarr['bispec']
    sigma = biarr['sigmab']

    A3 = (ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv1, pulse=Prior.pulse, mask=mask),
          ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv2, pulse=Prior.pulse, mask=mask),
          ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv3, pulse=Prior.pulse, mask=mask)
         )

    return (bi, sigma, A3)

def chisqdata_cphase(Obsdata, Prior, mask):
    """Return the closure phases, sigmas, and fourier matrices for and observation, prior, mask
    """

    clphasearr = Obsdata.c_phases(mode="all", count="min")
    uv1 = np.hstack((clphasearr['u1'].reshape(-1,1), clphasearr['v1'].reshape(-1,1)))
    uv2 = np.hstack((clphasearr['u2'].reshape(-1,1), clphasearr['v2'].reshape(-1,1)))
    uv3 = np.hstack((clphasearr['u3'].reshape(-1,1), clphasearr['v3'].reshape(-1,1)))
    clphase = clphasearr['cphase']
    sigma = clphasearr['sigmacp']

    A3 = (ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv1, pulse=Prior.pulse, mask=mask),
          ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv2, pulse=Prior.pulse, mask=mask),
          ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv3, pulse=Prior.pulse, mask=mask)
         )
    return (clphase, sigma, A3)

def chisqdata_camp(Obsdata, Prior, mask):
    """Return the closure amplitudes, sigmas, and fourier matrices for and observation, prior, mask
    """

    clamparr = Obsdata.c_amplitudes(mode='all', count='min', ctype='camp', debias=True)
    uv1 = np.hstack((clamparr['u1'].reshape(-1,1), clamparr['v1'].reshape(-1,1)))
    uv2 = np.hstack((clamparr['u2'].reshape(-1,1), clamparr['v2'].reshape(-1,1)))
    uv3 = np.hstack((clamparr['u3'].reshape(-1,1), clamparr['v3'].reshape(-1,1)))
    uv4 = np.hstack((clamparr['u4'].reshape(-1,1), clamparr['v4'].reshape(-1,1)))
    clamp = clamparr['camp']
    sigma = clamparr['sigmaca']

    A4 = (ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv1, pulse=Prior.pulse, mask=mask),
          ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv2, pulse=Prior.pulse, mask=mask),
          ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv3, pulse=Prior.pulse, mask=mask),
          ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv4, pulse=Prior.pulse, mask=mask)
         )

    return (clamp, sigma, A4)

def chisqdata_logcamp(Obsdata, Prior, mask):
    """Return the log closure amplitudes, sigmas, and fourier matrices for and observation, prior, mask
    """

    clamparr = Obsdata.c_amplitudes(mode='all', count='min', ctype='logcamp', debias=True)
    uv1 = np.hstack((clamparr['u1'].reshape(-1,1), clamparr['v1'].reshape(-1,1)))
    uv2 = np.hstack((clamparr['u2'].reshape(-1,1), clamparr['v2'].reshape(-1,1)))
    uv3 = np.hstack((clamparr['u3'].reshape(-1,1), clamparr['v3'].reshape(-1,1)))
    uv4 = np.hstack((clamparr['u4'].reshape(-1,1), clamparr['v4'].reshape(-1,1)))
    clamp = clamparr['camp']
    sigma = clamparr['sigmaca']

    A4 = (ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv1, pulse=Prior.pulse, mask=mask),
          ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv2, pulse=Prior.pulse, mask=mask),
          ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv3, pulse=Prior.pulse, mask=mask),
          ftmatrix(Prior.psize, Prior.xdim, Prior.ydim, uv4, pulse=Prior.pulse, mask=mask)
         )

    return (clamp, sigma, A4)



def chisqdata_vis_fft(Obsdata, Prior, fft_pad_frac=1):
    """Return the visibilities, sigmas, uv points, and image info
    """

    data_arr = Obsdata.unpack(['u','v','vis','sigma'])
    uv = np.hstack((data_arr['u'].reshape(-1,1), data_arr['v'].reshape(-1,1)))
    vis = data_arr['vis']
    sigma = data_arr['sigma']

    npad = fft_pad_frac * np.max((Prior.xdim, Prior.ydim))

    im_info = (Prior.xdim, Prior.ydim, npad, Prior.psize, Prior.pulse)

    A = (im_info, uv) 

    return (vis, sigma, A)

def chisqdata_amp_fft(Obsdata, Prior, fft_pad_frac=1):
    """Return the amplitudes, sigmas, uv points, and image info
    """

    data_arr = Obsdata.unpack(['u','v','amp','sigma'], debias=True)
    uv = np.hstack((data_arr['u'].reshape(-1,1), data_arr['v'].reshape(-1,1)))
    amp = data_arr['amp']
    sigma = data_arr['sigma']

    npad = fft_pad_frac * np.max((Prior.xdim, Prior.ydim))

    im_info = (Prior.xdim, Prior.ydim, npad, Prior.psize, Prior.pulse)

    A = (im_info, uv) # to maintain compatibility with chisq() wrapper

    return (amp, sigma, A)

def chisqdata_bs_fft(Obsdata, Prior, fft_pad_frac=1):
    """Return the bispectra, sigmas, uv points, and image info
    """
    biarr = Obsdata.bispectra(mode="all", count="min")
    uv1 = np.hstack((biarr['u1'].reshape(-1,1), biarr['v1'].reshape(-1,1)))
    uv2 = np.hstack((biarr['u2'].reshape(-1,1), biarr['v2'].reshape(-1,1)))
    uv3 = np.hstack((biarr['u3'].reshape(-1,1), biarr['v3'].reshape(-1,1)))
    bi = biarr['bispec']
    sigma = biarr['sigmab']

    npad = fft_pad_frac * np.max((Prior.xdim, Prior.ydim))

    im_info = (Prior.xdim, Prior.ydim, npad, Prior.psize, Prior.pulse)

    A = (im_info, [uv1, uv2, uv3]) 

    return (bi, sigma, A)

def chisqdata_cphase_fft(Obsdata, Prior, fft_pad_frac=1):
    """Return the closure phases, sigmas, uv points, and image info
    """
    clphasearr = Obsdata.c_phases(mode="all", count="min")
    uv1 = np.hstack((clphasearr['u1'].reshape(-1,1), clphasearr['v1'].reshape(-1,1)))
    uv2 = np.hstack((clphasearr['u2'].reshape(-1,1), clphasearr['v2'].reshape(-1,1)))
    uv3 = np.hstack((clphasearr['u3'].reshape(-1,1), clphasearr['v3'].reshape(-1,1)))
    clphase = clphasearr['cphase']
    sigma = clphasearr['sigmacp']

    npad = fft_pad_frac * np.max((Prior.xdim, Prior.ydim))

    im_info = (Prior.xdim, Prior.ydim, npad, Prior.psize, Prior.pulse)

    A = (im_info, [uv1, uv2, uv3])

    return (clphase, sigma, A)

def chisqdata_camp_fft(Obsdata, Prior, fft_pad_frac=1):
    """Return the closure phases, sigmas, uv points, and image info
    """
    clamparr = Obsdata.c_amplitudes(mode='all', count='min', ctype='camp', debias=True)
    uv1 = np.hstack((clamparr['u1'].reshape(-1,1), clamparr['v1'].reshape(-1,1)))
    uv2 = np.hstack((clamparr['u2'].reshape(-1,1), clamparr['v2'].reshape(-1,1)))
    uv3 = np.hstack((clamparr['u3'].reshape(-1,1), clamparr['v3'].reshape(-1,1)))
    uv4 = np.hstack((clamparr['u4'].reshape(-1,1), clamparr['v4'].reshape(-1,1)))
    clamp = clamparr['camp']
    sigma = clamparr['sigmaca']

    npad = fft_pad_frac * np.max((Prior.xdim, Prior.ydim))

    im_info = (Prior.xdim, Prior.ydim, npad, Prior.psize, Prior.pulse)

    A = (im_info, [uv1, uv2, uv3, uv4]) 

    return (clamp, sigma, A)

def chisqdata_logcamp_fft(Obsdata, Prior, fft_pad_frac=1):
    """Return the closure phases, sigmas, uv points, and image info
    """
    clamparr = Obsdata.c_amplitudes(mode='all', count='min', ctype='logcamp', debias=True)
    uv1 = np.hstack((clamparr['u1'].reshape(-1,1), clamparr['v1'].reshape(-1,1)))
    uv2 = np.hstack((clamparr['u2'].reshape(-1,1), clamparr['v2'].reshape(-1,1)))
    uv3 = np.hstack((clamparr['u3'].reshape(-1,1), clamparr['v3'].reshape(-1,1)))
    uv4 = np.hstack((clamparr['u4'].reshape(-1,1), clamparr['v4'].reshape(-1,1)))
    clamp = clamparr['camp']
    sigma = clamparr['sigmaca']

    npad = fft_pad_frac * np.max((Prior.xdim, Prior.ydim))

    im_info = (Prior.xdim, Prior.ydim, npad, Prior.psize, Prior.pulse)

    A = (im_info, [uv1, uv2, uv3, uv4]) 

    return (clamp, sigma, A)

##################################################################################################
# FFT functions
##################################################################################################

def conv_func_pill(x,y):
    if abs(x) < 0.5 and abs(y) < 0.5:
        out = 1.
    else:
        out = 0.
    return out

def conv_func_gauss(x,y):
    return np.exp(-(x**2 + y**2))

# There's a bug in my version of scipy with spheroidal function of order 0! - gives nans for eta<1
def conv_func_spheroidal(x,y,p,m):
    etax = 2.*x/float(p)
    etay = 2.*x/float(p)
    psix =  abs(1-etax**2)**m * scipy.special.pro_rad1(m,0,0.5*np.pi*p,etax)[0]
    psiy = abs(1-etay**2)**m * scipy.special.pro_rad1(m,0,0.5*np.pi*p,etay)[0]

    return psix*psiy

def fft_imvec(imvec, im_info):
    """
    Returns fft of imvec on  grid
    im_info = (xdim, ydim, npad, psize, pulse) = im_info
    order is the order of the spline interpolation
    """

    (xdim, ydim, npad, psize, pulse) = im_info

    # Pad image    
    padvalx1 = padvalx2 = int(np.floor((npad - xdim)/2.0))
    if xdim % 2:
        padvalx2 += 1
    padvaly1 = padvaly2 = int(np.floor((npad - ydim)/2.0))
    if ydim % 2:
        padvaly2 += 1

    imarr = imvec.reshape(ydim, xdim)
    imarr = np.pad(imarr, ((padvalx1,padvalx2),(padvaly1,padvaly2)), 'constant', constant_values=0.0)
    npad = imarr.shape[0]
    if imarr.shape[0]!=imarr.shape[1]:
        raise Exception("FFT padding did not return a square image!")

    # FFT for visibilities
    # TODO can we get rid of the fftshifts?
    vis_im = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(imarr)))

    return vis_im

# TODO add other sample types
def sampler(griddata, im_info, uvset, sample_type="vis", order=3):
    """
    Samples griddata (e.g. the FFT of an image) at uv points 
    the griddata should already be rotated so u,v = 0,0 is in the center
    im_info tuple contains (xdim, ydim, npad, psize, pulse) of the grid
    order is the order of the spline interpolation
    sample_type gives the type of sample returned: options are 'vis','bs','camp'
    """

    (xdim, ydim, npad, psize, pulse) = im_info

    if sample_type not in ["vis","bs","camp"]:
        raise Exception("sampler sample_type should be either 'vis','bs',or 'camp'!")
    if griddata.shape[0] != griddata.shape[0]:
        raise Exception("griddata should be a square array!")

    npix = griddata.shape[0]

    dataset = []
    for uv in uvset:
        vu2  = np.hstack((uv[:,1].reshape(-1,1), uv[:,0].reshape(-1,1)))
        du   = 1.0/(npix*psize)
        vu2  = (vu2/du + 0.5*npix).T

        datare = nd.map_coordinates(np.real(griddata), vu2, order=order)
        dataim = nd.map_coordinates(np.imag(griddata), vu2, order=order)
        data = datare + 1j*dataim

        # Extra phase to match centroid convention -- right??
        phase = np.exp(-1j*np.pi*psize*(uv[:,0] + uv[:,1]))
        data = data * phase

        # Multiply by the pulse function
        # TODO make faster?
        pulsefac = np.array([pulse(2*np.pi*uvpt[0], 2*np.pi*uvpt[1], psize, dom="F") for uvpt in uv])
        data = data * pulsefac

        dataset.append(data)
 
    if sample_type=="vis":
        out = dataset[0]
    if sample_type=="bs":
        out = dataset[0]*dataset[1]*dataset[2]
    if sample_type=="camp":
        out = np.abs((dataset[0]*dataset[1])/(dataset[2]*dataset[3]))
    return out

def gridder(data, im_info, uv, conv_func=GRIDDER_CONV_FUNC_DEFAULT, p_rad=GRIDDER_P_RAD_DEFAULT):
    """
    Grid data sampled at uv points on a square array
    im_info tuple contains (xdim, ydim, npad, psize, pulse) of the grid
    conv_func is the convolution function: current options are "pillbox", "gaussian"
    p_rad is the pixel radius inside wich the conv_func is nonzero
    """

    (xdim, ydim, npad, psize, pulse) = im_info

    if len(uv) != len(data):
        raise Exception("uv and data are not the same length!")
    if not (conv_func in ['pillbox','gaussian']):
        raise Exception("conv_func must be either 'pillbox' or 'gaussian'")

    vu2 = np.hstack((uv[:,1].reshape(-1,1), uv[:,0].reshape(-1,1)))
    du  = 1.0/(npad*psize)
    vu2 = (vu2/du + 0.5*npad)

    datagrid = np.zeros((npad, npad)).astype('c16')
    coords = np.round(vu2).astype(int)
    dcoords = vu2 - np.round(vu2).astype(int)
    norm = np.zeros_like(len(coords))

    for dy in range(-p_rad, p_rad+1): 
        for dx in range(-p_rad, p_rad+1):
            if conv_func == 'gaussian':
                norm = norm + conv_func_gauss(dy - dcoords[:,0], dx - dcoords[:,1]) 
            elif conv_func == 'pillbox':
                norm = norm + conv_func_pill(dy - dcoords[:,0], dx - dcoords[:,1]) 

    for dy in range(-p_rad, p_rad+1): 
        for dx in range(-p_rad, p_rad+1):
            if conv_func == 'gaussian':
                weight = conv_func_gauss(dy - dcoords[:,0], dx - dcoords[:,1])/norm
            elif conv_func == 'pillbox':
                weight = conv_func_pill(dy - dcoords[:,0], dx - dcoords[:,1])/norm
            np.add.at(datagrid, tuple(map(tuple, (coords + [dy, dx]).transpose())), data*weight)
    
    return datagrid

#    vu2 = np.hstack((uv[:,1].reshape(-1,1), uv[:,0].reshape(-1,1)))
#    du  = 1.0/(npad*psize)
#    vu2 = (vu2/du + 0.5*npad)

#    datagrid = np.zeros((npad, npad)).astype('c16')
#    for k in range(len(data)):
#        point = vu2[k]
#        vispoint = data[k]

#        vumin = np.ceil(point - p_rad).astype(int)
#        vumax = np.floor(point + p_rad).astype(int)

#        norm = 0.0 # Need to normalize the conv_func
#        for i in np.arange(vumin[0], vumax[0]+1):
#            for j in np.arange(vumin[1], vumax[1]+1):
#                
#                if conv_func == 'pillbox':
#                    norm += conv_func_pill(j-point[1], i-point[0])

#                elif conv_func == 'gaussian':
#                    norm += conv_func_gauss(j-point[1], i-point[0])
#        
#        for i in np.arange(vumin[0], vumax[0]+1):
#            for j in np.arange(vumin[1], vumax[1]+1):
#                
#                if conv_func == 'pillbox':
#                    datagrid[i,j] += conv_func_pill(j-point[1], i-point[0]) * vispoint / norm

#                elif conv_func == 'gaussian':
#                    datagrid[i,j] += conv_func_gauss(j-point[1], i-point[0]) * vispoint / norm

#    return datagrid

##################################################################################################
# Restoring and Plotting Functions
##################################################################################################
def threshold(image, frac_i=1.e-5, frac_pol=1.e-3):
    """Apply a hard threshold to the image.
    """

    imvec = np.copy(image.imvec)

    thresh = frac_i*np.abs(np.max(imvec))
    lowval = thresh
    flux = np.sum(imvec)

    for j in range(len(imvec)):
        if imvec[j] < thresh:
            imvec[j]=lowval

    imvec = flux*imvec/np.sum(imvec)
    out = image.Image(imvec.reshape(image.ydim,image.xdim), image.psize,
                   image.ra, image.dec, rf=image.rf, source=image.source, mjd=image.mjd)
    return out

def blur_circ(image, fwhm_i, fwhm_pol=0):
    """Apply a circular gaussian filter to the image.
       fwhm_i and fwhm_pol are in radians
    """

    # Blur Stokes I
    sigma = fwhm_i/(2. * np.sqrt(2. * np.log(2.)))
    sigmap = sigma/image.psize
    im = filt.gaussian_filter(image.imvec.reshape(image.ydim, image.xdim), (sigmap, sigmap))
    out = image.Image(im, image.psize, image.ra, image.dec, rf=image.rf, source=image.source, mjd=image.mjd)

    # Blur Stokes Q and U
    if len(image.qvec) and fwhm_pol:
        sigma = fwhm_pol/(2. * np.sqrt(2. * np.log(2.)))
        sigmap = sigma/image.psize
        imq = filt.gaussian_filter(image.qvec.reshape(image.ydim,image.xdim), (sigmap, sigmap))
        imu = filt.gaussian_filter(image.uvec.reshape(image.ydim,image.xdim), (sigmap, sigmap))
        out.add_qu(imq, imu)

    return out


def plot_i(im, Prior, nit, chi2_1, chi2_2, ipynb=False):
    """Plot the total intensity image at each iteration
    """

    plt.ion()
    plt.pause(0.00001)
    plt.clf()

    plt.imshow(im.reshape(Prior.ydim,Prior.xdim), cmap=plt.get_cmap('afmhot'), interpolation='gaussian')
    xticks = ticks(Prior.xdim, Prior.psize/RADPERAS/1e-6)
    yticks = ticks(Prior.ydim, Prior.psize/RADPERAS/1e-6)
    plt.xticks(xticks[0], xticks[1])
    plt.yticks(yticks[0], yticks[1])
    plt.xlabel('Relative RA ($\mu$as)')
    plt.ylabel('Relative Dec ($\mu$as)')
    plt.title("step: %i  $\chi^2_1$: %f  $\chi^2_2$: %f" % (nit, chi2_1, chi2_2), fontsize=20)
    #plt.draw()

    if ipynb:
        display.clear_output()
        display.display(plt.gcf())
