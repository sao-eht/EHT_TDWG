from __future__ import division
from __future__ import print_function

import numpy as np

from gatspy.periodic import LombScargleFast
from gatspy.periodic import LombScargle
from gatspy.periodic import LombScargleMultiband
from gatspy.periodic import LombScargleMultibandFast
from gatspy.periodic import TrendedLombScargle


def get_LS_periodogram( y, consider_error=True ):

    if consider_error==True:
        #model = LombScargleFast().fit(y[0], y[1], y[2])
        #model = LombScargle().fit(y[0], y[1], y[2])
        #model = LombScargleMultiband().fit(y[0], y[1], y[2])
        model = LombScargleMultibandFast().fit(y[0], y[1], y[2])
        #model = TrendedLombScargle().fit(y[0], y[1], y[2])
    else:
        #model = LombScargleFast().fit(y[0], y[1])
        #model = LombScargle().fit(y[0], y[1])
        #model = LombScargleMultiband().fit(y[0], y[1])
        model = LombScargleMultibandFast().fit(y[0], y[1])
        #model = TrendedLombScargle().fit(y[0], y[1])

    periods, power = model.periodogram_auto(nyquist_factor=1)

    N = len(y[0])
    power *= N/2 # make it equal to the normalized periodogram of NR eq. 13.8.4
    return [1./periods, power]


def get_significance( t, signif ):

    M = find_M(t,200)
    signif_z = Zp(signif,M)
    return signif_z


Pz = lambda z, M: 1.-(1.-np.exp(-z))**M # Numerical Recipes eq. 13.8.7
Zp = lambda p, M: -np.log(1.-(1.-p)**(1./M))
def find_M(t,ite):

    zmax = np.array([])
    for i in range(ite):
        #if i%100==0 or i==ite-1: print "find_M(): i=%d"%i
        noise = np.random.normal(0.,1.,len(t))
        LS_peri = get_LS_periodogram([t,noise],consider_error=False)
        zmax = np.append(zmax,LS_peri[1].max())
    
    z_choice = zmax.mean()
    # Probability that at least one z above z_choice appears is
    Pz_choice = (zmax>z_choice).sum()/float(ite)
    
    # Bisection method to find M
    M_l = 1.
    M_h = 10.*len(t)
    M_m = (M_l+M_h)/2.
    
    Pz_h = Pz(z_choice,M_h)
    Pz_l = Pz(z_choice,M_l)
    if Pz_choice<Pz_l or Pz_choice>Pz_h: print("error in find_M(): bisection range")
    
    for i in range(ite):
        Pz_m = Pz(z_choice,M_m)
        if abs(Pz_m-Pz_choice)/((Pz_m+Pz_choice)/2.)<0.05: break
        if Pz_choice > Pz_m:
            M_l = M_m
        else:
            M_h = M_m
        M_m = (M_l+M_h)/2.
    
    if i==ite-1: print("error in find_M(): max iteration reached")
    M = int(M_m)
    #print "Target Pz=%f, Resultant Pz=%f with M=%d"%(Pz_choice,Pz_m,M)
    
    return M


