# Hotaka Shiokawa, 08/30/2017

from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

import ehtim.io.load as lo
import closure as cl
import LS as ls


obs = lo.load_obs_uvfits("hops_uvfits_hi/hops_3597_3C279_hi_closed.uvfits")


#------------------------------------------------------------
# get closure quantity curves
clo = cl.Closure(obs)
"""
Mnemonics
clo.cp[triangle ID] = closure phase curves. (time, CP, error)
clo.tri[triangle ID] = list of stations that consist the triangle
clo.ca[quadrangle ID] = closure amplitude curves. (time, CA, error)
clo.quad[quadrangle ID] = list of stations that consist the quadrangle
All the metadata of Obsdata are also copied to Closure.
"""


#------------------------------------------------------------
# plotting closure quantities example
fig = plt.figure()
f = fig.add_subplot(211)

tri_id = 17 # choose 17th triangle

tri_name = clo.tri[tri_id][0]+"-"+clo.tri[tri_id][1]+"-"+clo.tri[tri_id][2]
f.errorbar( clo.cp[tri_id][0], clo.cp[tri_id][1], yerr=clo.cp[tri_id][2], fmt='k.' )
f.set_title( tri_name )
f.set_xlabel('UT [hr]')
f.set_ylabel('CP [deg]')


#------------------------------------------------------------
# getting and plotting periodogram
freq, power = ls.get_LS_periodogram(clo.cp[tri_id]) # frequency [1/hr] and normalized power

# calculate significance
t = clo.cp[tri_id][0]
signif = 0.001 # 0.1% significance

signif_z = ls.get_significance( t, signif )

# plotting periodogram
f = fig.add_subplot(212)
f.plot(freq,power,'k-')
f.plot([freq.min(),freq.max()],[signif_z,signif_z],'g-')
f.set_xscale('log')

f.set_xlabel('Frequency [1/hr]')
f.set_ylabel('Power')

plt.draw()
plt.pause(1)
raw_input("<Hit Enter To Close>")
plt.close(fig)

