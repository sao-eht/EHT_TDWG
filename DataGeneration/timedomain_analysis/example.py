from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import itertools as it

import ehtim as eh
import ehtim.io.load as lo
import ehtim.vex as ve
import ehtim.movie as mv
import closure as cl
import LS as ls


vex = ve.Vex("vex_files/EHT2017/night3.vex")

framedur_sec = 16.2 # seconds per frame

# observation duration in hr
mjd_s, mjd_e = vex.get_obs_timerange('SGRA')
duration_hr = (mjd_e - mjd_s)*24.

# reuired number of frames to observe 'duration_hr' hours
nframes = int(np.round(duration_hr/(framedur_sec/3600.)))

# Choose movie to load
# Contact hshiokawa@cfa.harvard.edu for the movie files
#----- Avery's hotspot model, Doeleman et al. (2009) Model B
#mov = lo.load_movie_fits_hotspot('HotSpot/model2-', nframes, 100, framedur_sec, mjd=int(mjd_s), start_hr=(mjd_s-np.floor(mjd_s))*24.)
#----- Hotspot + GRMHD disk model, Roelofs et al. (2017)
mov = lo.load_movie_txt("HotSpot_Disk/", nframes, framedur=framedur_sec)
#----- Hotspot + GRMHD jet model, Roelofs et al. (2017)
#mov = lo.load_movie_txt("HotSpot_Jet/", nframes, framedur=framedur_sec)

# observe movie
obs = mov.observe_vex( vex, 'SGRA', t_int=framedur_sec )

# get closure quantity curves
clo = cl.Closure(obs)
"""
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

tri_id = 12 # choose 12th triangle
tri_name = clo.tri[tri_id][0]+"-"+clo.tri[tri_id][1]+"-"+clo.tri[tri_id][2]
print("Plotting closure phase for "+tri_name )
f.errorbar( clo.cp[tri_id][0], clo.cp[tri_id][1], yerr=clo.cp[tri_id][2], fmt='k.' )
f.set_title( tri_name )
f.set_xlabel('UT [hr]')
f.set_ylabel('CP [deg]')

f = fig.add_subplot(212)
quad_id = 20 # choose 20th quadrangle
quad_name = clo.quad[quad_id][0]+"-"+clo.quad[quad_id][1]+"-"+clo.quad[quad_id][2]+"-"+clo.quad[quad_id][3]
print("Plotting closure amplitude for "+quad_name)
f.errorbar( clo.ca[quad_id][0], clo.ca[quad_id][1], yerr=clo.ca[quad_id][2], fmt='k.' )
f.set_title( quad_name )
f.set_xlabel('UT [hr]')
f.set_ylabel('CA')
plt.tight_layout()


#------------------------------------------------------------
# writing data examples 
clo.record_cp(12) # record t vs. cp with error for 12th triangle
clo.record_ca(20) # record t vs. ca with error for 20th quadrangle


#------------------------------------------------------------
# getting periodogram examples
tri_id = 12 # choose 12th triangle
tri_name = clo.tri[tri_id][0]+"-"+clo.tri[tri_id][1]+"-"+clo.tri[tri_id][2]
print("Getting a LS periodogram for "+tri_name+" closure phase." )

freq, power = ls.get_LS_periodogram(clo.cp[tri_id]) # frequency [1/hr] and normalized power


# calculate significance
t = clo.cp[tri_id][0]
signif = 0.001 # 0.1% significance
print("Calculating %f significance value assuming background Gaussian noise."%(signif) )

signif_z = ls.get_significance( t, signif )


# plotting periodogram example
print("Plotting...")
fig = plt.figure()
f = fig.add_subplot(111)
f.plot(freq,power,'k-')
f.plot([freq.min(),freq.max()],[signif_z,signif_z],'g-')
f.set_xscale('log')

period = 27. # minutes
for fac in range(7):
    f.plot([fac*60./period,fac*60./period],[0.,50],'b-') # mark the expected period and higher modes
	
f.set_title( tri_name )
f.set_xlabel('Frequency [1/hr]')
f.set_ylabel('Power')

#------------------------------------------------------------
plt.draw()
plt.pause(1)
raw_input("<Hit Enter To Close>")
plt.close(fig)
