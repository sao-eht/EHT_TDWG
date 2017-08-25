# Program to calculate the Q-metric and error for an example simulation of closure phases from a GRMHD movie and static source. The data set is divided in multiple segments and Q is calculated using equations 21 and 26-28 of Roelofs et al. (2017). The outcome with the example data set is Figure 6 of Roelofs et al. (2017).
# Freek Roelofs, 2017/08/10

import numpy as np
import matplotlib.pyplot as plt

rad2deg=180/np.pi
deg2rad=np.pi/180


#Functions

def gauss(sigma): 
    """Random number from Gaussian with zero mean and standard deviation sigma"""
    return np.random.normal(loc=0,scale=sigma)

def circ_stat(angles):
    """Calculate circular statistics quantities (circular average, circular variance, circular standard deviation) for list of angles"""
    cos=np.zeros(len(angles))
    sin=np.zeros(len(angles))
    for i in range(len(angles)):
        cos[i]=np.cos(angles[i]*deg2rad)
        sin[i]=np.sin(angles[i]*deg2rad)
    cos_avg=np.mean(cos)
    sin_avg=np.mean(sin)
    obs_mean=np.arctan2(sin_avg, cos_avg)*rad2deg
    obs_variance=1-np.sqrt(sin_avg**2+cos_avg**2)
    obs_stdev=np.sqrt(-2*np.log((1-obs_variance)))*rad2deg
    return obs_mean, obs_variance, obs_stdev

def drange(start, stop, step):
    """Get list of time stamps where data should be split for a certain segment size"""
    r = start
    while r < stop:
        yield r
        r += step

def indices(time, segsize):
    """Generate list of indices where list of data points should be split for a certain segment size"""
    indexlist=[0]
    hrs=drange(min(time), max(time), segsize)
    for hr in hrs:
        inthr=min(time, key=lambda x:abs(x-hr))
        if int(np.where(time==inthr)[0]) ==0:
            continue
        else:
            indexlist.append(int(np.where(time==inthr)[0]))
    return indexlist

def segment(seq, indexlist):
    """Split data sequence in multiple segments at indices specified in indexlist"""
    out = []
    for i in range(len(indexlist)):
        if i==len(indexlist)-1:
            out.append(seq[indexlist[i]:])
        else:
            out.append(seq[indexlist[i]:indexlist[i+1]])
    return out

def difflag(series, sigma, n):
    """Difference time series with error bars sigma at lag n"""
    newseries=[]
    newsigma=[]
    for i in range(n,len(series)):
        dif=series[i]-series[i-n]
        err=np.sqrt(sigma[i]**2+sigma[i-n]**2)
        if dif>180:
            dif=series[i]-series[i-n]-360
        if dif<-180:
            dif=series[i]+360-series[i-n]
        newseries.append(dif)
        newsigma.append(err)
    return newseries, newsigma
    
def eps(err):
    """Calculate tilde{epsilon} using Monte Carlo approach assuming Gaussian errors"""
    epsilons=np.zeros(1000)
    for i in range(1000):
        this_it=np.zeros(len(err))
        for j in range(len(err)):
            this_it[j]=gauss(err[j]*deg2rad)
        this_cosi=[np.cos(x) for x in this_it]
        this_sini=[np.sin(x) for x in this_it]
        this_cos_avg=np.mean(this_cosi)
        this_sin_avg=np.mean(this_sini)
        this_R=np.sqrt(this_sin_avg**2+this_cos_avg**2)
        this_obs_sigma=np.sqrt(-2*np.log(this_R))*rad2deg
        epsilons[i]=this_obs_sigma
    epsi=np.mean(epsilons)
    return epsi

def cldata(datfile, segsize, difsize, detrend):
    """Main function to calculate q-metric and error"""

    #Load data   
    data=np.loadtxt(datfile)    
    time=data[:,0]
    obs_cphase=data[:,1]
    ntot=float(len(obs_cphase))
    obs_sigmacp=data[:,2]

    #Plot input data

    #Split in segments
    indexlist=indices(time, segsize)
    time_seg=segment(time, indexlist)
    obs_cphase_seg=segment(obs_cphase, indexlist)
    obs_sigmacp_seg=segment(obs_sigmacp, indexlist)

    #Calculate circular mean, variance, \tilde{epsilon} for each segment
    obs_stdev_seg=[]
    obs_n_seg=[]
    obs_avgerr_seg=[]
    obs_metric_seg=[]
    obs_metric_seg_err=[]

    for i in range(len(time_seg)):
        if detrend==True:
            cpdet, sigmadet=difflag(obs_cphase_seg[i][:],obs_sigmacp_seg[i][:],difsize)

            if len(cpdet)<1:
                continue

            #Detrend
            obs_mean, obs_variance, obs_stdev = circ_stat(cpdet)
            obs_avgerr=eps(sigmadet)
            this_n=len(cpdet)
        else:
            obs_mean, obs_variance, obs_stdev = circ_stat(obs_cphase_seg[i][:])
            obs_avgerr=eps(obs_sigmacp_seg[i][:])
            this_n=len(obs_cphase_seg[i][:])

        obs_n_seg.append(this_n)
        obs_avgerr_seg.append(obs_avgerr)
        obs_stdev_seg.append(obs_stdev)
        obs_metric_seg.append((obs_stdev**2-obs_avgerr**2)/obs_stdev**2)
        obs_metric_seg_err.append(this_n*(obs_stdev**2)**2)

    #Calculate totals and errors
    obs_sigma_bar_sq=0
    obs_epsilon_bar_sq=0
    q_esum=0
    for i in range(len(obs_stdev_seg)):
        obs_sigma_bar_sq += (obs_n_seg[i]/ntot)*(obs_stdev_seg[i]**2)
        obs_epsilon_bar_sq += (obs_n_seg[i]/ntot)*obs_avgerr_seg[i]**2
        q_esum += obs_metric_seg_err[i]
    obs_m_bar=(obs_sigma_bar_sq-obs_epsilon_bar_sq)/obs_sigma_bar_sq
    obs_delta_m_bar=np.sqrt(2.)/ntot*(obs_epsilon_bar_sq/obs_sigma_bar_sq**2)*np.sqrt(q_esum)

    return obs_m_bar, obs_delta_m_bar

#Main program

#Specify segment sizes
segsizes=np.linspace(0.1, 1.5, 22)

#Lag for differencing, example time between measurements is 11 seconds
difsize=25

#Plot parameters
msize=12
eline=3
csize=6
cwidth=3
fig=plt.figure(figsize=(10,6))

#Specify input files for movie and static source
movie_datfile='./mov_i85_LMT-ALMA-SMA.dat'
frame_datfile='./middle_i85_LMT-ALMA-SMA.dat'

#Calculate metric + error and plot
ax=plt.subplot(1, 1, 1)
for segsize in segsizes:
    movie_m_bar, movie_delta_m=cldata(movie_datfile, segsize, difsize, detrend=False)
    frame_m_bar, frame_delta_m = cldata(frame_datfile, segsize, difsize, detrend=False)
    movie_m_bar_detrend, movie_delta_m_detrend = cldata(movie_datfile, segsize, difsize, detrend=True)
    frame_m_bar_detrend, frame_delta_m_detrend = cldata(frame_datfile, segsize,difsize, detrend=True)
    if segsize==segsizes[0]:
        (_, caps, _)=plt.errorbar(segsize, frame_m_bar, yerr=frame_delta_m, marker= '.', color= 'b', linestyle='None', label='Static source', markersize=msize, elinewidth=eline, capsize=csize)
        for cap in caps:
            cap.set_markeredgewidth(cwidth)
        (_, caps, _)=plt.errorbar(segsize, movie_m_bar, yerr=movie_delta_m, marker='.', color='g', linestyle='None', label='GRMHD movie', markersize=msize, elinewidth=eline, capsize=csize)
        for cap in caps:
            cap.set_markeredgewidth(cwidth)
        (_, caps, _)=plt.errorbar(segsize, frame_m_bar_detrend, yerr=frame_delta_m_detrend, marker= '.', color= 'r', linestyle='None', label='Static, differenced', markersize=msize, elinewidth=eline, capsize=csize)
        for cap in caps:
            cap.set_markeredgewidth(cwidth)
        (_, caps, _)=plt.errorbar(segsize, movie_m_bar_detrend, yerr=movie_delta_m_detrend, marker='.', color='black', linestyle='None', label='Movie, differenced', markersize=msize, elinewidth=eline, capsize=csize)
        for cap in caps:
            cap.set_markeredgewidth(cwidth)
    else:
        (_, caps, _)=plt.errorbar(segsize, frame_m_bar, yerr=frame_delta_m, marker= '.', color= 'b', linestyle='None', markersize=msize, elinewidth=eline, capsize=csize)
        for cap in caps:
            cap.set_markeredgewidth(cwidth)
        (_, caps, _)=plt.errorbar(segsize, movie_m_bar, yerr=movie_delta_m, marker='.', color='g', linestyle='None', markersize=msize, elinewidth=eline, capsize=csize)
        for cap in caps:
            cap.set_markeredgewidth(cwidth)
        (_, caps, _)=plt.errorbar(segsize, frame_m_bar_detrend, yerr=frame_delta_m_detrend, marker= '.', color= 'r', linestyle='None', markersize=msize, elinewidth=eline, capsize=csize)
        for cap in caps:
            cap.set_markeredgewidth(cwidth)
        (_, caps, _)=plt.errorbar(segsize, movie_m_bar_detrend, yerr=movie_delta_m_detrend, marker='.', color='black', linestyle='None', markersize=msize, elinewidth=eline, capsize=csize)
        for cap in caps:
            cap.set_markeredgewidth(cwidth) 
        
plt.xlabel('Segment size (h)', fontsize=19)
plt.ylabel(r'$\mathcal{Q}$', fontsize=21)
plt.axhline(y=0, color='black')
plt.legend(fontsize=15, loc="center right",framealpha=0.7)
plt.ylim(-0.15, 1.0)
plt.xlim(0,1.6)

plt.savefig('./qmetric_example.pdf', bbox_inches='tight')
plt.show()
