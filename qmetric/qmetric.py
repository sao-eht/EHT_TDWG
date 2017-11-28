# Program to calculate the Q-metric and error for an example simulation of closure phases from a GRMHD movie and static source. The data set is divided in multiple segments and Q is calculated using equations 21 and 26-28 of Roelofs et al. (2017). The outcome with the example data set is Figure 6 of Roelofs et al. (2017).
# Maciek Wielgus, 2017/11/28, maciek.wielgus@gmail.com
# based on Freek Roelofs et al., ApJ 847, Quantifying Intrinsic Variability of Sagittarius A* Using Closure Phase Measurements of the Event Horizon Telescope
import numpy as np

def qmetric(datfile, bintime=0, segtime=0, diftime=0, product='cphase',detrend_deg=-1,diff_accuracy = 0.1):
    """Main function to calculate q-metric and error
    bintime - binning period for unevenly sampled data
    defaultly calculated as median difference between observations
    segtime - duration of segment for detrending purposes
    diftime - distance between points used for differentiation
    product - what statistics
    detrend_deg - degree of polynomial used for segment-wise detrending
    -1 = no detrending, 0 = constant, 1 = linear, ...
    diff_accuracy = 
    """
    #Load data   
    data=np.loadtxt(datfile)    
    time=data[:,0] #units? #HOURS
    obs=data[:,1] #time series (e.g. closure phase)
    obs_err=data[:,2] #obs measurement error

    #CHECK IF DATA IS SAMPLED UNIFORMLY
    #IF IT IS, binning==False
    median_diff = np.median(np.diff(time))
    if bintime==0:
            bintime = median_diff#estimate bintime
    binning = not all((np.diff(time) - np.mean(np.diff(time)))==0)
    #IF UNUNIFORMLY SAMPLED DATA DO THE BINNING
    if binning==True:
        print('Non-uniform sampling detected, binning the data with bintime = %s' %str(bintime))
        bins = np.arange(min(time)-bintime/2., max(time)+bintime+1., bintime)
        digitized = np.digitize(time, bins, right=False) # Assigns a bin number to each of the closure phases
        bin_times = []
        bin_means = []
        bin_errors = []
        for bin in range(len(bins)+1):
            if len(obs[digitized==bin])>0:
                mean_local = circular_mean_weights(obs[digitized==bin],obs_err[digitized==bin])
                err_local = np.sqrt(np.sum(obs_err[digitized==bin]**2))/len(obs_err[digitized==bin])
                bin_times.append(0.5*(bins[bin-1] + bins[bin]) )
                bin_means.append(mean_local)
                bin_errors.append(err_local)
        time = np.asarray(bin_times)
        obs = np.asarray(bin_means)
        obs_err= np.asarray(bin_errors)
    else:
        print('Data uniformly sampled, no binning')

    #DIFFERENTIATING THE DATA
    if diftime>0:
        time,obs, obs_err = diff_time(time,obs,obs_err, diftime, accuracy = diff_accuracy)
        #print(obs)

    #SEGMENTATION OF DATA
    if segtime==0:
        if detrend_deg>-1:
            obs = detrending_polyfit(time,obs,detrend_deg)
        q, dq, n = find_q_basic(obs,obs_err)
    else:
        print('Segmenting data with segtime = %s' %str(segtime))
        segments = np.arange(min(time), max(time)+segtime, segtime)
        digitized = np.digitize(time, segments, right=False) # Assigns a segment number to each measurement
        time_segments = []
        obs_segments = []
        obs_err_segments = []
        for cou in range(1,len(segments)+1):
            time_local = time[digitized==cou]
            obs_local = obs[digitized==cou]
            obs_err_local = obs_err[digitized==cou]
            N = 10.
            if len(time_local)>N: #let's have at least N datapoints in each segment
                time_segments.append(time_local)
                if detrend_deg>-1:
                    obs_local = detrending_polyfit(time_local,obs_local,detrend_deg)
                obs_segments.append(obs_local)
                obs_err_segments.append(obs_err_local)
        #Calculate q metric in each segment
        sig2_segments = []
        eps2_segments = []
        n_segments = []
        tot_segments = len(time_segments)
        #print(time_segments)
        for cou in range(tot_segments):
            #q_loc, dq_loc, n_loc = find_q_basic(obs_segments[cou],obs_err_segments[cou])
            sig_loc, eps_loc, n_loc = find_sig_eps_basic(obs_segments[cou],obs_err_segments[cou],product)
            sig2_segments.append(sig_loc**2)
            eps2_segments.append(eps_loc**2)
            n_segments.append(float(n_loc))
        sig2_global = np.average(np.asarray(sig2_segments),weights=np.asarray(n_segments))
        eps2_global = np.average(np.asarray(eps2_segments),weights=np.asarray(n_segments))
        n_tot = np.sum(np.asarray(n_segments))
        q = (sig2_global - eps2_global)/sig2_global
        dq =  np.sqrt(2./n_tot)*eps2_global*np.sqrt(np.average( (np.asarray(sig2_segments))**2,weights=np.asarray(n_segments)))/sig2_global**2 
    return q, dq

def find_q_basic(obs,obs_err,product='cphase'):
    #most basic function to get q metric
    #no binning or detrending
    obs = np.asarray(obs)
    obs_err = np.asarray(obs_err)
    if product=='cphase':
        obs_sigma = circular_std(obs)
        eps_thermal = eps_analytic(obs_err)
        #eps_thermal = eps_MC(obs_err)
    elif product=='camplitude':
        obs_sigma = 1.#place holder
        eps_thermal = 1.#place holder
    n = len(obs)
    if n > 0:
        q=(obs_sigma**2- eps_thermal**2)/obs_sigma**2
        dq = np.sqrt(2./n)*(1-q)
    else:
        q = np.nan
        dq = np.nan
    return q, dq, n
    

def find_sig_eps_basic(obs,obs_err,product='cphase'):
    #most basic function to get q metric
    #no binning or detrending
    obs = np.asarray(obs)
    obs_err = np.asarray(obs_err)
    if product=='cphase':
        obs_sigma = circular_std(obs)
        eps_thermal = eps_analytic(obs_err)
    elif product=='camplitude':
        obs_sigma = np.std(obs)#place holder
        eps_thermal = (np.sum(obs_err))/len(obs_err) #place holder
    n = len(obs)
    return obs_sigma, eps_thermal, n

def eps_MC(err):
    deg2rad = np.pi/180.
    rad2deg = 180./np.pi
    """Calculate tilde{epsilon} using Monte Carlo approach assuming Gaussian errors"""
    N = int(1e3)
    epsilons=np.zeros(N)
    for i in range(N):
        this_it=np.zeros(len(err))
        for j in range(len(err)):
            this_it[j]=np.random.normal(0.,err[j]*deg2rad)
        #print(this_it)
        this_cosi=[np.cos(x) for x in this_it]
        this_sini=[np.sin(x) for x in this_it]
        this_cos_avg=np.mean(this_cosi)
        this_sin_avg=np.mean(this_sini)
        this_R=np.sqrt(this_sin_avg**2+this_cos_avg**2)
        #print('R= ',this_R)
        this_obs_sigma=np.sqrt(-2.*np.log(this_R))*rad2deg
        #print('eps= ',this_obs_sigma)
        epsilons[i]=this_obs_sigma
    epsi=np.mean(epsilons)
    return epsi

def eps_analytic(err):
    #err in deg
    err = np.asarray(err)
    R = np.mean( np.exp(-(err*np.pi/180.)**2/2.) )
    eps = np.sqrt(-2*np.log(R))*180./np.pi
    return eps

def stR(theta):
#i/o in degrees
    theta = np.asarray(theta)*np.pi/180.
    C = np.mean(np.cos(theta))
    S = np.mean(np.sin(theta))
    R = np.sqrt(C**2+S**2)
    st = np.sqrt(-2.*np.log(np.sqrt(C**2+S**2)))*180./np.pi
    return st,R

def R_from_st(st):
    #st in deg
    R = np.exp(-(st*np.pi/180.)**2/2.)
    return R

def detrending_polyfit(time,obs,deg=1):
    #just subtract linear fit inside segment
    time = np.asarray(time)
    obs = np.asarray(obs)
    fit = np.polyfit(time, obs, deg)
    obs = obs - np.polyval(fit,time)
    return obs

def diff_time(time,obs,err, dt, accuracy = 0.1):
    time_new = []
    obs_new = []
    err_new = []
    time = np.asarray(time)
    for cou in range(len(time)):
        delta = np.abs(time[cou]+dt - time)
        ind = np.argmin(delta)
        if delta[ind]<accuracy*dt:
            time_new.append(time[cou])
            obs_new.append(obs[ind] - obs[cou])
            err_new.append( np.sqrt(err[ind]**2+err[cou]**2) )
    return np.asarray(time_new),np.asarray(obs_new),np.asarray(err_new)


def inflate_noise(datfile, inflation_factor, savefile = ''):
    #inflates noise by factor sqrt(inflation_factor**2 + 1) 
    data=np.loadtxt(datfile)    
    time=data[:,0] #units? #HOURS
    obs=data[:,1] #time series (e.g. closure phase)
    obs_err=data[:,2] #obs measurement error

    obs_err = np.asarray(obs_err)*np.sqrt(1.+inflation_factor**2)
    obs_noise = [np.random.normal(0.,obs_err[x]) for x in range(len(obs_err))]
    obs = np.asarray(obs) + np.asarray(obs_noise)
    data = np.asarray([time, obs, obs_err])
    if savefile=='':
        savfile=datfile
    np.savetxt(savefile,np.transpose(data))

def add_noise(datfile, noise_sd, savefile = ''):
    #inflates noise by factor sqrt(inflation_factor**2 + 1) 
    data=np.loadtxt(datfile)    
    time=data[:,0] #units? #HOURS
    obs=data[:,1] #time series (e.g. closure phase)
    obs_err=data[:,2] #obs measurement error

    obs_err = [noise_sd]*len(obs_err)
    obs_noise = np.random.normal(0.,noise_sd,len(obs))
    obs = np.asarray(obs) + np.asarray(obs_noise)
    data = np.asarray([time, obs, obs_err])
    if savefile=='':
        savfile=datfile
    np.savetxt(savefile,np.transpose(data))


#####---------------------------------
#FUNCTIONS FOR CALCULATING STATISTICS
#####---------------------------------

def circ_mean_weights(angles, err='ones'):
    """Calculate circular average for list of angles + errors"""
    if str(err)=='ones':
        err = np.ones(len(angles))
    cos=np.zeros(len(angles))
    sin=np.zeros(len(angles))
    weights=np.zeros(len(angles))
    for i in range(len(angles)):
        cos[i]=np.cos(angles[i]*deg2rad)
        sin[i]=np.sin(angles[i]*deg2rad)
        err[i] *= deg2rad
        weights[i] = 1./err[i]**2
    cos_avg=np.average(cos, weights=weights)
    sin_avg=np.average(sin, weights=weights)
    obs_mean=np.arctan2(sin_avg, cos_avg)*rad2deg
    if len(angles) == 1:
        obs_stdev = err[0] * rad2deg
    else:
        #obs_variance=1-np.sqrt(sin_avg**2+cos_avg**2)
        #obs_stdev=np.sqrt(-2*np.log((1-obs_variance))/len(angles))*rad2deg
        s=0
        for i in range(len(err)):
            s += err[i]**2
        obs_stdev=np.sqrt(s)/len(err) * rad2deg
    
    return obs_mean, obs_stdev

def circular_mean_weights(theta,err='ones'):
    #i/o in degrees
    theta = np.asarray(theta)*np.pi/180. #to radians
    if str(err)=='ones':
        err = 0.*theta+1.
    err = np.asarray(err)*np.pi/180. #to radians
    weights = 1./np.asarray(err**2)
    C = np.average(np.cos(theta),weights=weights)
    S = np.average(np.sin(theta),weights=weights)
    mean = np.arctan2(S,C)*180./np.pi
    return mean

def circular_mean(theta):
    #i/o in degrees
    theta = np.asarray(theta)*np.pi/180.
    C = np.mean(np.cos(theta))
    S = np.mean(np.sin(theta))
    mt = np.arctan2(S,C)*180./np.pi
    return mt

def circular_std(theta):
    #i/o in degrees
    theta = np.asarray(theta)*np.pi/180.
    C = np.mean(np.cos(theta))
    S = np.mean(np.sin(theta))
    st = np.sqrt(-2.*np.log(np.sqrt(C**2+S**2)))*180./np.pi
    return st

def circular_std_of_mean(theta):
    #i/o in degrees
    theta = np.asarray(theta)*np.pi/180.
    C = np.mean(np.cos(theta))
    S = np.mean(np.sin(theta))
    st = np.sqrt(-2.*np.log(np.sqrt(C**2+S**2)))*180./np.pi/np.sqrt(len(theta))
    return st

def diff_side(x):
    x = np.asarray(x)
    xp = x[1:]
    xm = x[:-1]
    xdif = xp-xm
    dx = np.angle(np.exp(1j*xdif*np.pi/180.))*180./np.pi
    return dx

def circular_std_dif(theta):
    theta = np.asarray(theta)*np.pi/180.
    dif_theta = diff_side(theta)
    C = np.mean(np.cos(dif_theta))
    S = np.mean(np.sin(dif_theta))
    st = np.sqrt(-2.*np.log(np.sqrt(C**2+S**2)))*180./np.pi/np.sqrt(2.)
    return st

def circular_std_of_mean_dif(theta):
    theta = np.asarray(theta)*np.pi/180.
    dif_theta = diff_side(theta)
    C = np.mean(np.cos(dif_theta))
    S = np.mean(np.sin(dif_theta))
    st = np.sqrt(-2.*np.log(np.sqrt(C**2+S**2)))*180./np.pi/np.sqrt(len(theta))/np.sqrt(2.)
    return st