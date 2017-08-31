# Program to calculate the Q-metric and error.
# Freek Roelofs, 2017/08/10

import numpy as np
import matplotlib.pyplot as plt

rad2deg=180/np.pi
deg2rad=np.pi/180


class Qmetric(object):

    def gauss(self, sigma): 
        """Random number from Gaussian with zero mean and standard deviation sigma"""
        return np.random.normal(loc=0,scale=sigma)
    
    def circ_stat(self, angles):
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
    
    def drange(self, start, stop, step):
        """Get list of time stamps where data should be split for a certain segment size"""
        r = start
        while r < stop:
            yield r
            r += step
    
    def indices(self, time, segsize):
        """Generate list of indices where list of data points should be split for a certain segment size"""
        indexlist=[0]
        hrs=self.drange(min(time), max(time), segsize)
        for hr in hrs:
            inthr=min(time, key=lambda x:abs(x-hr))
            if int(np.where(time==inthr)[0]) ==0:
                continue
            else:
                indexlist.append(int(np.where(time==inthr)[0]))
        return indexlist
    
    def segment(self, seq, indexlist):
        """Split data sequence in multiple segments at indices specified in indexlist"""
        out = []
        for i in range(len(indexlist)):
            if i==len(indexlist)-1:
                out.append(seq[indexlist[i]:])
            else:
                out.append(seq[indexlist[i]:indexlist[i+1]])
        return out
    
    def difflag(self, series, sigma, n):
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
        
    def eps(self, err):
        """Calculate tilde{epsilon} using Monte Carlo approach assuming Gaussian errors"""
        epsilons=np.zeros(1000)
        for i in range(1000):
            this_it=np.zeros(len(err))
            for j in range(len(err)):
                this_it[j]=self.gauss(err[j]*deg2rad)
            this_cosi=[np.cos(x) for x in this_it]
            this_sini=[np.sin(x) for x in this_it]
            this_cos_avg=np.mean(this_cosi)
            this_sin_avg=np.mean(this_sini)
            this_R=np.sqrt(this_sin_avg**2+this_cos_avg**2)
            this_obs_sigma=np.sqrt(-2*np.log(this_R))*rad2deg
            epsilons[i]=this_obs_sigma
        epsi=np.mean(epsilons)
        return epsi
    
    #def cldata(self, datfile, segsize, difsize, detrend):
    def cldata(self, cp, segsize, difsize, detrend):
        """Main function to calculate q-metric and error"""
    
        #Load data   
        #data=np.loadtxt(datfile)    
        #time=data[:,0]
        #obs_cphase=data[:,1]
        #ntot=float(len(obs_cphase))
        #obs_sigmacp=data[:,2]
    
        time=cp[0]
        obs_cphase=cp[1]
        ntot=float(len(obs_cphase))
        obs_sigmacp=cp[2]

        #Plot input data
    
        #Split in segments
        indexlist=self.indices(time, segsize)
        time_seg=self.segment(time, indexlist)
        obs_cphase_seg=self.segment(obs_cphase, indexlist)
        obs_sigmacp_seg=self.segment(obs_sigmacp, indexlist)
    
        #Calculate circular mean, variance, \tilde{epsilon} for each segment
        obs_stdev_seg=[]
        obs_n_seg=[]
        obs_avgerr_seg=[]
        obs_metric_seg=[]
        obs_metric_seg_err=[]
    
        for i in range(len(time_seg)):
            if detrend==True:
                cpdet, sigmadet=self.difflag(obs_cphase_seg[i][:],obs_sigmacp_seg[i][:],difsize)
    
                if len(cpdet)<1:
                    continue
    
                #Detrend
                obs_mean, obs_variance, obs_stdev = self.circ_stat(cpdet)
                obs_avgerr=self.eps(sigmadet)
                this_n=len(cpdet)
            else:
                obs_mean, obs_variance, obs_stdev = self.circ_stat(obs_cphase_seg[i][:])
                obs_avgerr=self.eps(obs_sigmacp_seg[i][:])
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



