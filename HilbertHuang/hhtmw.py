import pyhht.emd as hht
import scipy.signal as ss
import scipy.interpolate as interp
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt


class obs:
    def __init__(self, data, Wmax = 10, smooth = 2., rangeHS = -1, file_name = ''):
        self.data = data
        self.file_name = file_name
        self.t = data[:,0]
        self.signal = data[:,1]
        self.Wmax = 2.*np.pi*Wmax
        self.smooth = smooth
        self.lent = len(self.t)
        self.lenW = len(self.t)
        self.treg = np.linspace(self.t[0],self.t[-1],self.lent)
        decomposer = hht.EMD(self.signal)
        self.imfs = decomposer.decompose()
        self.NumImfs = self.imfs.shape[0]
        self.Iimf = [interp.splrep(self.t,self.imfs[x,:]) for x in range(self.NumImfs)]
        self.imf = [interp.splev(self.t,self.Iimf[x]) for x in range(self.NumImfs)]  
        self.Hilb = [ss.hilbert(interp.splev(self.treg,self.Iimf[x])) for x in range(self.NumImfs)]
        self.Amp =  [np.abs(self.Hilb[x]) for x in range(self.NumImfs)]
        self.Phi =  [np.unwrap(np.angle(self.Hilb[x])) for x in range(self.NumImfs)]
        self.IPhi = [(interp.splrep(self.treg,self.Phi[x],s=10.)) for x in range(self.NumImfs)]
        self.DPhiDt = [(interp.splev(self.treg,self.IPhi[x],der=1)) for x in range(self.NumImfs)]
        for x in range(self.NumImfs):
            self.DPhiDt[x] = (self.DPhiDt[x] > 0)*self.DPhiDt[x]
            #self.DPhiDt[x] = (self.DPhiDt[x] < 4.*np.std(self.DPhiDt[x]))*self.DPhiDt[x]

        self.gridW = np.linspace(0,self.Wmax,self.lenW)
        self.extent =[self.t[0],self.t[-1],self.gridW[-1]/2./np.pi,self.gridW[0]/2./np.pi]
        self.HSpectr = np.zeros((self.lenW,self.lent))
        if rangeHS ==-1:
            rangeHS = range(self.NumImfs)

        for couT in range(self.lent):
            for x in rangeHS:    
                blob1d = st.norm.pdf((self.gridW-self.DPhiDt[x][couT])/self.smooth)/self.smooth
                self.HSpectr[:,couT] = self.HSpectr[:,couT] + self.Amp[x][couT]*blob1d

        self.MargHSpectr = np.sum(self.HSpectr,1)


    def ShowHSpectr(self,aspect=0.2):
      
        fig, ax = plt.subplots(figsize=(8,8))
        ax.imshow(np.sqrt(self.HSpectr), extent= self.extent)
        ax.set_aspect(aspect)
        ax.set_xlabel('time [h]',fontsize=15)
        ax.set_ylabel('frequency [1/h]',fontsize=15)
        ax.set_title('Hilbert Spectrum '+self.file_name)
        x = np.linspace(0,4,100)
        y = np.linspace(0,4,100)
        xx, yy = np.meshgrid(x, y)
        #ax.axhline(y=60/35.3, xmin=0., ymax=2e4, linewidth=3, color = 'w')
        plt.show()

    def ShowMargHSpectr(self,ymax=1e4):
        plt.plot(self.gridW/2./np.pi,self.MargHSpectr**2)
        plt.xlabel('frequency [1/h]',fontsize=15)
        plt.axvline(x=60/35.3, ymin=0., ymax=2e4, linewidth=2, color = 'k')
        plt.axvline(x=2*60/35.3, ymin=0., ymax=2e4, linewidth=2, color = 'k')
        plt.title('Marginal Hilbert spectrum '+self.file_name,y=1.05)
        plt.axis([self.gridW[0]/2./np.pi,self.gridW[-1]/2./np.pi,0,np.amax(self.MargHSpectr**2)])
        plt.show()


def ShowFourier(treg,testData,padfr=0, maxFreq = 10., imf_num = -1, file_name = ' '):

    toF =np.lib.pad(testData,(int(padfr*len(testData)),int(padfr*len(testData))), 'constant', constant_values=(0,0))
    freq = np.fft.fftshift(np.fft.fftfreq(len(toF),treg[1]-treg[0]))

    #plt.plot(np.fft.fftshift(freq))
    #plt.show()
    FAbs = np.abs(np.fft.fftshift(np.fft.fft(toF)))**2
    plt.plot(freq,FAbs);
    np.abs(np.fft.fft(toF));
    plt.axis([freq[int(len(freq)/2)+1],maxFreq,0,np.amax(FAbs)/2])
    plt.axvline(x=60/35.3, ymin=0., ymax=2e4, linewidth=2, color = 'k')
    plt.axvline(x=2*60/35.3, ymin=0., ymax=2e4, linewidth=2, color = 'k')
    plt.axvline(x=60/2/35.3, ymin=0., ymax=2e4, linewidth=2, color = 'k')
    plt.xlabel('frequency [1/h]', fontsize=15)
    plt.ylabel('frequency [1/h]', fontsize=15)
    if imf_num != -1:
        plt.title('Fourier spectrum of IMF'+imf_num+'\n'+str(file_name), y= 1.05)
    else:
        plt.title('Fourier spectrum')
    plt.show()


def ShowIMFandFourier(treg,testData,padfr=0, maxFreq = 10., imf_num = -1, file_name = ' '):

    toF =np.lib.pad(testData,(int(padfr*len(testData)),int(padfr*len(testData))), 'constant', constant_values=(0,0))
    freq = np.fft.fftshift(np.fft.fftfreq(len(toF),treg[1]-treg[0]))
    FAbs = np.abs(np.fft.fftshift(np.fft.fft(toF)))**2

    plt.subplot(211)
    plt.plot(treg, testData)
    plt.xlabel('time [h]',fontsize=15)
    plt.ylabel('signal amplitude',fontsize=15)
    plt.title('IMF'+imf_num+'\n'+str(file_name), y= 1.05)
    plt.show()
    
    plt.subplot(212)
    plt.plot(freq,FAbs);
    np.abs(np.fft.fft(toF));
    plt.axis([freq[int(len(freq)/2)+1],maxFreq,0,np.amax(FAbs)/2])
    plt.axvline(x=60/35.3, ymin=0., ymax=2e4, linewidth=2, color = 'k')
    plt.axvline(x=2*60/35.3, ymin=0., ymax=2e4, linewidth=2, color = 'k')
    plt.axvline(x=60/2/35.3, ymin=0., ymax=2e4, linewidth=2, color = 'k')
    plt.xlabel('frequency [1/h]', fontsize=15)
    plt.ylabel('power', fontsize=15)
    if imf_num != -1:
        plt.title('Fourier spectrum of IMF'+imf_num+' '+'\n'+str(file_name), y= 1.05)
    else:
        plt.title('Fourier spectrum')
    plt.show()    



def Show2IMFand2Fourier(treg,testData,treg2,testData2,padfr=0, maxFreq = 10., imf_num = -1, file_name = ' ', file2_name = ' '):

    toF =np.lib.pad(testData,(int(padfr*len(testData)),int(padfr*len(testData))), 'constant', constant_values=(0,0))                 
    freq = np.fft.fftshift(np.fft.fftfreq(len(toF),treg[1]-treg[0]))
    FAbs = np.abs(np.fft.fftshift(np.fft.fft(toF)))**2
    toF2 =np.lib.pad(testData2,(int(padfr*len(testData2)),int(padfr*len(testData2))), 'constant', constant_values=(0,0))
    freq2 = np.fft.fftshift(np.fft.fftfreq(len(toF2),treg2[1]-treg2[0]))
    FAbs2 = np.abs(np.fft.fftshift(np.fft.fft(toF2)))**2
    plt.subplot(211)
    a2 = plt.plot(treg, testData,label= 'non-continuous')
    a1 = plt.plot(treg2,testData2,label = 'continuous')
    plt.xlabel('time [h]',fontsize=15)
    plt.ylabel('signal amplitude',fontsize=15)
    plt.title('IMF'+imf_num+'\n'+str(file_name)+'\n'+str(file2_name), y= 1.05)
    plt.legend()
    plt.show()
    
    plt.subplot(212)
    plt.plot(freq,FAbs, label = 'non-continuous');
    plt.plot(freq2,FAbs2, label = 'continuous');
    np.abs(np.fft.fft(toF));
    plt.axis([freq[int(len(freq)/2)+1],maxFreq,0,np.amax(FAbs)/2])
    plt.axvline(x=60/35.3, ymin=0., ymax=2e4, linewidth=2, color = 'k')
    plt.axvline(x=2*60/35.3, ymin=0., ymax=2e4, linewidth=2, color = 'k')
    plt.axvline(x=60/2/35.3, ymin=0., ymax=2e4, linewidth=2, color = 'k')
    plt.xlabel('frequency [1/h]', fontsize=15)
    plt.ylabel('power', fontsize=15)
    plt.legend()
    if imf_num != -1:
        plt.title('Fourier spectrum of IMF'+imf_num+' '+'\n'+str(file_name)+'\n'+str(file2_name), y= 1.05)
    else:
        plt.title('Fourier spectrum')
    plt.show()    


def LombScargle(time, data,period_range):
    gridSize = 10000
    from gatspy import datasets, periodic
    model = periodic.LombScargleFast(fit_period=True)
    model.optimizer.period_range = period_range
    model.fit(time,data)
    periods = np.linspace(period_range[0], period_range[-1],gridSize)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = model.score(periods)
    bestF = 1./model.best_period
    print('Best-fit frequency L-S: ',bestF)

    return periods,scores,bestF


def ShowLombScargle2(periods, scores,periods2, scores2,imf_num,file_name = ' ', file2_name = ' '):

    #periods, scores = LombScargle(time,data,period_range)
    #periods2, scores2 = LombScargle(time2,data2,period_range)

    plt.plot(1/periods,scores,label='L-S of module IMF'+str(imf_num))
    plt.plot(1/periods2,scores2,label='standard L-S')
    plt.axis([1./periods[-1],1./periods[0],0,np.amax([np.amax(scores2),np.amax(scores)])])
    plt.axvline(x=60/35.3, ymin=0., ymax=2e4, linewidth=2, color = 'k')
    plt.axvline(x=2*60/35.3, ymin=0., ymax=2e4, linewidth=2, color = 'k')
    plt.axvline(x=60/2/35.3, ymin=0., ymax=2e4, linewidth=2, color = 'k')
    plt.xlabel('frequency [1/h]', fontsize=15)
    plt.ylabel('power Lomb-Scargle', fontsize=15)
    plt.title('Lomb-Scargle spectrum'+imf_num+' '+'\n'+str(file_name)+'\n'+str(file2_name), y= 1.05)
    plt.legend()
    plt.show()

    
