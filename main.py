from src.model import HHModel
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from tqdm import tqdm
import os
"""
I_inp = 0.002
dt = 0.01
niter = 10000
I_pat = np.ones(niter)
hh = HHModel(dt, niter)
hh(I_inp, I_pat)
hh.plot('images/simulation_Iinp{inp}_constant.png'.format(inp = I_inp))
"""
v_avg = []
m_avg = []
n_avg = []
h_avg = []
gNa_avg = []
gK_avg = []
def find_freq(hh, niter, dt, div):
    peaks, _ = sig.find_peaks(hh.v_hist)
    spikes = [0]
    pos = 0
    """"
    T = niter*div
    window = T
    for index in peaks:
        if(index < window):
            if(v_arr[index] > 0):
                spikes[pos] += 1
        else:
            window += T
            pos += 1
            spikes.append(0)
            if(v_arr[index] > 0):
                spikes[pos] += 1
    count = 0
    for num in spikes:
        count += num
    count = count/len(spikes)
    spike_freq = 0
    if count != 0: 
        spike_freq = count/(T*dt)
    return spike_freq
    """
    spikes = []
    v = 0
    m = 0
    n = 0
    h = 0
    gNa = 0
    gK = 0
    for peak in peaks:
        v += hh.v_hist[peak]
        m += hh.m_hist[peak]
        n += hh.n_hist[peak]
        h += hh.h_hist[peak]
        gNa += hh.gNa_hist[peak]
        gK += hh.gK_hist[peak]
        if hh.v_hist[peak]>0:
            spikes.append(peak)
    v_avg.append(v/len(peaks))
    n_avg.append(n/len(peaks))
    m_avg.append(m/len(peaks)) 
    h_avg.append(h/len(peaks))
    gNa_avg.append(gNa/len(peaks))
    gK_avg.append(gK/len(peaks))
    return 1000*(len(spikes)/(niter*dt))

div = 0.001

dt = 0.01
niter = 10000
hh = HHModel(dt, niter)
spike_freqs = []
curr = []
for i in tqdm(range(1800)):
    I_inp = i*0.0005
    #I_pat = np.ones(niter)
    #I_pat = np.sin(np.arange(0, niter*0.005, 0.005))
    
    #"""
    I_pat = np.ones(niter)
    count = 0
    sign = 1
    for i in range(niter):
        if count%2000 == 0:
            sign = -1*sign
        count +=1
        I_pat[i] = I_pat[i]*sign
    #"""
    """
    I_pat = np.zeros(niter)
    for i in range(niter):
        I_pat[i] = (i%2000)/2000
    #"""
    hh = HHModel(dt, niter)
    hh(I_inp, I_pat)
    if not os.path.exists('plots/square_wav_curr/'):
        os.mkdir('plots/square_wav_curr/')
    hh.plot('plots/square_wav_curr/simulation_Iinp{inp}.png'.format(inp = I_inp))
    curr.append(I_inp)
    spike_freqs.append(find_freq(hh, niter, dt, div))
    hh.reset()

fig, axes = plt.subplots(2, 2, figsize = (10, 10))
axes[0][0].plot(curr, spike_freqs)
axes[0][0].set_xlabel('current')
axes[0][0].set_ylabel('spike freq')
axes[0][0].set_title('Current vs Spike Frequency')
axes[0][1].plot(curr, m_avg, 'r', label = 'm')
axes[0][1].plot(curr, n_avg, 'b', label = 'n')
axes[0][1].plot(curr, h_avg, 'g', label = 'h')
axes[0][1].legend()
axes[0][1].set_xlabel('current')
axes[0][1].set_ylabel('average peak channel activation')
axes[0][1].set_title('Average Peak Channel Activations vs Current')
axes[1][0].plot(curr, gNa_avg, 'r', label = 'Na conductance')
axes[1][0].plot(curr, gK_avg, 'b', label = 'K conductance')
axes[1][0].legend()
axes[1][0].set_xlabel('current')
axes[1][0].set_ylabel('conductance(mS/cm^2)')
axes[1][0].set_title('Average Peak Channel Conductance vs Current')
axes[1][1].plot(curr, v_avg)
axes[1][1].set_xlabel('current')
axes[1][1].set_ylabel('average peak voltage')
axes[1][1].set_title('Average Peak Voltage vs Current')
fig.savefig('plots/freq_plot_square_wav_curr.png')
plt.show()
