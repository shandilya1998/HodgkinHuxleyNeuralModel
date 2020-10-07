from src.model import HHModel
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from tqdm import tqdm
"""
I_inp = 0.002
dt = 0.01
niter = 10000
I_pat = np.ones(niter)
hh = HHModel(dt, niter)
hh(I_inp, I_pat)
hh.plot('images/simulation_Iinp{inp}_constant.png'.format(inp = I_inp))
"""
def find_freq( v_arr, niter, dt, div):
    peaks, _ = sig.find_peaks(v_arr)
    spikes = [0]
    pos = 0
    T = niter*div
    window = T
    for index in peaks:
        if(v_arr[index] > 0 and index < window):
            spikes[pos] += 1
        elif window < index:
            window += T
            pos += 1
            spikes.append(0)
            if(v_arr[index] > 0):
                spikes[pos] += 1
    #print(spikes)
    count = 0
    for num in spikes:
        count += num
    count = count/len(spikes)
    spike_freq = 0
    if count != 0: 
        spike_freq = count/(T*dt)
    return spike_freq


div = 0.1

dt = 0.01
niter = 10000
hh = HHModel(dt, niter)
spike_freqs = []
curr = []
for i in tqdm(range(400)):
    I_inp = i*0.003
    I_pat = np.ones(niter)
    hh = HHModel(dt, niter)
    hh(I_inp, I_pat)
    hh.plot('images/simulation_Iinp{inp}_constant.png'.format(inp = I_inp))
    hh.reset()
    curr.append(I_inp)
    spike_freqs.append(find_freq(hh.v_hist, niter, dt, div))

fig, axes = plt.subplots(1, 1)
axes.plot(curr, spike_freqs)
axes.set_xlabel('current')
axes.set_ylabel('spike freq')
axes.set_title('Current vs Spike Frequency')
fig.savefig('freq_plot.png')
plt.show()
