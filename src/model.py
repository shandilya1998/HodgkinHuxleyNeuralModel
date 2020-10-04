import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 

dtype = np.float128

class HHModel:
    def __init__(self, dt, niter):
        self.gK_max = 0.36
        self.gNa_max = 1.20
        self.vK = -77
        self.vNa = 50
        self.gL = 0.003
        self.vL = -54.387
        self.Cm = 0.01
        self.dt = np.float128(dt)
        self.niter = niter
        self.t = np.arange(self.niter)*dt
        self.m_hist = np.zeros(self.niter)
        self.m = 0.0530
        self.gNa_hist = np.zeros(self.niter, dtype = dtype)
        self.gK_hist = np.zeros(self.niter, dtype= dtype)
        self.gTot_hist = np.zeros(self.niter, dtype= dtype)
        self.v_hist = np.zeros(self.niter, dtype= dtype)
        self.v = -64.9964
        self.h_hist = np.zeros(self.niter, dtype= dtype)
        self.h = 0.5960
        self.n_hist = np.zeros(self.niter, dtype= dtype)
        self.n = 0.3177

    def gNa(self):
        return self.gNa_max*(self.m**3)*self.h

    def gK(self):
        return self.gK_max*(self.n**4)

    def g_total(self, it):
        return self.gNa_hist[it] + self.gK_hist[it] + self.gL

    def v_inf(self, it):
        return ((self.gNa_hist[it]*self.vNa + self.gK_hist[it]*self.vK + self.gL*self.vL)+self.I[it])/self.gTot_hist[it]   
    
    def tauv(self, it):
        return self.Cm/self.gTot_hist[it]

    def _v(self, it):
        v_inf = self.v_inf(it)
        tauv = self.tauv(it)
        return v_inf + (self.v - v_inf)*np.exp(-self.dt/tauv, dtype= dtype)
    
    def alpha_m(self):
        return 0.1*(self.v + 40)/(1-np.exp(-(self.v + 55)/10, dtype= dtype))

    def beta_m(self):
        return 4*np.exp(-0.0556*(self.v + 65), dtype= dtype)

    def alpha_n(self):
        return 0.01*(self.v+55)/(1-np.exp(-(self.v+55)/10, dtype= dtype))

    def beta_n(self):
        return 0.125*np.exp(-(self.v+65)/80, dtype= dtype)

    def alpha_h(self):
        return 0.07*np.exp(-0.05*(self.v+65), dtype= dtype)

    def beta_h(self):
        return 1/(1+np.exp(-0.1*(self.v+35), dtype= dtype))

    def _m(self):
        alpha_m = self.alpha_m()
        beta_m = self.beta_m()
        taum = 1/(alpha_m + beta_m)
        m_inf = alpha_m*taum
        return m_inf + (self.m - m_inf)*np.exp(-self.dt/taum, dtype= dtype)

    def _h(self):
        alpha_h = self.alpha_h()
        beta_h = self.beta_h()
        tauh = 1/(alpha_h + beta_h)
        h_inf = alpha_h*tauh
        return h_inf + (self.h - h_inf)*np.exp(-self.dt/tauh, dtype= dtype)

    def _n(self):
        alpha_n = self.alpha_n()
        beta_n = self.beta_n()
        taun = 1/(alpha_n + beta_n)
        n_inf = alpha_n*taun
        return n_inf + (self.n - n_inf)*np.exp(-self.dt/taun, dtype= dtype)

    def __call__(self, I_inp, I_pat):
        self.I_inp = I_inp
        self.I = I_pat*self.I_inp
        for it in tqdm(range(self.niter)):
            self.gNa_hist[it] = self.gNa()
            self.gK_hist[it] = self.gK()
            self.gTot_hist[it] = self.g_total(it)
            self.v = self._v(it) 
            self.v_hist[it] = self.v
            self.m = self._m()
            self.m_hist[it] = self.m
            self.h = self._h()
            self.h_hist[it] = self.h
            self.n = self._n()
            self.n_hist[it] = self.n

    def plot(self, figname):
        fig, axes = plt.subplots(2, 2, figsize = (10, 10))
        axes[0][0].plot(self.t, self.v_hist)
        axes[0][0].set_xlabel('time(s)')
        axes[0][0].set_ylabel('Voltage(mV)')
        axes[0][0].set_title('Voltage vs Time')
        axes[0][1].plot(self.t, self.m_hist, 'r', label = 'm')
        axes[0][1].plot(self.t, self.n_hist, 'b', label = 'n')
        axes[0][1].plot(self.t, self.h_hist, 'g', label = 'h')
        axes[0][1].legend()
        axes[0][1].set_xlabel('time(s)')
        axes[0][1].set_ylabel('channel activation')
        axes[0][1].set_title('Channel Activations vs Time')
        axes[1][0].plot(self.t, self.gNa_hist, 'r', label = 'Na conductance')
        axes[1][0].plot(self.t, self.gK_hist, 'b', label = 'K conductance')
        axes[1][0].legend()
        axes[1][0].set_xlabel('time(s)')
        axes[1][0].set_ylabel('conductance(mS/cm^2)')
        axes[1][0].set_title('Channel Conductance vs Time')
        axes[1][1].plot(self.t, self.I)
        axes[1][1].set_xlabel('time(s)')
        axes[1][1].set_ylabel('current(mA)')
        axes[1][1].set_title('Current vs Time')
        fig.savefig(figname)
        plt.show()
