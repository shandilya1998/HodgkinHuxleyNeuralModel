import numpy as np

class HHModel:
    def __init__(self, I_inp, dt = 0.01, niter = 10000):
        self.gKmax = 0.36
        self.gNamax = 1.20
        self.vK = -77
        self.vNa = 50
        self.gL = 0.003
        self.vL = -54.387
        self.Cm = 0.01
        self.dt = dt
        self.niter = niter
        self.I_inp = I_inp
        self.t = np.arange(self.niter)*dt
        self.I = np.ones(niter)*self.I_inp
