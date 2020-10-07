from src.model import HHModel
import numpy as np
import matplotlib.pyplot as plt


I_inp = 0.1
dt = 0.01
niter = 10000
I_pat = np.ones(niter)
hh = HHModel(dt, niter)
hh(I_inp, I_pat)
hh.plot('images/simulation_Iinp{inp}_constant.png'.format(inp = I_inp))
