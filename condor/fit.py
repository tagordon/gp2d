from Modeling import model
from mygp import gp, kernel
import numpy as np
import batman 
import warnings
import matplotlib.pyplot as pl
pl.style.use('seaborn-deep')
import copy
import warnings
warnings.simplefilter('ignore')

import sys
sys.path.insert(0, '/Users/tgordon/research/libra/')
from libra import ObservationArchive, nirspec_pixel_wavelengths
import corner
import utilities

def transit(t, log_off, b, rp, t0, dt):
    
    t = np.array(t)
    T0 = dt*np.sqrt(1-b**2)/(1 - b**2 + rp)
    
    params = batman.TransitParams()
    params.t0 = t0                       
    params.per = 100                      
    params.rp = rp                      
    params.a = params.per/(T0*np.pi)                       
    params.inc = np.arccos(b/params.a)*(180/np.pi)                     
    params.ecc = 0.                      
    params.w = 90. 
    
    u1, u2 = 0.1, 0.3
    d4 = -u2
    d2 = u1 + 2*u2
    d1, d3 = 0, 0
    
    params.u = [u1, u2]                
    params.limb_dark = "quadratic" 
    
    m = batman.TransitModel(params, t)    
    return log_off*m.light_curve(params)

basedir = '/astro/users/tagordon/multigp/data/'
data = np.loadtxt(basedir + 'tyler' + sys.argv[1] + '_b_binned_small.txt')
t = data[0] - np.median(data[0])
data = data[::-1]
data = data[:-1]

k1_params = [0.2, 5e9]
k2_params = [5, 20]
white_noise = [2e3]*3
offsets = [np.median(data[0]), np.median(data[1]), np.median(data[2])]
mean_params = [0.0, 0.0, 0.0, 0.1]
initial_fit = model(k1_params, k2_params, white_noise, mean_params, offsets, transit, 
                    kernel.celerite_root2_kernel, kernel.scale_kernel, 3)

def map_likelihood(center, duration, depth, model):
    
    log_like = np.zeros((len(duration), len(center)))
    
    for i, c in enumerate(center):
        for j, d in enumerate(duration):
            model = copy.deepcopy(initial_fit)
            mean_params = [0, depth, c, d]
            model.update_gp(mean_params=mean_params)
            initial_fit.gp.compute(t)
            log_like[j][i] = initial_fit.gp.log_likelihood(data, t)
    return log_like

def plot_likelihood_map(log_like, center, duration, vmin=None):
    fig = pl.figure(figsize=(10, 10))
    pl.pcolormesh(center, duration, log_like, vmin=vmin)
