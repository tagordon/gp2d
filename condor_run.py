from Modeling import model
from mygp import gp, kernel
import numpy as np
import batman 
import warnings
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pl
import utilities
import sys

ident = sys.argv[1]

pl.style.use('seaborn-deep')
warnings.simplefilter('ignore')


# transit model 
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

# load data
basedir = '/astro/users/tagordon/multigp/data/'
data = np.loadtxt(basedir + 'tyler' + ident + '_b_binned.txt')
t = data[0] - np.median(data[0])
data = data[::-1]
data = data[:-1]

# fit variability and print best fit parameters
k1_params = [0.2, 6e9]
k2_params = [5, 20]
white_noise = [2e3]*3
offsets = [np.median(data[0]), np.median(data[1]), np.median(data[2])]
mean_params = [0.0, 0.0, 0.0, 0.1]
transit_model = model(k1_params, k2_params, white_noise, 
                      mean_params, offsets, transit, 
                      kernel.celerite_root2_kernel, 
                      kernel.scale_kernel, 3)
transit_model.fit_variability(data, t, method='L-BFGS-B')
transit_model.print_params()

def like(t0, rp, dur):
    transit_model.update_gp(mean_params=[0, rp, t0, dur])
    transit_model.gp.compute(t)
    return transit_model.gp.log_likelihood(data, t)

rp = np.linspace(0, 0.05, 10)
t0 = np.linspace(t.min(), t.max(), 100)
dur = np.linspace(0.01, 0.1, 10)

fig, axs = pl.subplots(len(dur), 2, figsize=(20, 10*len(dur)))
for i, d in enumerate(dur):
    
    f = lambda t0, rp: like(t0, rp, d)
    likelihood, fig = utilities.plot_likelihood(f, t0, rp, plot=False)
    axs[i][0].pcolormesh(t0, rp, likelihood)
    axs[i][1].pcolormesh(t0, rp, likelihood, vmin=np.max(likelihood)-100)
    axs[i][0].set_title('duration {:.2f}'.format(d))
    axs[i][1].set_title('duration {:.2f}'.format(d))

    max_like = np.max(likelihood)
    BIC_transit = transit_model.BIC_like(data, t, max_like, 12)
    transit_model.update_gp(mean_params=[0, 0, 0, 0.01])
    transit_model.gp.compute(t)
    BIC_no_transit = transit_model.BIC_like(data, t, transit_model.gp.log_likelihood(data, t), 10)
    print("BIC ratio (transit/no transit) for duration {:.2f}: {:.2f}".format(d, BIC_transit/BIC_no_transit))
    
pl.savefig('like' + ident + '.png')