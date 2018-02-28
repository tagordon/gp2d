# generates curves with and without gaussian signals, stores parameters in separate file.

import numpy as np
from mygp import gp, kernel
import sys
import matplotlib.pyplot as pl

# takes readable params and makes a long list out of them
def make_params(n_samples, log_A=None, log_l=None, log_wn=None, c=None, mean_params=None, offset=None):
    if n_samples > 1:
        return np.concatenate(([log_A], [log_l], log_wn, [1], c, mean_params, offset))
    elif n_samples == 1:
        return np.concatenate(([log_A], [log_l], [log_wn], mean_params, [offset]))

# does the opposite
def unpack_params(params, n_samples, n_mean_params):
    
    if n_samples > 1:
        lengths = 1, 1, n_samples, n_samples, n_mean_params, n_samples
    elif n_samples == 1:
        lengths = 1, 1, 1, n_mean_params, 1
        
    breaks = [sum(lengths[:i]) for i in range(len(lengths))]
    breaks = np.concatenate((breaks, [len(params)]))
    return [params[breaks[i]:breaks[i+1]] for i in range(len(breaks)-1)]
        

# sample a GP with the given parameters, mean, and number of samples (at points t)
def make_data(params, mean_func, n_samples, n_mean_params, t):
    
    if n_samples > 1:
        log_A, log_l, log_wn, c, mean_params, offset = unpack_params(params, n_samples, n_mean_params)    
        k2 = kernel.scale_kernel([scale for scale in c])
    else:
        log_A, log_l, log_wn, mean_params, offset = unpack_params(params, n_samples, n_mean_params)
        k2 = None

    def make_mean(off): return (lambda x: mean_func(x, off, *mean_params))
    mean = [make_mean(off) for off in offset]
    
    k1 = kernel.exp_sq_kernel(np.exp(log_l), np.exp(log_A))
    
    dim = (n_samples > 1) + 1
    g = gp(mean, k1, kernel2=k2, wn=[np.exp(lwn) for lwn in log_wn], dim=dim)
    g.compute(t)
    data = g.sample()
    return data 

# print the parameters nicely 
def print_params(params, n_samples, n_mean_params):
    if n_samples > 1:
        log_A, log_l, log_wn, c, mean_params, offset = unpack_params(params, n_samples, n_mean_params)
    elif n_samples == 1:
        log_A, log_l, log_wn, mean_params, offset = unpack_params(params, n_samples, n_mean_params)
    print('A: ', np.exp(log_A))
    print('l: ', np.exp(log_l))
    print('white noise: ', np.exp(log_wn))
    if n_samples > 1:
        print('scale factors: ', c)
    print('mean parameters: ', mean_params)
    print('offsets: ', offset)

    
# random parameters
n = int(sys.argv[1])
n_samp = int(sys.argv[2])
t_max = 10

A = np.random.rand(n)*20
log_A = np.log(A)
log_l = np.log(np.random.rand(n))
log_wn = np.log(np.random.rand(n, n_samp)*10 + 0.5)
c = np.random.rand(n, n_samp-1)*5
offset = np.random.rand(n, n_samp)*100

# mean params
amp = np.random.rand(n)
amp = [amp*100 + A/2 if (x > 0.5) else amp*(-100) - A/2 for amp, A, x in zip(amp, A, np.random.rand(n))]
t0 = np.random.rand(n)*t_max
sig = np.random.rand(n) + 0.5
mean_params = [[amp, t0, sig] for amp, t0, sig in zip(amp, t0, sig)]

params = [make_params(3, log_A=log_A[i], log_l=log_l[i], log_wn=log_wn[i], c=c[i], mean_params=mean_params[i], offset=offset[i]) for i in range(n)]

t = np.linspace(0, t_max, 100)
tp = np.linspace(0, t_max, 1000)
has_gaussian = np.random.rand(n) > 0.3

def gauss(t, offset, A, t0, sig):
    return offset + A*np.exp(-((t - t0)**2)/(2*sig*sig))

for i in range(n):
    textfilename = 'lc' + str(i) + '.txt'
    imagefilename_nosignal = 'lc_nosignal' + str(i) + '.pdf'
    imagefilename_withsignal = 'lc_withsignal' + str(i) + '.pdf'    
    print(textfilename)
    print('-------')
    
    if has_gaussian[i]:
        data = make_data(params[i], gauss, 3, 3, t)
        print_params(params[i], n_samp, 3)
    else:
        data = make_data(params[i], lambda x1, x2, x3, x4, x5: 0, 3, 3, t)
        print('no gaussian signal')
        print_params(params[i], n_samp, 3)
    print('---------------------------------------------------------------')
    y1, y2, y3 = data
    
    # plot without signal
    fig = pl.figure(figsize=(12, 8))
    pl.plot(t, y1, 'o')
    pl.plot(t, y2, 'o')
    pl.plot(t, y3, 'o')
    pl.savefig(imagefilename_nosignal)
    fig.clf()
    
    # overplot signal
    fig = pl.figure(figsize=(12, 8))
    pl.plot(t, y1, 'o')
    pl.plot(t, y2, 'o')
    pl.plot(t, y3, 'o')
    if has_gaussian[i]:
        pl.plot(tp, gauss(tp, 0, *mean_params[i]), linewidth=10, alpha=0.5, color='k')
        pl.figtext(0.2, 0.75, 
                   'mean parameters\n amplitude: {:.3f}\n t0:'.format(amp[i]) + 
                   '{:.3f}\n sigma: {:.3f}'.format(t0[i], sig[i]))
    else:
        pl.axhline(0, linewidth=10, alpha=0.5, color='k')
    pl.savefig(imagefilename_withsignal)
    fig.clf()
    
    data.append(t)
    np.savetxt(textfilename, data)
    