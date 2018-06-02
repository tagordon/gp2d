import numpy as np
import matplotlib.pyplot as pl

def psd(y, t):
    f = np.fft.rfftfreq(len(t), t[1] - t[0])
    fft = np.fft.rfft(y)
    fft *= np.conj(fft)
    return f, fft.real / len(y)**2

def plot_psd(data, t):
    
    pl.figure()
    for d in data:
        f, fft = psd(d, t)
        pl.loglog(f, fft, 'o')
    pl.title("Power Spectra")
    pl.xlabel(r"Frequency (dats$^{-1}$)")

def plot_likelihood(func, xarr, yarr, plot=True, vmin=None):
    
    like = np.zeros((len(yarr), len(xarr)))
    for i, x in enumerate(xarr):
        for j, y in enumerate(yarr):
            like[j][i] = func(x, y)
    if plot:
        if vmin is 'auto':
            vmin = np.max(like) - 100
        fig = pl.figure(figsize=(10, 10))
        pl.pcolormesh(xarr, yarr, like, vmin=vmin)
        return like, fig
    else:
        return like, None

def plot_samples(data, t, title=None, mean_func=None, mean_params=None):
    
    fig, axs = pl.subplots(1, 2, figsize=(20, 6))
    fig.set_facecolor('white')
    
    for d in data:
        axs[0].plot(t, d, 'o')
    
    for d in data:
        axs[1].plot(t, d/np.median(d), 'o')
        
    if title is not None:
        pl.suptitle(title)
    axs[0].set_xlabel('time (JD)')
    axs[1].set_xlabel('time (JD)')
    axs[0].set_ylabel('flux')
    axs[1].set_ylabel('normalized flux')
    
    if mean_func is not None:
        axs[0].plot(t, mean_func(t, 0, *mean_params))
    return fig, axs