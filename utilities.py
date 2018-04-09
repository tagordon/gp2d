import numpy as np

def psd(y, t):
    f = np.fft.rfftfreq(len(t), t[1] - t[0])
    fft = np.fft.rfft(y)
    fft *= np.conj(fft)
    return f, fft.real / len(y)**2