import numpy as np
from scipy.signal import convolve,medfilt,hilbert,butter,filtfilt,deconvolve,fftconvolve

def convolve_ramp(mc,rc):

    for row_idx in range(16):

        convolved = fftconvolve(mc.data[row_idx,:], rc.data[row_idx,:], mode='same')
        convolved /= np.max(convolved)
        convolved *= max(mc.data[row_idx,:])
        mc.data[row_idx,:] = convolved
    return mc


def convolve_ramp_reverse(mc,rc):

    for row_idx in range(16):

        convolved = fftconvolve(mc.data[row_idx,:], rc.data[15-row_idx,:], mode='same')
        convolved /= np.max(convolved)
        convolved *= max(mc.data[row_idx,:])
        mc.data[row_idx,:] = convolved
    return mc

def norm_vector_convolve_fft(vc1, vc2):
    convolved = fftconvolve(vc1.data, vc2.data, mode='same')
    convolved /= np.max(convolved)
    convolved *= max(vc2.data)
    vc1.data = convolved
    return vc1