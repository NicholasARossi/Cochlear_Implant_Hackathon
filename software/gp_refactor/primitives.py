import numpy as np
from scipy.signal import convolve,medfilt,hilbert,butter,filtfilt


def round_up_to_odd(f):
    return int(np.ceil(f) // 2 * 2 + 1)


def vector_medfilter(vc, integer):
    ## requires odd integers
    oddint = round_up_to_odd(integer)

    vc.data = medfilt(vc.data, oddint)
    return vc


def vector_low_freq_filter(vc):
    fs = vc.frequency

    fc = 1000  # Cut-off frequency of the filter
    w = fc / (fs / 2)  # Normalize the frequency
    b, a = butter(5, w, 'low')
    vc.data = filtfilt(b, a, vc.data)
    return vc


def vector_super_low_freq_filter(vc):
    fs = vc.frequency

    fc = 500  # Cut-off frequency of the filter
    w = fc / (fs / 2)  # Normalize the frequency
    b, a = butter(5, w, 'low')
    vc.data = filtfilt(b, a, vc.data)
    return vc


def vector_ultra_low_freq_filter(vc):
    fs = vc.frequency

    fc = 200  # Cut-off frequency of the filter
    w = fc / (fs / 2)  # Normalize the frequency
    b, a = butter(5, w, 'low')
    vc.data = filtfilt(b, a, vc.data)
    return vc


def vector_high_freq_filter(vc):
    fs = vc.frequency

    fc = 1000  # Cut-off frequency of the filter
    w = fc / (fs / 2)  # Normalize the frequency
    b, a = butter(5, w, 'high')
    vc.data = filtfilt(b, a, vc.data)
    return vc


def vector_super_high_freq_filter(vc):
    fs = vc.frequency

    fc = 500  # Cut-off frequency of the filter
    w = fc / (fs / 2)  # Normalize the frequency
    b, a = butter(5, w, 'high')
    vc.data = filtfilt(b, a, vc.data)
    return vc


def vector_ultra_high_freq_filter(vc):
    fs = vc.frequency

    fc = 200  # Cut-off frequency of the filter
    w = fc / (fs / 2)  # Normalize the frequency
    b, a = butter(5, w, 'high')
    vc.data = filtfilt(b, a, vc.data)
    return vc


def vector_multiply(vc, x):
    if x is not None:
        vc.data = np.multiply(vc.data, x)
        return vc
    else:
        return vc


def vector_power(vc, x):
    if x is not None:
        vc.data = np.power(vc.data, x)
        return vc
    else:
        return vc


def vector_amplify(vc):
    vc.data = np.multiply(vc.data, 10)
    return vc


def vector_super_amplify(vc):
    vc.data = np.multiply(vc.data, 100)
    return vc


def vector_ultra_amplify(vc):
    vc.data = np.multiply(vc.data, 1000)
    return vc


def vector_divide(vc, x):
    if x is not None:
        vc.data = np.divide(vc.data, x)
        return vc
    else:
        return vc


def norm_vector_convolve(vc1, vc2):
    convolved = convolve(vc1.data, vc2.data, mode='same')
    convolved /= np.max(convolved)
    convolved *= max(vc2.data)
    vc1.data = convolved
    return vc1


def norm_hilbert(vc):
    vc.data = np.abs((hilbert(vc.data)))
    return vc


def vector_add(vc1, vc2):
    vc1.data = np.add(vc1.data, vc2.data)
    return vc1


def vector_subtract(vc1, vc2):
    vc1.data = np.subtract(vc1.data, vc2.data)
    return vc1


def pass_primitive(x):
    return x