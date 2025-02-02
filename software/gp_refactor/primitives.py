import numpy as np

from scipy.signal import convolve, medfilt, hilbert, butter, filtfilt, fftconvolve,resample
from software.gp_refactor.classes import VectorClass, MatrixClass, NoiseClass
from sklearn.preprocessing import minmax_scale,QuantileTransformer,PowerTransformer
from scipy.stats import norm


def generate_sin_wav(vc, fc):
    w = (vc.frequency / 2) / fc  # Normalize the frequency
    sin_wav = np.sin(np.arange(len(vc.data)) / w)

    return VectorClass(sin_wav, vc.frequency)


def round_up_to_odd(f):
    return int(np.ceil(f) // 2 * 2 + 1)


def vector_medfilter(vc, integer):
    # requires odd integers
    oddint = round_up_to_odd(integer)

    vc.data = medfilt(vc.data, oddint)
    return vc


def flex_low_freq(vc, into):
    fs = vc.frequency

    fc = into  # Cut-off frequency of the filter
    w = fc / (fs / 2)  # Normalize the frequency
    b, a = butter(5, w, 'low')
    vc.data = filtfilt(b, a, vc.data)
    return vc


def flex_high_freq(vc, into):
    fs = vc.frequency

    fc = into  # Cut-off frequency of the filter
    w = fc / (fs / 2)  # Normalize the frequency
    b, a = butter(5, w, 'high')
    vc.data = filtfilt(b, a, vc.data)
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
    vc.data=np.multiply(np.power(abs(vc.data), x), np.sign(vc.data))
    return vc



def vector_flex_amplify(vc, into):
    vc.data = np.multiply(vc.data, into)
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


def phase_shift(vc, shift):
    y = vc.data
    y_copy = np.zeros(len(y))
    y_copy[shift:] = y[:len(y)-shift]
    vc.data = y_copy
    return vc


def vector_super_amplify(vc):
    vc.data = np.multiply(vc.data, 100)
    return vc


def convolve_ramp(mc, rc):

    for row_idx in range(16):

        convolved = fftconvolve(mc.data[row_idx, :], rc.data[row_idx, :], mode='same')
        convolved /= np.max(convolved)
        convolved *= max(mc.data[row_idx, :])
        mc.data[row_idx, :] = convolved
    return MatrixClass(mc.data)


def convolve_ramp_reverse(mc, rc):

    for row_idx in range(16):

        convolved = fftconvolve(mc.data[row_idx, :], rc.data[15-row_idx, :], mode='same')
        convolved /= np.max(convolved)
        convolved *= max(mc.data[row_idx, :])
        mc.data[row_idx, :] = convolved
    return mc


def norm_vector_convolve_fft(vc1, vc2):
    convolved = fftconvolve(vc1.data, vc2.data, mode='same')
    convolved /= np.max(convolved)
    convolved *= max(vc2.data)
    vc1.data = convolved
    return vc1


def fftnoise(f):
    f = np.array(f, dtype='complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np+1] *= phases
    f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
    return np.fft.ifft(f).real


def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1):
    freqs = np.abs(np.fft.fftfreq(samples, 1/samplerate))
    f = np.zeros(samples)
    idx = np.where(np.logical_and(freqs >= min_freq, freqs <= max_freq))[0]
    f[idx] = 1
    return fftnoise(f)


def return_band_noise(freq):
    x = band_limited_noise(freq, freq + 10, 55556, 55556)
    return NoiseClass(x)


def invert_vector(vc):
    vc.data = vc.data*-1
    return vc

def min_max_scale(vc):
    scaled = minmax_scale(vc.data, [-500, 500])
    vc.data=scaled
    return vc

def normal_scaler(vc):
    quantile_transformer = QuantileTransformer(
        output_distribution='normal', random_state=88)

    data = quantile_transformer.fit_transform(vc.data.reshape(-1, 1))
    vc.data=data.ravel()
    return vc

def uniform_scaler(vc):
    quantile_transformer = QuantileTransformer(
        output_distribution='uniform', random_state=88)

    data = quantile_transformer.fit_transform(vc.data.reshape(-1, 1))
    vc.data=data.ravel()
    return vc

def yeo_power_scaler(vc):
    power_transform = PowerTransformer(
        method='yeo-johnson', standardize=True)

    data = power_transform.fit_transform(vc.data.reshape(-1, 1))
    vc.data = data.ravel()
    return vc

def robust_scale(vc):
    scaled = robust_scale(vc.data.reshape(-1, 1))
    vc.data=scaled.ravel()
    return vc


def max_norm(vc):
    y = norm.pdf(np.linspace(-1, 1, 100))

    output = np.dot(vc.data.reshape(-1, 1), y.reshape(-1, 1).T)

    output=np.max(output, 1)
    vc.data=output.ravel()
    return vc

def min_norm(vc):
    y = norm.pdf(np.linspace(-1, 1, 100))

    output = np.dot(vc.data.reshape(-1, 1), y.reshape(-1, 1).T)

    output=np.min(output, 1)
    vc.data=output.ravel()
    return vc

def vector_resample(vc,integer):

    y = resample(vc.data, integer)
    y = resample(y, int(len(vc.data)))
    vc.data=y
    return vc

def vector_clip(vc,integer):
    vc.data=np.clip(vc.data, -integer, integer)
    return vc

def vector_threshold(vc,float):
    if float>=1:
        temp_float=float-1
    else:
        temp_float=float
    vc.data[np.abs(vc.data)<(temp_float*np.absmax(vc.data))]=0
    return vc