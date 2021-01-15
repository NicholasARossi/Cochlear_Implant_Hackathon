

import numpy as np
from scipy import interpolate
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def convert_sample_rate(old_audio, old_samplerate, newrate=16000):
    duration = old_audio.shape[0] / old_samplerate

    time_old = np.linspace(0, duration, old_audio.shape[0])
    time_new = np.linspace(0, duration, int(old_audio.shape[0] * newrate / old_samplerate))

    interpolator = interpolate.interp1d(time_old, old_audio.T)
    new_audio = interpolator(time_new).T
    return new_audio


def wavefile_MSE(reference,reference_rate,output,output_rate):
    ref_remastered = convert_sample_rate(reference, reference_rate)
    output_remastered = convert_sample_rate(output, output_rate)
    min_len=min([len(ref_remastered),len(output_remastered)])
    return mean_squared_error(ref_remastered[:min_len],output_remastered[:min_len])


def wavefile_correlation(reference,reference_rate,output,output_rate):
    ref_remastered = convert_sample_rate(reference, reference_rate)
    output_remastered = convert_sample_rate(output, output_rate)
    min_len=min([len(ref_remastered),len(output_remastered)])
    return pearsonr(ref_remastered[:min_len],output_remastered[:min_len])[0]

def wavefile_max_xcor(reference,reference_rate,output,output_rate):
    ref_remastered = convert_sample_rate(reference, reference_rate)
    output_remastered = convert_sample_rate(output, output_rate)
    min_len=min([len(ref_remastered),len(output_remastered)])

    a=ref_remastered[:min_len]
    b=output_remastered[:min_len]
    a = (a - np.mean(a)) / (np.std(a) * len(a))
    b = (b - np.mean(b)) / (np.std(b))
    c = np.correlate(a, b, 'full')
    return max(c[min_len-1000:min_len+1000])

def fft_MSE(reference,reference_rate,output,output_rate):
    ref_remastered = convert_sample_rate(reference, reference_rate)
    output_remastered = convert_sample_rate(output, output_rate)
    ref_fft=np.fft.rfft(ref_remastered / (2 ** 15 - 1))[:5000]
    output_fft=np.fft.rfft(output_remastered)[:5000]
    return np.sum(abs(ref_fft-output_fft))