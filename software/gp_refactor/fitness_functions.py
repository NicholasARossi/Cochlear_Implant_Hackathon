

import numpy as np
from scipy import interpolate
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from pesq import pesq

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


def fft_correlation(reference,reference_rate,output,output_rate):
    ref_remastered = convert_sample_rate(reference, reference_rate)
    output_remastered = convert_sample_rate(output, output_rate)
    a=ref_remastered
    b=output_remastered
    a = (a - np.mean(a)) / (np.std(a) * len(a))
    b = (b - np.mean(b)) / (np.std(b))
    a=abs(np.fft.rfft(a))
    b = abs(np.fft.rfft(b))

    min_len=min([len(a),len(b)])

    return pearsonr(a[:min_len],b[:min_len])[0]


def compute_wavfile_delta(reference,reference_rate,output,output_rate,newrate=16000,function='nb'):
    '''
    Simple function for comparing the relatedness of two distinct wave files
    :param reference: the original wave file values
    :param reference_rate: the original wave file sampling rate
    :param output: the changed wave file values
    :param output_rate: the changed wave file sampling rate
    :param newrate: the new rate to compare the two at
    :param function: the PESQ cost method
    :return:
    '''
    ref_remastered = convert_sample_rate(reference, reference_rate,newrate=newrate)
    output_remastered = convert_sample_rate(output, output_rate,newrate=newrate)


    pesq_max=pesq(newrate, ref_remastered, ref_remastered, function)
    pesq_obs=pesq(newrate, ref_remastered, output_remastered, function)

    output_val=(pesq_obs)/(pesq_max)

    return output_val