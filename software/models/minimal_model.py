# Import necessary functions
import numpy as np
import os
import sys
sys.path.append(os.path.abspath('../AB_imports/'))

from sklearn.preprocessing import StandardScaler
from scipy.io import wavfile
from ..fitness_functions.delta_wav import convert_sample_rate
from Vocoder.vocoder import vocoderFunc


def basic_model(wavefile_path=os.path.abspath('../sample_data/sentence1_55_clean.wav')):

    results={}
    results['sourceName']=wavefile_path
    # READ IN ORIGINAL
    samplerate, data = wavfile.read(wavefile_path)
    results['original_data']=data
    results['original_rate']=samplerate

    # resample for internal
    implant_rate=17400
    implant_results = convert_sample_rate(data, samplerate, newrate=implant_rate)
    results['implant_data'] = implant_results
    results['implant_rate'] = implant_rate

    # convert for elgram
    output_rate=55556
    output_results = convert_sample_rate(data, samplerate, newrate=output_rate)


    results['output_data'] = output_results
    results['output_rate'] = output_rate

    #TODO this is where we evolve mathematical operations on the

    #normalizing the data so it sums to 0
    original_std=np.std(results['output_data'])

    values = results['output_data'].reshape(-1, 1)
    scaler = StandardScaler()
    scaler = scaler.fit(values)
    normalized = scaler.transform(values)*original_std

    # convert to elgram type
    results['elGram']=np.vstack([normalized.T] * 16)

    #TODO this iw where the changes end



    results['audioOut'], results['audioFs'] = vocoderFunc(results['elGram'], saveOutput=False)


    return results

if __name__ == '__main__':
    results=basic_model()
    print(results)