# Import necessary functions
import numpy as np
from pathlib import Path
import os
import sys
sys.path.append(os.path.abspath('../AB_imports/'))

# Import the rest of the GpyT subpackage functions for the demo here
from Frontend.readWav import readWavFunc
from Frontend.tdFilter import tdFilterFunc
from Agc.dualLoopTdAgc import dualLoopTdAgcFunc
from WinBuf.winBuf import winBufFunc
from Filterbank.fftFilterbank import fftFilterbankFunc
from Filterbank.hilbertEnvelope import hilbertEnvelopeFunc
from Filterbank.channelEnergy import channelEnergyFunc
from NoiseReduction.noiseReduction import noiseReductionFunc
from PostFilterbank.specPeakLocator import specPeakLocatorFunc
from PostFilterbank.currentSteeringWeights import currentSteeringWeightsFunc
from PostFilterbank.carrierSynthesis import carrierSynthesisFunc
from Mapping.f120Mapping import f120MappingFunc
from Electrodogram.f120Electrodogram import f120ElectrodogramFunc
from Validation.validateOutput import validateOutputFunc
from Vocoder.vocoder import vocoderFunc


def demo4_procedural(wavefile=None):
    stratWindow = 0.5 * (np.blackman(256) + np.hanning(256))
    stratWindow = stratWindow.reshape(1, stratWindow.size)
    if wavefile == None:
        basepath = Path(__file__).parent.parent.absolute()
        wavefile = basepath / 'Sounds/AzBio_3sent_65dBSPL.wav'

    parStrat = {
        'wavFile': wavefile,  # this should be a complete absolute path to your sound file of choice
        'fs': 17400,  # this value matches implant internal audio rate. incoming wav files resampled to match
        'nFft': 256,
        'nHop': 20,
        'nChan': 15,  # do not change
        'startBin': 6,
        'nBinLims': np.array([2, 2, 1, 2, 2, 2, 3, 4, 4, 5, 6, 7, 8, 10, 56]),
        'window': stratWindow,
        'pulseWidth': 18,  # DO NOT CHANGE
        'verbose': 0
    }

    parReadWav = {
        'parent': parStrat,
        'tStartEnd': [],
        'iChannel': 1,
    }



    parElectrodogram = {
        'parent': parStrat,
        'cathodicFirst': True,
        'channelOrder': np.array([1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12]),
    # DO NOT CHANGE (different order of pulses will have no effect in vocoder output)
        'enablePlot': True,
        'outputFs': 55556,
    # DO NOT CHANGE (validation depends on matched output rate, vocoder would not produce different results at higher or lower Fs when parameters match accordingly)
    }

    parValidate = {
        'parent': parStrat,
        'lengthTolerance': 50,
        'saveIfSimilar': True,  # save even if the are too similar to default strategy
        'differenceThreshold': 1,
        'maxSimilarChannels': 8,
        'elGramFs': parElectrodogram['outputFs'],
    # this is linked to the previous electrodogram generation step, it should always match [55556 Hz]
        'outFile': None
    # This should be the full path including filename to a location where electrode matrix output will be saved after validation
    }

    results = {}  # initialize demo results structure

    # read specified wav file and scale
    results['sig_smp_wavIn'], results['sourceName'] = readWavFunc(
        parReadWav)  # load the file specified in parReadWav; assume correct scaling in wav file (111.6 dB SPL peak full-scale)



    # convert amplitude words to simulated electrodogram for vocoder imput
    results['elGram'] = f120ElectrodogramFunc(parElectrodogram, results['sig_ft_ampWords'])

    # # load output of default processing strategy to compare with  results['elGram'], return errors if data matrix is an invalid shape/unacceptable to the vocoder,save results['elGram'] to a file
    results['outputSaved'] = validateOutputFunc(parValidate, results['elGram'], results['sourceName']);

    # process electrodogram potentially saving as a file (change to saveOutput=True)
    results['audioOut'], results['audioFs'] = vocoderFunc(results['elGram'], saveOutput=False)


    return results
