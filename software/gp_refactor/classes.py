import numpy as np
import time
import datetime
from scipy.io import wavfile

from sklearn.preprocessing import StandardScaler

from software.gp_refactor.fitness_functions import convert_sample_rate, wavefile_max_xcor, fft_MSE, compute_wavfile_delta, fft_correlation
from software.AB_imports.Vocoder.vocoder import vocoderFunc


class FitnessWrapper:
    def __init__(self, wavefile_path):
        '''
        :param wavefile_path: Path to audio file
        '''
        self.wavefile_path = wavefile_path
        self._prep_file()

    def _prep_file(self):

        self.sourceName = self.wavefile_path
        # READ IN ORIGINAL
        samplerate, data = wavfile.read(self.wavefile_path)
        self.original_data = data
        self.original_rate = samplerate

        # resample for internal
        implant_rate = 17400
        implant_results = convert_sample_rate(data, samplerate, newrate=implant_rate)
        self.implant_data = implant_results
        self.implant_rate = implant_rate

        # convert for elgram
        output_rate = 55556
        output_results = convert_sample_rate(data, samplerate, newrate=output_rate)
        self.prepped_data = output_results
        self.prepped_rate = output_rate

    def score_new_transform(self, transform):
        '''
        Run and score transform
        '''
        self._run_transform(transform)
        self._convert_elgram()
        # print(f'{np.std(self.elGram)},{np.max(self.elGram)},{np.min(self.elGram)},{np.median(self.elGram)}')
        score = self._score_elgram()

        return score

    def _run_transform(self, transform):
        '''
        Vectorize the waveform
        '''
        vc = VectorClass(self.prepped_data, self.prepped_rate)
        self.transformed_data = transform(vc).data

    def _convert_elgram(self, rounding=False):
        '''
        What the hell is going on here? El grams must sum to 0, and must be sparse to make computation easier.
        '''

        # normalize rows (they need to sum to 0)
        for row in range(self.transformed_data.shape[0]):
            values = np.round(self.transformed_data[row, :].reshape(-1, 1)).astype('int')

            # Run standard scalar
            new_values = self._get_standard_scalar_transform(values)
            # Normalize and scale
            if max(new_values) != 0:
                new_values = self._normalize_and_scale(new_values)

            if rounding == True:
                self.transformed_data[row, :] = self._round(new_values)
            else:
                self.transformed_data[row, :] = new_values

        self.elGram = self.transformed_data

    @staticmethod
    def _get_standard_scalar_transform(values):
        '''
        Run standard scalar
        '''
        # using a standard scaler to get the values close to 0
        scaler = StandardScaler(with_std=False)
        scaler = scaler.fit(values)
        new_values = scaler.transform(values).ravel()

        return new_values

    @staticmethod
    def _normalize_and_scale(values, scaling_value=500):
        # Pick max based on absolute
        maxval = abs(max(values, key=abs))
        # Scale to 500, as recommended
        values = (values / maxval) * scaling_value

        return values

    @staticmethod
    def _round(values):
        '''
        Scale the array such that the overall sum = 0
        '''
        r = np.random.RandomState(8888)

        # Rounding the values to the nearest 10s
        rounded_vect = np.around(values, -1)
        deficit = np.sum(rounded_vect)
        def_sign = np.sign(deficit)  # -1 if deficit is negative else positive

        # Randomly select indices to increase/decrease by 1 based on deficit negative or positive resp.
        choices = np.argwhere(rounded_vect != 0)
        corrections = r.choice(choices.ravel(), size=int(abs(deficit)), replace=True)
        rounded_vect[corrections] += (-1 * def_sign)

        if sum(rounded_vect) != 0:
            print('Warning: rounding failed')

        return rounded_vect

    def _score_elgram(self):

        self.audioOut, self.audioFs = vocoderFunc(self.elGram, saveOutput=False)
        # print(f'{np.std(audioOut)},{np.max(audioOut)},{np.min(audioOut)},{np.median(audioOut)}')

        # TODO figure out the best possible fitness function
        score = wavefile_max_xcor(self.original_data, self.original_rate, self.audioOut, self.audioFs)
        score2 = fft_correlation(self.original_data, self.original_rate, self.audioOut, self.audioFs)

        return score, score2*0.1


class MatrixClass:
    def __init__(self, data):
        self.data = data

    @staticmethod
    def create_matrix(*vector_list):
        matrix = np.vstack([vect.data for vect in vector_list])

        return MatrixClass(matrix)


class VocoderRamp:
    def __init__(self, data_loc):
        self.data = np.load(data_loc).T


class NoiseClass:
    def __init__(self, data):
        self.data = data


class VectorClass:
    def __init__(self, data, frequency):
        self.data = data
        self.frequency = frequency
