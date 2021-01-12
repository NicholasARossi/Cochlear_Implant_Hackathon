import numpy as np
import os
from scipy.io import wavfile
import sys
sys.path.append(os.path.abspath('../AB_imports/'))

from Vocoder.vocoder import vocoderFunc
from fitness_functions import convert_sample_rate,wavefile_max_xcor
from sklearn.preprocessing import StandardScaler


class FitnessWrapper:
    def __init__(self, wavefile_path=os.path.abspath('../../sample_data/sentence1_55_clean.wav')):

        self.wavefile_path=wavefile_path

        self.prep_file()
    def prep_file(self):

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



    def run_transform(self,transform):
        vc=VectorClass(self.prepped_data,self.prepped_rate)

        self.transformed_data=transform(vc).data


    def convert_elgram(self,traceback=None):




        ### normalize rows (they need to sum to 0)
        for row in range(self.transformed_data.shape[0]):
            values=self.transformed_data[row,:].reshape(-1,1)
            scaler = StandardScaler(with_std=False)
            #scaler = StandardScaler()
            scaler = scaler.fit(values)
            new_values=scaler.transform(values).ravel()
            if max(new_values)<1 and max(new_values)!=0:
                #print('new normals')
                new_values/=max(new_values)
                new_values*=400
            self.transformed_data[row,:]=new_values

        # if np.sum(np.sum(self.transformed_data, 1)) > 1:
        #     print('debug')


        # convert to elgram type
        self.elGram =  self.transformed_data

    def score_elgram(self):

        if np.sum(np.sum(self.elGram, 1))>1 or np.max(self.elGram)==0:
            print('normalization failed')
            self.score = 0
            return self.score
        else:

            self.audioOut, self.audioFs = vocoderFunc(self.elGram, saveOutput=False)

            if np.isnan(self.audioOut).any()==False:
                self.score=wavefile_max_xcor(self.original_data,self.original_rate,self.audioOut,self.audioFs)
            else:
                self.score=0
            return self.score

    def score_new_transform(self,transform,traceback=None):
        self.run_transform(transform)
        self.convert_elgram(traceback=traceback)
        return self.score_elgram()



class MatrixClass:
    def __init__(self, data):
        self.data=data

    @staticmethod
    def create_matrix(*vector_list):
        matrix=np.vstack([vect.data for vect in vector_list])

        return MatrixClass(matrix)

class VectorClass:
    def __init__(self, data,frequency):

        self.data=data
        self.frequency=frequency