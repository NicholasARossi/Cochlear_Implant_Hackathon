# Import necessary functions
import numpy as np
import os
import sys
sys.path.append(os.path.abspath('../AB_imports/'))
sys.path.append(os.path.abspath('../fitness_functions/'))
sys.path.append(os.path.abspath('../../software/'))

from sklearn.preprocessing import StandardScaler
from scipy.io import wavfile
from software.fitness_functions.delta_wav import convert_sample_rate,compute_wavfile_delta
from software.AB_imports.Vocoder.vocoder import vocoderFunc
from scipy.signal import convolve,medfilt

import string

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
        vc=VectorClass(self.prepped_data)

        self.transformed_data=transform(vc).data


    def convert_elgram(self):



        values = self.transformed_data.reshape(-1, 1)
        scaler = StandardScaler(with_std=False)
        scaler = scaler.fit(values)
        normalized = scaler.transform(values)


        # convert to elgram type
        self.elGram = np.vstack([normalized.T] * 16)

    def score_elgram(self):
        self.audioOut, self.audioFs = vocoderFunc(self.elGram, saveOutput=False)

        if np.isnan(self.audioOut).any()==False:
            self.score=compute_wavfile_delta(self.original_data ,self.original_rate,self.audioOut,self.audioFs)
        else:
            self.score=0
        return self.score

    def score_new_transform(self,transform):
        self.run_transform(transform)
        self.convert_elgram()
        return self.score_elgram()


class ShellModel:
    def __init__(self, wavefile_path=os.path.abspath('../sample_data/sentence1_55_clean.wav')):

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

    # defining primitives
    def convolution_prim(self,out1,out2):
        pass


    ### compiling
    def compile_transform(self):
        pass



    def run_transform(self,transform):
        self.transformed_data=transform(self.prepped_data)

    def convert_elgram(self):


        original_std = np.std(self.transformed_data)

        values = self.transformed_data.reshape(-1, 1)
        scaler = StandardScaler()
        scaler = scaler.fit(values)
        normalized = scaler.transform(values) * original_std

        # convert to elgram type
        self.elGram = np.vstack([normalized.T] * 16)

    def score_elgram(self):
        self.audioOut, self.audioFs = vocoderFunc(self.elGram, saveOutput=False)

        self.score=compute_wavfile_delta(self.original_data ,self.original_rate,self.audioOut,self.audioFs)
        return self.score

    def score_new_transform(self,transform):
        self.run_transform(transform)
        self.convert_elgram()
        return self.score_elgram()


class VectorClass:
    def __init__(self, data):

        self.data=data



class oddint:
    def __init__(self,range):
        self.data=random.randrange(1, range+1, 20)


if __name__ == '__main__':
    # gp packages
    import random
    from deap import algorithms, base, creator, tools, gp
    import operator


    fw = FitnessWrapper()

    og_vect=VectorClass(fw.prepped_data)


    def round_up_to_odd(f):
        return np.ceil(f) // 2 * 2 + 1

    def vector_medfilter(vc, integer):
        ## requires odd integers
        oddint=round_up_to_odd(integer)

        vc.data = medfilt(vc.data, oddint)
        return vc


    def vector_multiply(vc,x):
        if x is not None:
            vc.data=np.multiply(vc.data,x)
            return vc
        else:
            return vc

    def vector_convolve(vc1,vc2):
        vc1.data=convolve(vc1.data, vc2.data, mode='same')
        return vc1


    def pass_primitive(x):

        return x

    pset = gp.PrimitiveSetTyped("MAIN", [VectorClass],VectorClass)

    pset.addPrimitive(vector_multiply, [VectorClass,float], VectorClass)
    pset.addPrimitive(vector_convolve,[VectorClass,VectorClass],VectorClass)
    pset.addPrimitive(vector_medfilter,[VectorClass,oddint],VectorClass)

    pset.addPrimitive(operator.neg, [float],float)

    #pset.addEphemeralConstant("odd_int", lambda: random.randrange(1, 101+1, 20), int)
    pset.addEphemeralConstant("rand_int", lambda: random.randrange(1, 101+1, 20), oddint)
    #pset.addTerminal(oddint(101), oddint)
    pset.addEphemeralConstant("uniform", lambda: random.uniform(0.001, 2), float)
    pset.renameArguments(ARG0="input_audio")
    pset.addPrimitive(pass_primitive,[oddint],oddint)


    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("expr_init", gp.genFull, pset=pset, min_=1, max_=2)

    # Structure initializers
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)


    def evalSymbTransform(individual, fw):
        transform = toolbox.compile(expr=individual)
        score = fw.score_new_transform(transform)
        return score,


    toolbox.register("evaluate", evalSymbTransform, fw=fw)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, stats=mstats,
                                   halloffame=hof, verbose=True)