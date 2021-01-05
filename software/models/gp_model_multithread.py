# Import necessary functions
import numpy as np
import os
import sys
sys.path.append(os.path.abspath('../AB_imports/'))
sys.path.append(os.path.abspath('../fitness_functions/'))
sys.path.append(os.path.abspath('../../software/'))
#from scoop import futures
import multiprocessing

from sklearn.preprocessing import StandardScaler
from scipy.io import wavfile
from fitness_functions.delta_wav import convert_sample_rate,compute_wavfile_delta,wavefile_correlation,wavefile_max_xcor

from Vocoder.vocoder import vocoderFunc
from scipy.signal import convolve,medfilt,hilbert
import itertools
import pickle
import datetime

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
    def __init__(self, data):

        self.data=data








if __name__ == '__main__':
    # gp packages
    import random
    from deap import algorithms, base, creator, tools, gp
    import operator


    fw = FitnessWrapper()

    og_vect=VectorClass(fw.prepped_data)

    og_mat=MatrixClass(np.vstack([fw.prepped_data]*16))

    def round_up_to_odd(f):
        return int(np.ceil(f) // 2 * 2 + 1)

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

    def vector_divide(vc,x):
        if x is not None:
            vc.data=np.divide(vc.data,x)
            return vc
        else:
            return vc

    def norm_vector_convolve(vc1,vc2):
        convolved=convolve(vc1.data, vc2.data, mode='same')
        convolved/=np.max(convolved)
        convolved*=max(vc2.data)
        vc1.data=convolved
        return vc1

    def norm_hilbert(vc):
        vc.data=np.abs((hilbert(vc.data)))
        return vc


    def vector_add(vc1,vc2):
        vc1.data = np.add(vc1.data, vc2.data)
        return vc1

    def vector_subtract(vc1,vc2):
        vc1.data = np.subtract(vc1.data, vc2.data)
        return vc1


    def pass_primitive(x):

        return x

    pset = gp.PrimitiveSetTyped("MAIN", [VectorClass],MatrixClass)
    pset.addPrimitive(MatrixClass.create_matrix,list(itertools.repeat(VectorClass, 16)),MatrixClass)


    ### vector vector primitives
    pset.addPrimitive(norm_vector_convolve,[VectorClass,VectorClass],VectorClass)
    # pset.addPrimitive(vector_add,[VectorClass,VectorClass],VectorClass)
    # pset.addPrimitive(vector_subtract,[VectorClass,VectorClass],VectorClass)

    ### vector value primitives
    pset.addPrimitive(vector_multiply, [VectorClass,float], VectorClass)
    pset.addPrimitive(norm_hilbert,[VectorClass],VectorClass)
    #pset.addPrimitive(vector_divide, [VectorClass,float], VectorClass)

    pset.addPrimitive(vector_medfilter,[VectorClass,int],VectorClass)
    pset.addPrimitive(pass_primitive,[int],int)
    pset.addPrimitive(pass_primitive,[float],float)

    ### value value primitives
    pset.addPrimitive(operator.neg, [float],float)
    pset.addPrimitive(operator.neg, [float],float)


    ### ephermerals and terminals
    pset.addTerminal(og_mat,MatrixClass)
    pset.addEphemeralConstant("rand_int", lambda: random.randrange(1, 101+1, 20), int)
    pset.addEphemeralConstant("uniform", lambda: random.uniform(0.5, 1.5), float)

    pset.renameArguments(ARG0="input_audio")


    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("expr_init", gp.genFull, pset=pset, min_=1, max_=3)
    #pool = multiprocessing.Pool()
    #toolbox.register("map", pool.map)

   # toolbox.register("map", futures.map)

    # Structure initializers
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)


    def evalSymbTransform(individual, fw):
        ## investigate with gp.graph(individual)

        transform = toolbox.compile(expr=individual)
        score = fw.score_new_transform(transform,traceback=individual)
        return score,


    toolbox.register("evaluate", evalSymbTransform, fw=fw)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=3)
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

    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(5)
    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 10, stats=mstats,
                                   halloffame=hof, verbose=True)


    ### saving best output
    for example in hof:

        try:
            transform = toolbox.compile(expr=example)
            score = fw.score_new_transform(transform)

            results_scores={
                     'audio_input':fw.original_data,
                     'frequency_input':fw.original_rate,
                     'audio_out':fw.audioOut,
                     'frequency_out':fw.audioFs,
                     'score':score

            }

            results_models={'max_result':example
            }


            output_directory=os.path.abspath('../../results')
            timestamp = '{:%Y_%m_%d_%H%M%S}'.format(datetime.datetime.now())
            out_file = os.path.join(output_directory, "results_scores_{}.pkl".format(timestamp))
            pickle.dump(results_scores, open(out_file, "wb"))
            out_file = os.path.join(output_directory, "results_models_{}.pkl".format(timestamp))
            pickle.dump(results_models, open(out_file, "wb"))

            break
        except:
            pass
