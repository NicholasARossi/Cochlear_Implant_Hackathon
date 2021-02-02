import random
import operator
import itertools
import os
from deap import algorithms, base, creator, tools, gp

from software.gp_refactor.classes import FitnessWrapper, VectorClass, MatrixClass, VocoderRamp, NoiseClass
from software.gp_refactor.primitives import *




def add_primitives(pset,primitives_list):
    '''
    This is a helper function mean to add suites of primitives for a given toolbox


    example primitive sets:
=======
    og_vect = VectorClass(fw.prepped_data, fw.prepped_rate)
    og_mat = MatrixClass(np.vstack([fw.prepped_data] * 16))
    og_ramp = VocoderRamp(os.path.abspath('./software/AB_imports/Vocoder/norm_ramp.npy'))
    white_noise = NoiseClass(np.random.normal(0, 1, 1000))

    :param pset:
    :param primitives_list:
    :return:
    '''
    if 'convolution' in primitives_list:
        pset.addPrimitive(norm_vector_convolve_fft, [VectorClass, VectorClass], VectorClass)
        pset.addPrimitive(norm_hilbert, [VectorClass], VectorClass)

    if 'noise' in primitives_list and 'convolution' in primitives_list:
        pset.addPrimitive(norm_vector_convolve_fft, [VectorClass, NoiseClass], VectorClass)

    if 'addition' in primitives_list:
        pset.addPrimitive(vector_add, [VectorClass, VectorClass], VectorClass)
        pset.addPrimitive(vector_subtract,[VectorClass,VectorClass],VectorClass)


    if 'multiplication' in primitives_list:
        pset.addPrimitive(invert_vector, [VectorClass], VectorClass)
        pset.addPrimitive(vector_multiply, [VectorClass, float], VectorClass)
        pset.addPrimitive(vector_flex_amplify, [VectorClass, int], VectorClass)
        pset.addPrimitive(operator.neg, [float], float)

    if 'filter' in primitives_list:
        pset.addPrimitive(flex_low_freq, [VectorClass, int], VectorClass)
        pset.addPrimitive(flex_high_freq, [VectorClass, int], VectorClass)

    if 'phase' in primitives_list:
        pset.addPrimitive(phase_shift, [VectorClass, int], VectorClass)

    if 'noise' in primitives_list:
        white_noise = NoiseClass(np.random.normal(0, 1, 1000))
        pset.addTerminal(white_noise, NoiseClass, name='white_noise')
        pset.addPrimitive(return_band_noise, [int], NoiseClass)


    if 'scale' in primitives_list:
        pset.addPrimitive(min_max_scale, [VectorClass], VectorClass)
        pset.addPrimitive(normal_scaler, [VectorClass], VectorClass)
        pset.addPrimitive(uniform_scaler, [VectorClass], VectorClass)
        pset.addPrimitive(yeo_power_scaler, [VectorClass], VectorClass)
        pset.addPrimitive(robust_scale, [VectorClass], VectorClass)
        # pset.addPrimitive(max_norm, [VectorClass], VectorClass)
        # pset.addPrimitive(min_norm, [VectorClass], VectorClass)
        pset.addPrimitive(vector_power, [VectorClass,float], VectorClass)
        pset.addPrimitive(vector_resample,[VectorClass,int], VectorClass)

    if 'clip' in primitives_list:
        pset.addPrimitive(vector_clip,[VectorClass,int], VectorClass)
        pset.addPrimitive(vector_threshold,[VectorClass,float], VectorClass)


    ### add integer range used by all classes
    for z in np.arange(10, 2010, 100):
        pset.addTerminal(int(z), int)

    for z in np.linspace(0.5,1.5,5):
        pset.addTerminal(float(z), float)

    ### add pass primitives to prevent code from breaking

    pset.addPrimitive(pass_primitive, [VocoderRamp], VocoderRamp)
    pset.addPrimitive(pass_primitive, [VectorClass], VectorClass)
    pset.addPrimitive(pass_primitive, [NoiseClass], NoiseClass)
    pset.addPrimitive(pass_primitive, [int], int)
    pset.addPrimitive(pass_primitive, [float], float)
    pset.addEphemeralConstant("uniform", lambda: random.uniform(0.5, 1.5), float)

    return pset



def custom_toolbox(wavefile_path,
                   optimization='maximum',
                   max_depth=4,
                   primitives_list=['convolution','noise',
                                    'addition','multiplication',
                                    'filter','phase','noise']):



    # prepping bare bone toolbox (no additional computation)
    fw = FitnessWrapper(wavefile_path)
    og_mat = MatrixClass(np.vstack([fw.prepped_data] * 16))
    pset = gp.PrimitiveSetTyped("MAIN", [VectorClass], MatrixClass)
    pset.addPrimitive(MatrixClass.create_matrix, list(itertools.repeat(VectorClass, 16)), MatrixClass)
    pset.addTerminal(og_mat, MatrixClass, name='default_mat')
    pset.renameArguments(ARG0="input_audio")

    # add additional primitives
    pset=add_primitives(pset,primitives_list)


    # we're only maximizing here
    creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


    toolbox = base.Toolbox()
    toolbox.register("expr_init", gp.genHalfAndHalf, pset=pset, min_=1, max_=max_depth)

    # Structure initializers
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    #toolbox.register("select", tools.selNSGA2)
    toolbox.register("select", tools.selTournament,tournsize=3)
    #toolbox.register("mate", gp.cxOnePointLeafBiased,termpb=0.1)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=1, max_=max_depth)
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

    return toolbox, mstats, fw
