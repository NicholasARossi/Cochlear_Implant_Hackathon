import random
import operator
import itertools
import os
from deap import algorithms, base, creator, tools, gp

from software.gp_refactor.classes import FitnessWrapper, VectorClass, MatrixClass, VocoderRamp, NoiseClass
from software.gp_refactor.primitives import *







def all_primitives(wavefile_path, optimization='maximum', max_depth=4):


    fw = FitnessWrapper(wavefile_path)

    og_vect = VectorClass(fw.prepped_data, fw.prepped_rate)
    og_mat = MatrixClass(np.vstack([fw.prepped_data] * 16))
    og_ramp = VocoderRamp(os.path.abspath('./software/AB_imports/Vocoder/norm_ramp.npy'))

    white_noise = NoiseClass(np.random.normal(0, 1, 1000))

    pset = gp.PrimitiveSetTyped("MAIN", [VectorClass], MatrixClass)
    pset.addPrimitive(MatrixClass.create_matrix, list(itertools.repeat(VectorClass, 16)), MatrixClass)

    # vector vector primitives
    pset.addPrimitive(norm_vector_convolve_fft, [VectorClass, NoiseClass], VectorClass)
    pset.addPrimitive(norm_vector_convolve_fft, [VectorClass, VectorClass], VectorClass)

    pset.addPrimitive(convolve_ramp, [MatrixClass, VocoderRamp], MatrixClass)
    pset.addPrimitive(convolve_ramp_reverse, [MatrixClass, VocoderRamp], MatrixClass)
    pset.addPrimitive(invert_vector, [VectorClass], VectorClass)
    pset.addPrimitive(return_band_noise, [int], NoiseClass)
    pset.addPrimitive(norm_vector_convolve, [VectorClass, VectorClass], VectorClass)

    # pset.addPrimitive(vector_add,[VectorClass,VectorClass],VectorClass)
    # pset.addPrimitive(vector_subtract,[VectorClass,VectorClass],VectorClass)

    # vector value primitives
    pset.addPrimitive(vector_multiply, [VectorClass, float], VectorClass)

    pset.addPrimitive(norm_hilbert, [VectorClass], VectorClass)

    pset.addPrimitive(flex_low_freq, [VectorClass, int], VectorClass)
    pset.addPrimitive(flex_high_freq, [VectorClass, int], VectorClass)
    pset.addPrimitive(vector_flex_amplify, [VectorClass, int], VectorClass)
    pset.addPrimitive(phase_shift, [VectorClass, int], VectorClass)
    # pset.addPrimitive(convolve_ramp,[MatrixClass,VocoderRamp],MatrixClass)

    # pass values
    pset.addPrimitive(pass_primitive, [VocoderRamp], VocoderRamp)
    pset.addPrimitive(pass_primitive, [VectorClass], VectorClass)
    pset.addPrimitive(pass_primitive, [NoiseClass], NoiseClass)

    pset.addPrimitive(pass_primitive, [int], int)
    pset.addPrimitive(pass_primitive, [float], float)

    # value value primitives
    pset.addPrimitive(operator.neg, [float], float)

    ### ephermerals and terminals
    pset.addTerminal(white_noise, NoiseClass, name='white_noise')
    pset.addTerminal(og_mat, MatrixClass, name='default_mat')
    pset.addTerminal(og_ramp, VocoderRamp, name='ramp')

    for z in np.arange(10, 2010, 100):
        pset.addTerminal(int(z), int)

    pset.addEphemeralConstant("uniform", lambda: random.uniform(0.5, 5), float)

    pset.renameArguments(ARG0="input_audio")
    if optimization == 'maximum':
        print('we are currently maximizing')
        creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
    else:
        print('we are currently minimizing')
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("expr_init", gp.genHalfAndHalf, pset=pset, min_=1, max_=max_depth)

    # Structure initializers
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register("select", tools.selNSGA2)

    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=3, max_=max_depth)
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



def add_primitives(pset,primitives_list):
    '''
    This is a helper function mean to add suites of primitives for a given toolbox

    example primitive sets:

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
        pset.addEphemeralConstant("uniform", lambda: random.uniform(0.5, 5), float)
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

    ### add integer range used by all classes
    for z in np.arange(10, 2010, 100):
        pset.addTerminal(int(z), int)

    ### add pass primitives to prevent code from breaking

    pset.addPrimitive(pass_primitive, [VocoderRamp], VocoderRamp)
    pset.addPrimitive(pass_primitive, [VectorClass], VectorClass)
    pset.addPrimitive(pass_primitive, [NoiseClass], NoiseClass)
    pset.addPrimitive(pass_primitive, [int], int)
    pset.addPrimitive(pass_primitive, [float], float)

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

    toolbox.register("select", tools.selNSGA2)

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
