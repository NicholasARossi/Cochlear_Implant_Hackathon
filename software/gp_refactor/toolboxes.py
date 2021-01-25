import random
from deap import algorithms, base, creator, tools, gp
import operator
import numpy as np
import itertools
from classes import FitnessWrapper,VectorClass,MatrixClass,VocoderRamp,NoiseClass
from primitives import *
#from vocoder_primitives import convolve_ramp,convolve_ramp_reverse,norm_vector_convolve_fft





def all_primitives(wavefile_path,optimization='maximum',max_depth=4):

    fw = FitnessWrapper(wavefile_path)

    og_vect = VectorClass(fw.prepped_data, fw.prepped_rate)
    og_mat = MatrixClass(np.vstack([fw.prepped_data] * 16))
    og_ramp=VocoderRamp('/Users/nicholas.rossi/Documents/Personal/Cochlear_Implant_Hackathon/software/AB_imports/Vocoder/norm_ramp.npy')
    white_noise=NoiseClass(np.random.normal(0, 1, 1000))


    pset = gp.PrimitiveSetTyped("MAIN", [VectorClass], MatrixClass)
    pset.addPrimitive(MatrixClass.create_matrix, list(itertools.repeat(VectorClass, 16)), MatrixClass)

    ### vector vector primitives
    pset.addPrimitive(norm_vector_convolve_fft, [VectorClass, NoiseClass], VectorClass)
    pset.addPrimitive(norm_vector_convolve_fft, [VectorClass, VectorClass], VectorClass)

    pset.addPrimitive(convolve_ramp,[MatrixClass,VocoderRamp],MatrixClass)
    pset.addPrimitive(convolve_ramp_reverse,[MatrixClass,VocoderRamp],MatrixClass)
    pset.addPrimitive(invert_vector,[VectorClass],VectorClass)
    pset.addPrimitive(return_band_noise,[int],NoiseClass)
    # pset.addPrimitive(vector_add,[VectorClass,VectorClass],VectorClass)
    # pset.addPrimitive(vector_subtract,[VectorClass,VectorClass],VectorClass)

    ### vector value primitives
    pset.addPrimitive(vector_multiply, [VectorClass, float], VectorClass)

    pset.addPrimitive(norm_hilbert, [VectorClass], VectorClass)
    pset.addPrimitive(flex_low_freq, [VectorClass,int], VectorClass)
    pset.addPrimitive(flex_high_freq, [VectorClass,int], VectorClass)
    pset.addPrimitive(vector_flex_amplify, [VectorClass,int], VectorClass)
    pset.addPrimitive(phase_shift, [VectorClass,int], VectorClass)
    #pset.addPrimitive(convolve_ramp,[MatrixClass,VocoderRamp],MatrixClass)


    ### pass values
    pset.addPrimitive(pass_primitive, [VocoderRamp], VocoderRamp)
    pset.addPrimitive(pass_primitive, [VectorClass], VectorClass)
    pset.addPrimitive(pass_primitive, [NoiseClass], NoiseClass)

    pset.addPrimitive(pass_primitive, [int], int)
    pset.addPrimitive(pass_primitive, [float], float)

    ### value value primitives
    pset.addPrimitive(operator.neg, [float], float)

    ### ephermerals and terminals
    pset.addTerminal(white_noise, NoiseClass,name='white_noise')
    pset.addTerminal(og_mat, MatrixClass,name='default_mat')
    pset.addTerminal(og_ramp, VocoderRamp,name='ramp')

    for z in np.arange(10,2010,100):
        pset.addTerminal(int(z), int)

    pset.addEphemeralConstant("uniform", lambda: random.uniform(0.5, 5), float)

    pset.renameArguments(ARG0="input_audio")
    if optimization=='maximum':
        print('we are currently maximizing')
        creator.create("FitnessMax", base.Fitness, weights=(1.0,1.0,))
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



    return toolbox,mstats,fw




def debugger(wavefile_path,optimization='maximum',max_depth=4):

    fw = FitnessWrapper(wavefile_path)

    og_vect = VectorClass(fw.prepped_data, fw.prepped_rate)
    og_mat = MatrixClass(np.vstack([fw.prepped_data] * 16))
    og_ramp=VocoderRamp('/Users/nicholas.rossi/Documents/Personal/Cochlear_Implant_Hackathon/software/AB_imports/Vocoder/norm_ramp.npy')
    white_noise=NoiseClass(np.random.normal(0, 1, 1000))


    pset = gp.PrimitiveSetTyped("MAIN", [VectorClass], MatrixClass)
    pset.addPrimitive(MatrixClass.create_matrix, list(itertools.repeat(VectorClass, 16)), MatrixClass)

    ### vector vector primitives



    ### vector value primitives
    pset.addPrimitive(phase_shift, [VectorClass,int], VectorClass)
    #pset.addPrimitive(convolve_ramp,[MatrixClass,VocoderRamp],MatrixClass)


    ### pass values
    pset.addPrimitive(pass_primitive, [VocoderRamp], VocoderRamp)
    pset.addPrimitive(pass_primitive, [VectorClass], VectorClass)
    pset.addPrimitive(pass_primitive, [NoiseClass], NoiseClass)

    pset.addPrimitive(pass_primitive, [int], int)
    pset.addPrimitive(pass_primitive, [float], float)

    ### value value primitives
    pset.addPrimitive(operator.neg, [float], float)

    ### ephermerals and terminals
    pset.addTerminal(white_noise, NoiseClass,name='white_noise')
    pset.addTerminal(og_mat, MatrixClass,name='default_mat')
    pset.addTerminal(og_ramp, VocoderRamp,name='ramp')

    for z in np.arange(100,2000,100):
        pset.addTerminal(int(z), int)

    pset.addEphemeralConstant("uniform", lambda: random.uniform(0.5, 5), float)

    pset.renameArguments(ARG0="input_audio")
    if optimization=='maximum':
        print('we are currently maximizing')
        creator.create("FitnessMax", base.Fitness, weights=(1.0,1.0,))
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



    return toolbox,mstats,fw