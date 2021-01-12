import random
from deap import algorithms, base, creator, tools, gp
import operator
import numpy as np
import itertools
from classes import FitnessWrapper,VectorClass,MatrixClass
from primitives import *





def create_toolbox():
    fw = FitnessWrapper()

    og_vect = VectorClass(fw.prepped_data, fw.prepped_rate)

    og_mat = MatrixClass(np.vstack([fw.prepped_data] * 16))

    pset = gp.PrimitiveSetTyped("MAIN", [VectorClass], MatrixClass)
    pset.addPrimitive(MatrixClass.create_matrix, list(itertools.repeat(VectorClass, 16)), MatrixClass)

    ### vector vector primitives
    pset.addPrimitive(norm_vector_convolve, [VectorClass, VectorClass], VectorClass)
    # pset.addPrimitive(vector_add,[VectorClass,VectorClass],VectorClass)
    # pset.addPrimitive(vector_subtract,[VectorClass,VectorClass],VectorClass)

    ### vector value primitives
    pset.addPrimitive(vector_multiply, [VectorClass, float], VectorClass)
    # pset.addPrimitive(vector_power, [VectorClass, float], VectorClass)

    pset.addPrimitive(norm_hilbert, [VectorClass], VectorClass)
    pset.addPrimitive(vector_low_freq_filter, [VectorClass], VectorClass)
    pset.addPrimitive(vector_super_low_freq_filter, [VectorClass], VectorClass)
    pset.addPrimitive(vector_ultra_low_freq_filter, [VectorClass], VectorClass)

    pset.addPrimitive(vector_high_freq_filter, [VectorClass], VectorClass)
    pset.addPrimitive(vector_super_high_freq_filter, [VectorClass], VectorClass)
    pset.addPrimitive(vector_ultra_high_freq_filter, [VectorClass], VectorClass)

    pset.addPrimitive(vector_amplify, [VectorClass], VectorClass)
    pset.addPrimitive(vector_super_amplify, [VectorClass], VectorClass)
    pset.addPrimitive(vector_ultra_amplify, [VectorClass], VectorClass)

    # pset.addPrimitive(vector_divide, [VectorClass,float], VectorClass)

    # pset.addPrimitive(vector_medfilter,[VectorClass,int],VectorClass)
    pset.addPrimitive(pass_primitive, [int], int)
    pset.addPrimitive(pass_primitive, [float], float)

    ### value value primitives
    pset.addPrimitive(operator.neg, [float], float)

    ### ephermerals and terminals
    pset.addTerminal(og_mat, MatrixClass)
    pset.addEphemeralConstant("rand_int", lambda: random.randrange(1, 101 + 1, 20), int)
    pset.addEphemeralConstant("uniform", lambda: random.uniform(0.5, 5), float)

    pset.renameArguments(ARG0="input_audio")

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("expr_init", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)


    # Structure initializers
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    def evaluate_symbolic_transform(individual, fw):
        ## investigate with gp.graph(individual)

        transform = toolbox.compile(expr=individual)
        score = fw.score_new_transform(transform, traceback=individual)
        return score,


    toolbox.register("evaluate", evaluate_symbolic_transform, fw=fw)
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



    return toolbox,mstats,fw