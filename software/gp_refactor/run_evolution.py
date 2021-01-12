import pickle
from deap import algorithms, tools,gp
import numpy as np
import random
import time
import datetime
from toolboxes import create_toolbox,filters_only

def main(verbose=True):
    total_time = time.time()

    toolbox,mstats,fw=filters_only()
    # Start a new evolution
    population = toolbox.population(n=300)
    start_gen = 0
    end_gen=30
    halloffame = tools.HallOfFame(maxsize=1)


    logbook = tools.Logbook()



    for gen in range(start_gen, end_gen):
        gen_time=time.time()
        population = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]


        # evaluate the fitnesses of the individuals
        fitnesses=[]
        for ind in invalid_ind:
            # some solutions produce non rational answers
            try:
                fitnesses.append(toolbox.evaluate(ind))
            except:
                fitnesses.append((0,))

        for ind, fit in zip(invalid_ind, fitnesses):

            ind.fitness.values = fit


        halloffame.update(population)
        record = mstats.compile(population)
        now=time.time()
        time_run=int(now-total_time)
        time_remaining=int((time_run/(gen-start_gen+1))*(end_gen-gen))
        record.update({'time':{'total':datetime.timedelta(seconds=time_run),
                               'remaining':datetime.timedelta(seconds=time_remaining)}})
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        if verbose==True:
            print(logbook.stream)
        population = toolbox.select(population, k=len(population))


        if gen % 5 == 0:

            for example in halloffame:
                transform = toolbox.compile(expr=example)
                score = fw.score_new_transform(transform)

                results_scores = {
                    'audio_input': fw.original_data,
                    'frequency_input': fw.original_rate,
                    'audio_out': fw.audioOut,
                    'frequency_out': fw.audioFs,
                    'score': score

                }

            cp = dict(population=population, generation=gen, halloffame=halloffame,result_scores=results_scores,
                      logbook=logbook, rndstate=random.getstate())

            with open(f"checkpoints/{gen}_checkpoint.pkl", "wb") as cp_file:
                pickle.dump(cp, cp_file)

if __name__ == '__main__':
    main()