import pickle
from deap import algorithms, tools,gp
import random
import time
import datetime
from toolboxes import all_primitives
import pandas as pd

def main(wavefile_path, pop_size=50,end_gen=30,verbose=True,optimization='maximum'):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')


    total_time = time.time()

    toolbox,mstats,fw=all_primitives(wavefile_path,optimization=optimization)
    # Start a new evolution
    population = toolbox.population(n=pop_size)
    start_gen = 0
    # halloffame = tools.HallOfFame(maxsize=1)
    if optimization=='maximum':
        bad_val=0
    else:
        bad_val=10000000

    logbook = tools.Logbook()
    best_individual={'score':bad_val,'individual':''}


    for gen in range(start_gen, end_gen):
        population = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]


        # evaluate the fitnesses of the individuals
        fitnesses=[]


        for ind in invalid_ind:
            # some solutions produce non rational answers
            try:
                transform = toolbox.compile(expr=ind)
                score = fw.score_new_transform(transform)
                if pd.isnull(score)==True:
                    score=bad_val
                fitnesses.append((score,))
            except:
                fitnesses.append((bad_val,))

        for ind, fit in zip(invalid_ind, fitnesses):
            if optimization=='maximum':
                if fit[0]>best_individual['score']:
                    best_individual['score']=fit[0]
                    best_individual['individual']=ind

            else:
                if fit[0]<best_individual['score']:
                    best_individual['score']=fit[0]
                    best_individual['individual']=ind

            ind.fitness.values = fit


        # halloffame.update(population)
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


        if (gen+1)%5==0:
            transform = toolbox.compile(expr=best_individual['individual'])
            score = fw.score_new_transform(transform)

            results_scores = {
                'audio_input': fw.original_data,
                'frequency_input': fw.original_rate,
                'audio_out': fw.audioOut,
                'frequency_out': fw.audioFs,
                'elgram':fw.elGram,
                'score': score

            }

            cp = dict(population=population, generation=gen, halloffame=best_individual,result_scores=results_scores,
                      logbook=logbook, rndstate=random.getstate())

            with open(f"checkpoints/{gen}_checkpoint.pkl", "wb") as cp_file:
                pickle.dump(cp, cp_file)

if __name__ == '__main__':
    import os
    wavefile_path = os.path.abspath('../../sample_data/bladerunner_replicant_test.wav')

    main(wavefile_path,pop_size=3,end_gen=2)