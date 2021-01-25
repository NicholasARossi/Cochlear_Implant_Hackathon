import pickle
from deap import algorithms, tools,gp
import random
import time
import datetime
from toolboxes import all_primitives,debugger
import pandas as pd


def get_deep_vibe_logo():
    logo = r''' 
    ///////// m i d n i g h t r u n ////////
           __                     _ __       
      ____/ /__  ___  ____ _   __(_) /_  ___ 
     / __  / _ \/ _ \/ __ \ | / / / __ \/ _ \
    / /_/ /  __/  __/ /_/ / |/ / / /_/ /  __/
    \__,_/\___/\___/ .___/|___/_/_.___/\___/ 
                  /_/                        

    ██████████████ 人工耳蜗软件 █████████████ 
    '''
    logo_lines = logo.splitlines()
    for line in logo_lines:
        print(line)


def main(wavefile_path, pop_size=50,end_gen=30,max_depth=4,verbose=True,optimization='maximum',outpath='checkpoints'):
    if not os.path.exists(outpath):
        os.makedirs(outpath)


    total_time = time.time()

    #toolbox,mstats,fw=all_primitives(wavefile_path,optimization=optimization)
    toolbox,mstats,fw=all_primitives(wavefile_path,max_depth=max_depth)

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
            # # some solutions produce non rational answers
            try:
                transform = toolbox.compile(expr=ind)
                score,score2 = fw.score_new_transform(transform)
                if pd.isnull(score)==True or pd.isnull(score2)==True:
                    score=bad_val
                    score2 = bad_val
                fitnesses.append((score,score2,))
            except:
                fitnesses.append((bad_val,bad_val,))
            # transform = toolbox.compile(expr=ind)
            # score,score2 = fw.score_new_transform(transform)
            # fitnesses.append((score,score2,))

        for ind, fit in zip(invalid_ind, fitnesses):
            if optimization=='maximum':
                if (fit[0]+fit[1])>best_individual['score']:
                    best_individual['score']=(fit[0]+fit[1])
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

            with open(f"{outpath}/{gen}_checkpoint.pkl", "wb") as cp_file:
                pickle.dump(cp, cp_file)

if __name__ == '__main__':
    import os
    # import argparse
    #
    # get_deep_vibe_logo()
    #
    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument('-w', '--wav_file',type=str, default='../../sample_data/bladerunner_replicant_test.wav',help='path file for analysis')
    # parser.add_argument('-p', '--pop_size',type=int, default=3,help='size of population')
    # parser.add_argument('-g', '--num_gen',type=int, default=2,help='number of generations')
    # parser.add_argument('-d', '--max_depth',type=int, default=4,help='max depth of trees during generation')
    #
    # parser.add_argument('-o', '--outpath',type=str, default='checkpoints',help='output directory')
    # parser.add_argument('-v', '--verbose',type=bool, default=True,help='verbose')
    #
    # args = parser.parse_args()
    #
    # job_args = os.path.abspath(args.wav_file)
    #
    # job_kwargs = {
    #     'pop_size': args.pop_size,
    #     'end_gen':args.num_gen,
    #     'outpath':args.outpath,
    #     'verbose':args.verbose,
    # }
    #
    #
    # main(job_args, **job_kwargs)

    main('../../sample_data/bladerunner_replicant_test.wav', pop_size=3,end_gen=2,max_depth=4,verbose=True,optimization='maximum',outpath='checkpoints')