import pickle
from toolboxes import create_toolbox
from deap import algorithms, tools,gp
import numpy as np
import random
from tqdm import tqdm

def main():

    toolbox,mstats,fw=create_toolbox()
    # Start a new evolution
    population = toolbox.population(n=50)
    start_gen = 0
    end_gen=300
    halloffame = tools.HallOfFame(maxsize=1)


    logbook = tools.Logbook()



    for gen in tqdm(range(start_gen, end_gen)):
        population = algorithms.varAnd(population, toolbox, cxpb=0.1, mutpb=0.1)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        halloffame.update(population)
        record = mstats.compile(population)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)

        population = toolbox.select(population, k=len(population))


        if gen % 10 == 0:

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