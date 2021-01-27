from software.gp_refactor.toolboxes import all_primitives
from pathos.multiprocessing import ProcessingPool as Pool
from deap import algorithms, tools
import pandas as pd
import os
import argparse
import pickle
import random
import time
import datetime


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


class Evolve:
    def __init__(
        self,
        wavefile_path: str,
        pop_size: int = 50,
        end_gen: int = 30,
        verbose: bool = True,
        optimization: str = 'maximum',
        max_depth: int = 3,
        checkpoint_folder: str = 'checkpoints',
        checkpoint_interval: int = 5
    ) -> None:
        '''
        :param wavefile_path: path (str), path to audio file
        :param pop_size: int, Population size for GP
        :param end_gen: int, Number of evolutions for the GP
        :param verbose: bool, verbose output True/False
        :param optimization: str, optimization method
        :param max_depth: int, Max depth of tree
        :param checkpoint_folder: str, folder for saving checkpoint pickle
        :param checkpoint_interval: int, intervals to save checkpoints at
        '''
        # Init basic params
        self.wavefile_path = wavefile_path
        self.pop_size = pop_size
        self.end_gen = end_gen
        self.verbose = True
        self.optimization = optimization
        self.max_depth = max_depth
        self.checkpoint_folder = checkpoint_folder
        self.checkpoint_interval = checkpoint_interval

        # Initialize logbook
        self.logbook = tools.Logbook()

        # Initialize primitives
        self.toolbox, self.mstats, self.fw = all_primitives(self.wavefile_path, optimization=self.optimization,
                                                            max_depth=self.max_depth)

        # Initialize score dictionary to store the best fit
        self.best_individual = {'score': self.bad_value, 'individual': ''}

        # create the output folder if it doesn't exist
        if not os.path.exists(self.checkpoint_folder):
            os.makedirs(self.checkpoint_folder)

        # Initialize multiple process pool
        self.pool = Pool()

        # Start a logging file if needed
        self.logfile = open(os.path.abspath(f'{self.checkpoint_folder}/logfile.txt'), 'w+')

    @property
    def bad_value(self):
        # Initialize bad value
        if self.optimization == 'maximum':
            return 0
        return 10000000

    @property
    def result_dict(self):
        return {'audio_input': self.fw.original_data,
                'frequency_input': self.fw.original_rate,
                'audio_out': self.fw.audioOut,
                'frequency_out': self.fw.audioFs,
                'elgram': self.fw.elGram}

    def run(self):

        # Start a new evolution
        population = self.toolbox.population(n=self.pop_size)
        # halloffame = tools.HallOfFame(maxsize=1)

        # start timer
        start_time = time.time()

        for gen in range(self.end_gen):
            population = self._evolve(population, gen, start_time)

        self.logfile.close()

    def _evolve(self, population, gen, start_time):
        '''
        Evolve first generation
        '''
        population = algorithms.varAnd(population, self.toolbox, cxpb=0.5, mutpb=0.1)

        # Select to re-evaluate the indices with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        # fitnesses = [self._evaluate_fitness(ind) for ind in invalid_ind]
        fitnesses = self.pool.map(self._evaluate_fitness, invalid_ind)
        self._update_score_dictionary(invalid_ind, fitnesses)

        # Generate and store results
        self._update_logbook(population, start_time, gen, len(invalid_ind))
        # Write logfile
        if self.verbose:
            self.logfile.writelines(self.logbook.stream)
            self.logfile.write("\n")
            print(self.logbook.stream)
        population = self.toolbox.select(population, k=len(population))

        # After every n iterations create a checkpoint
        if (gen + 1) % self.checkpoint_interval == 0:
            self._update_checkpoint(population, gen)

        return population

    def _evaluate_fitness(self, ind):
        '''
        Score the fitness of individuals
        '''
        try:
            transform = self.toolbox.compile(expr=ind)
            score = self.fw.score_new_transform(transform)
            print(score)
            if pd.isna(score):
                return (self.bad_value, self.bad_value)
            return score
        except:
            return (self.bad_value, self.bad_value)

    def _update_score_dictionary(self, invalid_ind, fitnesses):
        for ind, fit in zip(invalid_ind, fitnesses):
            # Update best score // to handel multi-parameter optimization we simply sum the values
            if (self.optimization == 'maximum' and sum(fit) > self.best_individual['score']) or \
                    (self.optimization != 'maximum' and sum(fit) < self.best_individual['score']):
                self.best_individual['score'] = sum(fit)
                self.best_individual['individual'] = ind

            # Update fitness value for the population index
            ind.fitness.values = fit

    def _update_logbook(self, population, start_time, gen, num_evals):
        '''
        Add time ran and time remaining to logbook
        '''
        # halloffame.update(population)
        record = self.mstats.compile(population)

        # Add time fields to record
        record.update(self._evalute_time_remaining(start_time, gen+1, self.end_gen))

        self.logbook.record(gen=gen, evals=num_evals, **record)

    @staticmethod
    def _evalute_time_remaining(start_time, gens_run, total_gen):
        '''
        Evaluate how much time is remaining and generate time dictionary
        '''
        # Calculate time elapsed
        time_run = int(time.time() - start_time)
        # Estimate how much time is left to finish
        time_remaining = int((time_run / gens_run) * (total_gen - gens_run))

        return {'time': {'total': datetime.timedelta(seconds=time_run),
                         'remaining': datetime.timedelta(seconds=time_remaining)}}

    def _update_checkpoint(self, population, gen):
        '''
        Create checkpoint
        '''
        transform = self.toolbox.compile(expr=self.best_individual['individual'])
        score = self.fw.score_new_transform(transform)

        # Generate result dict
        results_scores = self.result_dict.update({'score': score})

        cp = dict(population=population, generation=gen, halloffame=self.best_individual, result_scores=results_scores,
                  logbook=self.logbook, rndstate=random.getstate())

        # Save cp in a file
        with open(f"{self.checkpoint_folder}/{gen}_checkpoint.pkl", "wb") as cp_file:
            pickle.dump(cp, cp_file)


if __name__ == '__main__':

    import os
    get_deep_vibe_logo()

    parser = argparse.ArgumentParser()

    parser.add_argument('-w', '--wavefile_path', dest="wavefile_path", type=str,
                        default=os.path.abspath('./sample_data/bladerunner_replicant_test.wav'), help="Path to wavefile")
    parser.add_argument('-p', '--pop_size', dest="pop_size", type=int, default=50, help="Population size")
    parser.add_argument('-e', '--end_gen', dest="end_gen", type=int, default=30,
                        help="Number of generations for evolution")
    parser.add_argument('-v', '--verbose', dest="verbose", type=bool, default=True,
                        help="Verbose")
    parser.add_argument('-o', '--optimization', dest="optimization", type=str, default='maximum',
                        help="Optimization method")
    parser.add_argument('-d', '--max_depth', dest="max_depth", type=int, default=3,
                        help="maximum depth of tree during generation")
    parser.add_argument('-c', '--checkpoint_folder', dest="checkpoint_folder", type=str, default='checkpoints',
                        help="output folder for checkpoints")
    parser.add_argument('-i', '--checkpoint_interval', dest="checkpoint_interval", type=int, default=5,
                        help="save the checkpoints every n generations")

    args = parser.parse_args()

    run_evolution = Evolve(args.wavefile_path, pop_size=args.pop_size, end_gen=args.end_gen,
                           verbose=args.verbose, optimization=args.optimization, max_depth=args.max_depth,
                           checkpoint_folder=args.checkpoint_folder, checkpoint_interval=args.checkpoint_interval)
    run_evolution.run()
