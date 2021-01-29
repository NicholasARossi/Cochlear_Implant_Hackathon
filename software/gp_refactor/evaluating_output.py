

import numpy as np
from glob import glob
from software.AB_imports.Demo.proceduralDemo import demo4_procedural
from software.gp_refactor.fitness_functions import convert_sample_rate
from software.gp_refactor.toolboxes import custom_toolbox
from scipy.io.wavfile import write as write_wav
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import re
import argparse
import os

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]



class Audit:
    def __init__(
            self,
            wavefile_path: str,
            checkpoint_folder: str = 'checkpoints',
        ) -> None:
        self.wavefile_path = os.path.abspath(wavefile_path)
        self.checkpoint_folder = os.path.abspath(checkpoint_folder)

        ## load metric data
        csvs=glob(self.checkpoint_folder+'/*csv')
        if len(csvs)!=1:
            print('incorrect number of csvs in folder!')
        else:
            self.metric_df=pd.read_csv(csvs[0],header=[0, 1], index_col=0)

        # load and sort checkpoints
        checkpoints=glob(self.checkpoint_folder+'/*pkl')
        # sort
        checkpoints.sort(key=natural_keys)
        self.checkpoints=checkpoints

        self.toolbox = custom_toolbox(self.wavefile_path)
        self.figure_folder=os.path.join(self.checkpoint_folder,'figures')

        if not os.path.exists(self.figure_folder):
            os.makedirs(self.figure_folder)

    def _get_default_results(self):
        self.default_results = demo4_procedural(self.wavefile_path)





    def _compare_time_series(self):
        len_checkpoints=len(self.checkpoints)
        fig,ax=plt.subplots(len_checkpoints+2,1,figsize=(4,len_checkpoints*2),sharex=True,sharey=True,constrained_layout=True)
        results = pickle.load(open(self.checkpoints[0], "rb"))
        resampled_input = convert_sample_rate(results['result_scores']['audio_input'],
                                              results['result_scores']['frequency_input'],
                                              results['result_scores']['frequency_out'])

        ax[0].plot(resampled_input/(2**15-1))
        ax[0].set_title('Input Audio')

        ax[1].plot(self.default_results['audioOut'])
        ax[1].set_title('Default Algorithm')


        for l,checkpoint in enumerate(self.checkpoints):
            results = pickle.load(open(checkpoint, "rb"))
            ax[l + 2].plot(results['result_scores']['audio_out'])
            ax[l + 2].set_title(f'{checkpoint.split("/")[-1]}')

        fig.savefig(f'{self.figure_folder}/timeseries.png',dpi=300,bbox_inches='tight')


    def _compare_frequency(self):
        len_checkpoints=len(self.checkpoints)
        fig,ax=plt.subplots(len_checkpoints+2,1,figsize=(4,len_checkpoints*2),
                            sharex=True,sharey=True,
                            constrained_layout=True)
        results = pickle.load(open(self.checkpoints[0], "rb"))
        resampled_input = convert_sample_rate(results['result_scores']['audio_input'],
                                              results['result_scores']['frequency_input'],
                                              results['result_scores']['frequency_out'])

        input_fft = abs(np.fft.rfft(resampled_input / (2 ** 15 - 1))[:5000])

        ax[0].plot(input_fft)
        ax[0].set_title('Input Audio')

        ax[1].plot( abs(np.fft.rfft(self.default_results['audioOut']))[:5000])
        ax[1].set_title('Default Algorithm')


        for l,checkpoint in enumerate(self.checkpoints):
            results = pickle.load(open(checkpoint, "rb"))
            ax[l + 2].plot( abs(np.fft.rfft(results['result_scores']['audio_out'])[:5000]))
            ax[l + 2].set_title(f'{checkpoint.split("/")[-1]}')

        fig.savefig(f'{self.figure_folder}/frequency.png',dpi=300,bbox_inches='tight')


    def _bake_sound_files(self):
        results = pickle.load(open(self.checkpoints[0], "rb"))
        resampled_input = convert_sample_rate(results['result_scores']['audio_input'],
                                              results['result_scores']['frequency_input'],
                                              results['result_scores']['frequency_out'])


        sound_files=[resampled_input/ (2 ** 15 - 1),self.default_results['audioOut']]
        for l, checkpoint in enumerate(self.checkpoints):
            results = pickle.load(open(checkpoint, "rb"))
            sound_files.append(results['result_scores']['audio_out'])

        write_wav(f'{self.figure_folder}/comparison_wav.wav',results['result_scores']['frequency_out'],np.concatenate(sound_files))


    def _plot_metrics(self):
        fig,ax=plt.subplots(1,2,figsize=(12,4))

        ax[0].plot(self.metric_df.index,
                self.metric_df['fitness','avg'],
                label='average fitness')

        ax[0].plot(self.metric_df.index,
                self.metric_df['fitness','min'],
                label='minimum fitness')

        ax[0].plot(self.metric_df.index,
                self.metric_df['fitness','max'],
                label='maximum fitness')
        ax[0].legend()
        ax[0].set_title('Fitness')
        ax[0].set_xlabel('generations')

        ax[1].plot(self.metric_df.index,
                self.metric_df['size','avg'],
                label='average size')

        ax[1].plot(self.metric_df.index,
                self.metric_df['size','min'],
                label='minimum size')

        ax[1].plot(self.metric_df.index,
                self.metric_df['size','max'],
                label='maximum size')
        ax[1].legend()
        ax[1].set_title('Size')
        ax[1].set_xlabel('generations')
        fig.savefig(f'{self.figure_folder}/metrics.png',dpi=300,bbox_inches='tight')


    def run(self):
        self._get_default_results()
        self._compare_time_series()
        self._compare_frequency()
        self._bake_sound_files()
        self._plot_metrics()


if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    parser.add_argument('-w', '--wavefile_path', dest="wavefile_path", type=str,
                        default=os.path.abspath('./sample_data/bladerunner_replicant_test.wav'), help="Path to wavefile")

    parser.add_argument('-c', '--checkpoint_folder', dest="checkpoint_folder", type=str, default='checkpoints',
                        help="output folder for checkpoints")

    args = parser.parse_args()


    run_audit=Audit(wavefile_path=os.path.abspath(args.wavefile_path),
          checkpoint_folder=os.path.abspath(args.checkpoint_folder))
    run_audit.run()
