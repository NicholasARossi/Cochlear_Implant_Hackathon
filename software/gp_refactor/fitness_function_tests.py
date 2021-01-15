
import sys
import os

sys.path.append(os.path.abspath('../../Cochlear_Implant_Hackathon/'))

from software.gp_refactor import classes
sys.modules['classes'] = classes

from toolboxes import all_primitives,filters_only



import pickle

wavefile_path = os.path.abspath('../../sample_data/bladerunner_replicant_test.wav')
toolbox, mstats, fw = filters_only(wavefile_path)

results=pickle.load( open('checkpoints/0_checkpoint.pkl', "rb" ) )

transform = toolbox.compile(expr=results['halloffame']['individual'])
score = fw.score_new_transform(transform)
print(score)

transform = toolbox.compile(expr=results['halloffame']['individual'])
score = fw.score_new_transform(transform)
print(score)
