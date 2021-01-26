import pickle
import sys
import os

from software.gp_refactor import classes
from software.gp_refactor.toolboxes import all_primitives, filters_only

sys.modules['classes'] = classes


wavefile_path = os.path.abspath('../../sample_data/bladerunner_replicant_test.wav')
toolbox, mstats, fw = filters_only(wavefile_path)

results = pickle.load(open('checkpoints/0_checkpoint.pkl', "rb"))

transform = toolbox.compile(expr=results['halloffame']['individual'])
score = fw.score_new_transform(transform)
print(score)

transform = toolbox.compile(expr=results['halloffame']['individual'])
score = fw.score_new_transform(transform)
print(score)
