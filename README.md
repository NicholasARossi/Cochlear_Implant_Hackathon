# Instuctions for running GP solution to hackathon

### Intro : 
Below is a codebase for our team's solution to the [2021 Cochlear Implant Hackathon](https://cihackathon.com/)

This solution relies hevily on the [DEAP](https://deap.readthedocs.io/en/master/) package. Reading the documentation there for [genetic programming](https://en.wikipedia.org/wiki/Genetic_programming) could be useful before understanding what's at play here.



### Getting started:
Navigate to the ```gp_refactor``` folder and run the following actions.
```bash
conda create -n deep_vibe python=3.7
brew install portaudio
python -m pip install -r requirements.txt 
```

You've got the key ingredients now you can run the evolution to find a solution.

This is not fleshed out as a CLI tool, but rather just for scripting. As such, you're going to use ```run_evolution.py``` for all of your needs. 

Basically the key variables here are
* **wavefile_path** this is a given wav file you want to optimze for.
* **pop_size** This is the size of the populations of solution you're going to run, the bigger, the longer it takes (3 is a good starting place if you want to watch it finish, rule of thumb - one minute per individual on a single core)
* **end_gen** How many iterations do you want to run the simulation for? More iterations takes more time.

