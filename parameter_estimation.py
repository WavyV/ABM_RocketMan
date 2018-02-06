from model import *
from social import network
from data_processing import process_form, form_answers
import model_comparison
import run_model
import matplotlib.pyplot as plt
import numpy as np
from noisyopt import minimizeSPSA
import time
from os import path
import json
import sys

MODEL_DATA_PATH = "model_output/parameter_estimation"
FILE_NAME = time.strftime("%Y%m%d-%H%M%S") + ".json"

DATA = ['fri-form.json', 'thurs-9-form.json', 'wed_24.json']

RESULTS_JSON = []


"""
Compute the value of the objective function for parameter estimation.
Several seating processes with different random seeds are simulated and the output patterns are compared to the desired output.

Args:
    coefs: coefficients for the utility function
    class_size: total number of students to be included into the social network
    num_iterations: number of students that enter the classroom
    target_output: binary matrix representing the desired seating distribution
    method: {'lbp', 'cluster', 'entropy'} The method to be used to compute the profiles of the seating distributions

Returns:
    mean_error: mean MSE between model output profiles and target output profiles
"""
def objective_function(coefs, num_repetitions, method):

    # assure that the coefficients sum up to one
    coefs = [(c/sum(coefs) if sum(coefs) > 0 else 0) for c in coefs]
    print("###########################################################################")
    print("run the model with coefficients [{:.4f} {:.4f} {:.4f} {:.4f}]".format(coefs[0],coefs[1],coefs[2],coefs[3]))

    # run the model several times to handle stochasticity
    errors = []
    for dataset in DATA:
        print("compare to the following dataset: {}".format(dataset))
        # get target seating pattern from collected data
        target_output = np.minimum(form_answers.get_seats(form_answers.load(dataset), "seatlocation"),1)
        class_size = int(np.sum(target_output))

        for seed in range(num_repetitions):
            # run multiple simulations for each dataset
            print("repetition {}".format(seed + 1))
            model = run_model.init_default_model(coefs, class_size, seed)
            for n in range(class_size):
                model.step()
            model_output = model.get_binary_model_state()

            # compute the error between model output and target output
            aisles_x = model.classroom.aisles_x
            errors.append(model_comparison.compare(model_output, target_output, method=method, aisles=aisles_x))

    # compute the error averaged over the set of runs
    mean_error = np.mean(errors)
    print("mean error = {:.4f}".format(mean_error))

    # save the results
    RESULTS_JSON.append({"coefs":coefs, "errors":errors, "mean_error":mean_error})

    return mean_error


"""
Save the results from repeated simulation with given coefficients as a json file_name.
All results collected from one parameter estimation process are included into the same file.
"""
def save_json(folder):

    with open(path.join(MODEL_DATA_PATH, folder, FILE_NAME), mode='w', encoding='utf-8') as f:
        json.dump(RESULTS_JSON, f)

"""
Load the results from parameter estimation from the given folder.
"""
def save_json(folder):

    with open(path.join(MODEL_DATA_PATH, folder, FILE_NAME), mode='w', encoding='utf-8') as f:
        json.dump(RESULTS_JSON, f)


"""
Create a dummy seating distribution

Args:
    width: horizontal number of seats
    height: vertical number of seats
    num_iterations: number of Students

Returns:
    output: binary (width x height)-matrix
"""
def create_random_test_output(width, height, num_iterations):

    output = np.zeros((width,height))
    output[:num_iterations] = 1
    np.random.shuffle(output)

    return output


if __name__ == "__main__":

    """
    run the parameter estimation
    """
    if sys.argv[1] == "run":
        for method in ['entropy', 'lbp', 'cluster']:
            RESULTS_JSON = []
            #method = 'entropy' # the method used for comparison. One of {'lbp', 'cluster', 'entropy'}
            num_repetitions = 10 # number of runs with different random seeds per parameter combination and per dataset
            bounds = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]] # bounds for the parameters to be estimated
            x0 = np.array([0.25, 0.25, 0.25, 0.25]) # initial guess for parameters


            # simultaneous perturbation stochastic approximation algorithm
            # TODO: play with 'a'-values and 'niter' to get good results
            result = minimizeSPSA(objective_function, x0,
                    args=(num_repetitions, method),
                    bounds=bounds, niter=200, paired=False)

            save_json(method)

            print(result)

    if sys.argv[1] == "load":
        file_path = sys.argv[2]
        with open(path.join(MODEL_DATA_PATH, file_path), mode='r', encoding='utf-8') as f:
            content = json.load(f)

        mean_errors = [c.get("mean_error") for c in content]
        coefs = [c.get("coefs") for c in content]
        idx_sorted = np.argsort(mean_errors)
        top10_coefs = np.array(mean_errors)[idx_sorted][:10]
        top10_mean_errors = np.array(coefs)[idx_sorted][:10]
        for e,c in zip(top10_coefs, top10_mean_errors):
            print("coefs: {}, \t mean error: {}".format(c,e))
