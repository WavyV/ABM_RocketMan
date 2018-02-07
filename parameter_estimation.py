from model import *
from social import network
from data_processing import process_form, form_answers, observed_seating_patterns
import model_comparison
import run_model
import matplotlib.pyplot as plt
import numpy as np
from noisyopt import minimizeSPSA
import time
from os import path
import json
import sys
from scipy import stats
import matplotlib.pyplot as plt

MODEL_DATA_PATH = "model_output/parameter_estimation"
FILE_NAME = time.strftime("%Y%m%d-%H%M%S") + ".json"

DATA = ['fri-form.json', 'thurs-9-form.json', 'wed_24.json']
TARGET_OUTPUTS = [
    observed_seating_patterns.get_friday_seats(),
    observed_seating_patterns.get_thursday_seats(),
    observed_seating_patterns.get_wednesday_seats()]

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
    for i in range(len(DATA)):
        print("compare to the following dataset: {}".format(DATA[i]))
        # get target seating pattern from collected data
        #target_output = np.minimum(form_answers.get_seats(form_answers.load(DATA[i]), "seatlocation"),1)
        target_output = TARGET_OUTPUTS[i]
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
    Run the parameter estimation.

    Usage: python3 parameter_estimation.py run method

    where 'method' has to be one of {'entropy', 'lbp', 'cluster'}
    """
    if sys.argv[1] == "run":
        method = sys.argv[2]
        RESULTS_JSON = []
        #method = 'entropy' # the method used for comparison. One of {'lbp', 'cluster', 'entropy'}
        num_repetitions = 10 # number of runs with different random seeds per parameter combination and per dataset
        bounds = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]] # bounds for the parameters to be estimated
        x0 = np.array([0.25, 0.25, 0.25, 0.25]) # initial guess for parameters


        # simultaneous perturbation stochastic approximation algorithm
        result = minimizeSPSA(objective_function, x0,
                args=(num_repetitions, method),
                bounds=bounds, niter=200, paired=False)

        save_json(method)

        print(result)

    """
    Load and analyse the results from parameter estimation.

    Usage: python3 parameter_estimation.py load method/file_name.json [--plot]

    where the second argument specifies which results to load.
    If --plot is given, the seating patterns are plotted and saved for all datasets individually.
    """
    if sys.argv[1] == "load":
        file_path = sys.argv[2]
        with open(path.join(MODEL_DATA_PATH, file_path), mode='r', encoding='utf-8') as f:
            content = json.load(f)

        errors = [c.get("errors") for c in content]
        mean_errors = [c.get("mean_error") for c in content]
        coefs = [c.get("coefs") for c in content]

        # sort the entries based on increasing mean errors
        idx_sorted = np.argsort(mean_errors)
        coefs_sorted = np.array(coefs)[idx_sorted]
        mean_errors_sorted = np.array(mean_errors)[idx_sorted]
        errors_sorted = np.array(errors)[idx_sorted]

        # get the best performing parameter combination
        top_coefs = errors_sorted[0]

        # perform t-test on the error distributions of the best performing coefficients and all other coefficients
        p_values = []
        for other_coefs in errors_sorted:
             p_values.append(stats.ttest_ind(top_coefs, other_coefs)[1])


        # discard all parameter sets that differ significantly from the best performing one
        top_coefs = []
        top_mean_errors = []
        top_errors = []
        for i in range(len(p_values)):
            if p_values[i] > 0.1 and np.sum(coefs_sorted[i]) > 0:
                top_coefs.append(coefs_sorted[i])
                top_mean_errors.append(mean_errors_sorted[i])
                top_errors.append(errors_sorted[i])
                print("coefs: {}, \t mean error: {}".format(coefs_sorted[i], mean_errors_sorted[i]))

        # get some statistics about the remaining successful sets of coefficients
        final_coefs = np.mean(top_coefs, axis=0)
        print("average best coefs: {}".format(final_coefs))
        print("average error: {}".format(np.mean(top_errors)))
        print("error variance: {}".format(np.var(top_errors)))
        print("mean error variance: {}".format(np.var(top_mean_errors)))

        if "--plot" in sys.argv:
            for i in range(len(DATA)):
                # get target seating pattern from collected data
                target_output = TARGET_OUTPUTS[i]
                class_size = int(np.sum(target_output))
                # run model
                model = run_model.init_default_model(final_coefs, class_size, seed=0)
                for n in range(class_size):
                    model.step()
                model_output = model.get_binary_model_state()

                # plot model output and target output
                fig, (ax1, ax2) = plt.subplots(1,2)
                ax1.axis("off")
                ax2.axis("off")
                im_model = ax1.imshow(model_output, cmap="gray", interpolation=None)
                im_data = ax2.imshow(target_output, cmap="gray", interpolation=None)
                fig_name = file_path.split('.')[0] + "_" + DATA[i].split('.')[0]
                fig.savefig(path.join(MODEL_DATA_PATH, fig_name), bbox_inches='tight')
