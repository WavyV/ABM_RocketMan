import collections
import concurrent
import pickle
import sys

from SALib.sample import saltelli
from SALib.analyze import sobol
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint

import run_model
import model_comparison


"""
This module is for sensitivity analysis using OFAT and Sobol methods.

The OFAT and Sobol parameters are global variables, documented below. Other
parameters, used by both methods are also found as global variables, again
documented.

For OFAT analysis generate the data and analyze it separately:
    python3 sensitivity_analysis.py --ofat-run
    python3 sensitivity_analysis.py --ofat-analysis

For Sobol analysis generate the data and analyze it separately:
    python3 sensitivity_analysis.py --sobol-run
    python3 sensitivity_analysis.py --sobol-analysis
"""


# OFAT parameters
RUNS_PER_SAMPLE = 10  # Amount of replicates per run.
SAMPLES_PER_PARAM = 10  # Points on the interval per parameter.
OFAT_RESULTS_FILENAME = "_ofat_results.pickle"

# Sobol parameters
NUM_SAMPLES = 1000  # Total samples using Saltelli sampling: `NUM_SAMPLES` * 12
RESULTS_FILENAME = "sobol_results.pickle"

# The output measures used to analyze the final state of a model.
# Each function takes a model (in end state) as first and only argument.
COMPARISONS = collections.OrderedDict({
    "happiness": lambda m: sum(sum(m.get_happiness_model_state())),
    # "clusters": lambda m: sum(
    #     model_comparison.count_clusters(m.get_binary_model_state())),
    # "entropy": lambda m: sum(
    #     model_comparison.get_entropy(m.get_binary_model_state()))
})

# Iterations to run each model for.
MODEL_ITERATIONS = 300

# Parameters in the funky format that SALib expects.
# These parameters are used by both OFAT and Sobol.
PARAMETERS = {
    "names": ["b1", "b2", "b3", "b4", "class_size"],
    "bounds": [
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [1, 240],
    ],
    "_defaults": [1, 1, 1, 1, 120]  # Not used by Sobol, but by OFAT.
}


def run(b1, b2, b3, b4, class_size, model_iterations, comparison_methods,
        fixed_class_size=None):
    """
    Run the model with given class size and beta coefficients.
    Return a list containing a result for each given comparison method.

    Args:
        b1-b4: float, the coefficients for each beta term.
        class_size: float, class size that will be cast to an int.
        fixed_class_size: float, convenience way to override class_size.
        model_iterations: int, amount of iterations to run each model.
        comparison_methods: dict, of string to comparison function.
    """
    # Setup parameters.
    if fixed_class_size is not None:
        class_size = fixed_class_size
    class_size = int(class_size)
    coefficients = [b1, b2, b3, b4]

    # Setup initial model and run it.
    model = run_model.init_default_model(coefficients, class_size)
    final_model = run_model.final_model(model, model_iterations)

    # Collect comparison measures and return them.
    comparison_values = []
    for comparison_method, comparison_f in comparison_methods.items():
        comparison_values.append(comparison_f(final_model))
    return list(map(lambda x: 0 if np.isinf(x) else x, comparison_values))


def run_sobol_analysis(parameters=PARAMETERS, num_samples=NUM_SAMPLES,
                       model_iterations=MODEL_ITERATIONS,
                       comparison_methods=COMPARISONS,
                       results_filename=RESULTS_FILENAME,
                       fixed_class_size=None):
    """Run, print and save sensitivity analysis.

    Args:
        parameters: dict, of parameter ranges, see PARAMETERS.
        num_samples: int, amount of samples, as per the argument to salib.
        model_iterations: int, amount of iterations to run each model.
        comparison_methods: dict of string to comparison function.
        results_filename: str, where to save the sobol results.
        fixed_class_size: int, fix class size to given number (note that in
            this case class size must not be in the given parameters)
    """
    parameters["num_vars"] = len(parameters["names"])
    samples = saltelli.sample(parameters, num_samples)
    print("\nTotal runs: {}".format(samples.shape[0]))

    def sample_measures(sample):
        """Return output measures for the given sample."""
        return run(*sample,
                   fixed_class_size=fixed_class_size,
                   model_iterations=model_iterations,
                   comparison_methods=comparison_methods)

    # Calculate measures for each sample and append to this results array.
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        run_count = 0
        for measures, sample_params in (
                zip(executor.map(sample_measures, samples), samples)):
            print("\nRun: {}\nparameters: {}\nmeasures: {}\nfixed class size: {}".format(
                  run_count, sample_params, measures, fixed_class_size))
            run_count += 1
            results.append(measures)

    # Calculate and print sensitivity analysis results.
    for i, comparison_method in enumerate(comparison_methods):
        print("\nSensitivity using {0}:\n".format(comparison_method))
        sensitivity = sobol.analyze(
            parameters, np.array(list(map(lambda x: x[i], results))))
        pprint(sensitivity)

    # Finally save results to given filename.
    with open(results_filename, "wb") as f:
        f.write(pickle.dumps({
            "sensitivity": sensitivity,
            "data": results
        }))
    print("Saved results to {0}".format(results_filename))


def run_ofat_analysis(parameters=PARAMETERS,
                      samples_per_param=SAMPLES_PER_PARAM,
                      runs_per_sample=RUNS_PER_SAMPLE,
                      model_iterations=MODEL_ITERATIONS,
                      comparison_methods=COMPARISONS):
    """Run OFAT for each of the given parameters ranges.

    Return a (samples_per_param, num_params, num_comparisons) size matrix. Thus
    each row corresponds to a sample, and each column for a parameter. So if we
    consider the returned value as a 2d matrix, then each element E is an array
    of length equal to the amount of comparison methods. Each element of E is a
    length 3 array containing the min, max and mean values for that sample.

    Example E (min, max and mean for each comparison method):

        [[0.4, 2, 1.8], [1, 3.3, 1.4]]

    Args:
        parameters: dict, of parameter ranges, see PARAMETERS.
        samples_per_param: int, points on the interval for each parameter.
        runs_per_sample: int, the amount of replicates for each sample.
        model_iterations: int, amount of iterations to run each model.
        comparison_methods: dict of string to comparison function.

    """
    # Set up before the run, including results matrix.
    results = np.empty(
        (samples_per_param,
         len(parameters["names"]),
         len(comparison_methods),
         3))
    default_params = parameters["_defaults"]
    run_count = 0

    # Just printing some useful information before running.
    for i, param_name in enumerate(parameters["names"]):
        print("Default {} for parameter {}".format(
            default_params[i], param_name))
    print("\nTotal runs: {}".format(
        len(parameters["names"]) * samples_per_param * runs_per_sample))

    # Iterate through each parameter e.g. class_size.
    for j, param_name in enumerate(parameters["names"]):
        bounds = parameters["bounds"][j]
        print("\nRunning OFAT on {}, bounds {}".format(param_name, bounds))
        param_values = np.linspace(*bounds, samples_per_param)

        # Iterate through all the values for this parameter.
        for i, param_value in enumerate(param_values):
            sample_params = default_params[:]  # Copy of default parameters.
            sample_params[j] = param_value  # Set value for current parameter.
            sample_measures = []  # Collect results from each replicate here.

            # One run for each replicate.
            for _ in range(runs_per_sample):
                measures = run(
                    *sample_params,
                    model_iterations=model_iterations,
                    comparison_methods=comparison_methods)
                sample_measures.append(measures)
                run_count += 1
                print("\nRun: {}\nparameters: {}\nmeasures: {}".format(
                    run_count, sample_params, measures))

            # Set the element E (see function docstring) in results matrix.
            E = np.empty((len(comparison_methods), 3))
            # TODO: Why is this axis=0 and not axis=1 :s ? But it works so..
            mean = np.array(sample_measures).mean(axis=0)
            min_ = np.array(sample_measures).min(axis=0)
            max_ = np.array(sample_measures).max(axis=0)
            print("min: {}".format(min_))
            print("mean: {}".format(mean))
            print("max: {}".format(max_))
            for k in range(len(comparison_methods)):
                E[k] = [min_[k], max_[k], mean[k]]
            results[i][j] = E
    return results


def ofat_single_comparison_results(results, comparison_index, value_index):
    """This function takes the value returned by `run_ofat_analysis`, please
    understand that value first. Then in a new matrix, for each element E, it
    reduces E to a single value given by `comparison_index` and `value_index`.

    Example given `comparison_index=0` and `value_index=1`:

        E: [[0.4, 2, 1.8], [1, 3.3, 1.4]]

        Reduced value: 2

    """
    single_comparison_results = np.empty(results.shape[:2])
    for i in range(results.shape[0]):
        for j in range(results.shape[1]):
            single_comparison_results[i][j] = (
                results[i][j][comparison_index][value_index])
    return single_comparison_results


def display_ofat_results(results, parameters=PARAMETERS,
                         comparison_methods=COMPARISONS):
    """Display OFAT results that were returned from `run_ofat_analysis`.

    Args:
        results: the returned value from `run_ofat_analysis`.
        ...: the other two are the same parameters as to `run_ofat_analysis`.

    """
    for k, comparison_method in enumerate(comparison_methods):

        min_plot_data = ofat_single_comparison_results(results, k, 0)
        max_plot_data = ofat_single_comparison_results(results, k, 1)
        mean_plot_data = ofat_single_comparison_results(results, k, 2)

        for j, param_name in enumerate(parameters["names"]):
            bounds = parameters["bounds"][j]

            for label, plot_data in zip(
                    ["min", "max", "mean"],
                    [min_plot_data, max_plot_data, mean_plot_data]):
                plt.scatter(np.linspace(*bounds, results.shape[0]),
                            plot_data[:, j],
                            label=label)
                plt.title("{} measure for parameter {}".format(
                    comparison_method, param_name))

            plt.legend()
            plt.show()


if __name__ == "__main__":
    if "--ofat-run" in sys.argv:
        print("Starting OFAT run...\n")
        results = run_ofat_analysis()
        with open(OFAT_RESULTS_FILENAME, "wb") as f:
            pickle.dump(results, f)

    elif "--ofat-analysis" in sys.argv:
        print("Starting OFAT analysis...\n")
        with open(OFAT_RESULTS_FILENAME, "rb") as f:
            results = pickle.load(f)
        display_ofat_results(results)

    # Run with default settings.
    # run_sobol_analysis()

    # Run with fixed class size.
    # run_sobol_analysis(fixed_class_size=200)

    # OFAT!
    # results = run_ofat_analysis()
