import collections
import os
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
parameters, used by both methods are also found similarly.

For OFAT analysis generate the data and analyze it separately:
    python3 sensitivity_analysis.py --ofat-run
    python3 sensitivity_analysis.py --ofat-analysis

For Sobol analysis generate the data and analyze it separately:
    python3 sensitivity_analysis.py --sobol-run
    python3 sensitivity_analysis.py --sobol-analysis

NOTE: Between a run and analysis the parameters below should remain unchanged.
"""


# OFAT parameters
RUNS_PER_SAMPLE = 10  # Amount of replicates per run.
SAMPLES_PER_PARAM = 15  # Points on the interval per parameter.
OFAT_RESULTS_FILENAME = "_ofat-runs-{}-samples-{}.pickle".format(
    RUNS_PER_SAMPLE, SAMPLES_PER_PARAM)

# Sobol parameters
SOBOL_SAMPLES = 1000  # Total Saltelli samples: `SOBOL_SAMPLES` * 12
SOBOL_REPLICATES = 3  # Replicates per each Sobol sample.
SOBOL_RESULTS_FILENAME = "_sobol-samples-{}-replicates-{}.pickle".format(
    SOBOL_SAMPLES, SOBOL_REPLICATES)

# The output measures used to analyze the final state of a model.
# Each function takes a model (in end state) as first and only argument.
COMPARISONS = collections.OrderedDict({
    # "happiness": lambda m: sum(sum(m.get_happiness_model_state())),

    "homogeneity": lambda m: model_comparison.get_characteristic_value(
        m.get_binary_model_state()),

    "correlation": lambda m: model_comparison.get_characteristic_value(
        m.get_binary_model_state(), method="correlation"),

    "rl_nonuniformity": lambda m: model_comparison.get_characteristic_value(
        m.get_binary_model_state(), method="rl_nonuniformity"),

    "rl_long_run_emphasis": lambda m: model_comparison.get_characteristic_value(
        m.get_binary_model_state(), method="rl_long_run_emphasis")
})

# Iterations to run each model for.
MODEL_ITERATIONS = 240

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

# Path to where OFAT and SOBOL results are saved.
RESULTS_PATH = "./sensitivity-analysis-data"


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


def run_sobol_analysis(parameters=PARAMETERS, num_samples=SOBOL_SAMPLES,
                       model_iterations=MODEL_ITERATIONS,
                       comparison_methods=COMPARISONS,
                       sobol_replicates=SOBOL_REPLICATES,
                       fixed_class_size=None):
    """Run, print and save sensitivity analysis.

    Args:
        parameters: dict, of parameter ranges, see PARAMETERS.
        num_samples: int, amount of samples, as per the argument to salib.
        model_iterations: int, amount of iterations to run each model.
        comparison_methods: dict of string to comparison function.
        sobol_replicates: int, replicates per each sample (averaged).
        fixed_class_size: int, fix class size to given number (note that in
            this case class size must not be in the given parameters)
    """
    parameters["num_vars"] = len(parameters["names"])
    samples = saltelli.sample(parameters, num_samples)
    print("\nSamples: {} x replicates: {} = total: {}".format(
        samples.shape[0], sobol_replicates,
        samples.shape[0] * sobol_replicates))

    def get_sample_measures(sample):
        """Return output measures for the given sample."""
        return run(*sample,
                   fixed_class_size=fixed_class_size,
                   model_iterations=model_iterations,
                   comparison_methods=comparison_methods)

    # Calculate measures for each sample and append to this results array.
    results = []
    sample_count = 0
    for sample_params in samples:

        # Run `sobol_replicates` replicates and take the mean of each measure.
        replicate_measures = []
        for _ in range(sobol_replicates):
            replicate_measures.append(get_sample_measures(sample_params))
        sample_measures = [np.mean(x) for x in np.array(replicate_measures).T]

        print("\nSample: {}\nparameters: {}\nmeasures: {}\nfixed class size: {}".format(
                sample_count, sample_params, sample_measures, fixed_class_size))
        results.append(sample_measures)
        sample_count += 1

    return results


def display_sobol_results(results, parameters=PARAMETERS,
                          comparison_methods=COMPARISONS):
    """Analyze and display Sobol analysis on the given data.

    Args:
        results: the data as returned by `run_sobol_analysis`.
        parameters: dict, of parameter ranges, see PARAMETERS.
        comparison_methods: dict of string to comparison function.
    """
    num_params = len(parameters["names"])
    parameters["num_vars"] = num_params
    print("Total samples: {}".format(len(results)))
    for i, comparison_method in enumerate(comparison_methods):
        print("\nSensitivity using {0}:\n".format(comparison_method))
        sensitivity = sobol.analyze(
            parameters, np.array(list(map(lambda x: x[i], results))))
        pprint(sensitivity)

        for key, order in zip(["S1", "ST"], ["first", "total"]):
            ax = plt.gca()
            ax.set_yticks(range(num_params))
            ax.set_yticklabels(parameters["names"])
            plt.title("{} order sensitivity of {}".format(
                order, comparison_method))
            ax.set_xlim([-0.1, 1.1])
            plt.errorbar(sensitivity[key],
                         range(num_params),
                         xerr=sensitivity["{}_conf".format(key)],
                         fmt="o",
                         capsize=4)

            plt.savefig(os.path.join(
                RESULTS_PATH,
                "sobol-measure-{}-samples-{}-order-{}-replicates-{}.png".format(
                    comparison_method, SOBOL_SAMPLES, order, SOBOL_REPLICATES)))
            plt.show()


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
    num_points = 4  # min, max, mean , variance
    results = np.empty(
        (samples_per_param,
         len(parameters["names"]),
         len(comparison_methods),
         num_points))
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
            E = np.empty((len(comparison_methods), num_points))
            # TODO: Why is this axis=0 and not axis=1 :s ? But it works so..
            mean = np.array(sample_measures).mean(axis=0)
            min_ = np.array(sample_measures).min(axis=0)
            max_ = np.array(sample_measures).max(axis=0)
            var = np.array(sample_measures).var(axis=0)
            print("min: {}".format(min_))
            print("mean: {}".format(mean))
            print("max: {}".format(max_))
            print("var: {}".format(var))
            for k in range(len(comparison_methods)):
                E[k] = [min_[k], max_[k], mean[k], var[k]]
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
        var_plot_data = ofat_single_comparison_results(results, k, 3)

        for j, param_name in enumerate(parameters["names"]):
            bounds = parameters["bounds"][j]
            print("variance: {} measure, parameter {}:\n\t{}".format(
                comparison_method, param_name, var_plot_data[:, j]))

            # for label, plot_data in zip(
            #         ["min", "max", "mean"],
            #         [min_plot_data, max_plot_data, mean_plot_data]):
            #     plt.scatter(np.linspace(*bounds, results.shape[0]),
            #                 plot_data[:, j],
            #                 label=label)
            #     plt.title("{} measure for parameter {}".format(
            #         comparison_method, param_name))

            err_min = mean_plot_data[:, j] - min_plot_data[:, j]
            err_max = max_plot_data[:, j] - mean_plot_data[:, j]
            plt.errorbar(np.linspace(*bounds, results.shape[0]),
                         mean_plot_data[:, j], yerr=[err_max, err_min],
                         ls='None', marker='o', ms=4, capsize=3)

            plt.title("{} measure for parameter {}".format(comparison_method, param_name))

            # plt.legend()
            plt.savefig(os.path.join(
                RESULTS_PATH,
                "ofat-measure-{}-parameter-{}-samples-{}-replicates-{}.png".format(
                    comparison_method, param_name, SAMPLES_PER_PARAM,
                    RUNS_PER_SAMPLE)))
            plt.show()


if __name__ == "__main__":
    ofat_results_path = os.path.join(RESULTS_PATH, OFAT_RESULTS_FILENAME)
    sobol_results_path = os.path.join(RESULTS_PATH, SOBOL_RESULTS_FILENAME)

    if "--ofat-run" in sys.argv:
        print("Starting OFAT run...\n")
        results = run_ofat_analysis()
        with open(ofat_results_path, "wb") as f:
            pickle.dump(results, f)
        print("\nSaved results to {}".format(ofat_results_path))

    elif "--ofat-analysis" in sys.argv:
        print("Starting OFAT analysis...\nLoaded results from {}".format(
            ofat_results_path))
        with open(ofat_results_path, "rb") as f:
            results = pickle.load(f)
        display_ofat_results(results)

    elif "--sobol-run" in sys.argv:
        print("Starting SOBOL run...\n")
        results = run_sobol_analysis()
        with open(sobol_results_path, "wb") as f:
            pickle.dump(results, f)
        print("\nSaved results to {}".format(sobol_results_path))

    elif "--sobol-analysis" in sys.argv:
        print("Starting SOBOL analysis...\nLoaded results from {}".format(
            sobol_results_path))
        with open(sobol_results_path, "rb") as f:
            results = pickle.load(f)
        display_sobol_results(results)

    else:
        print("Invalid flag")
