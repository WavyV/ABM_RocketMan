import collections
import concurrent
import pickle

from SALib.sample import saltelli
from SALib.analyze import sobol
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint

import run_model
import model_comparison

# All of the output measure functions with an identifying key.
# Each function takes a model (in end state) as first and only argument.
COMPARISONS = collections.OrderedDict({
    # "clusters": lambda m: sum(
    #     model_comparison.count_clusters(m.get_binary_model_state())),
    "happiness": lambda m: sum(sum(m.get_happiness_model_state())),
    # "entropy": lambda m: sum(
    #     model_comparison.get_entropy(m.get_binary_model_state())),
    # "lbp": lambda m: sum(
    #     model_comparison.get_entropy(m.get_binary_model_state())),
})

# Iterations to run each model for.
MODEL_ITERATIONS = 300

# Parameters in the weird format that SALib expects.
PARAMETERS = {
    "names": ["b1", "b2", "b3", "b4", "class_size"],
    "bounds": [
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [1, 240],
    ],
    "_defaults": [1, 1, 1, 1, 120]  # Not used by SALib, but by OFAT.
}

# Sobol parameters
NUM_SAMPLES = 1000  # Total samples using Saltelli sampling: `NUM_SAMPLES` * 12
RESULTS_FILENAME = "sobol_results.pickle"

# OFAT parameters
RUNS_PER_SAMPLE = 10
SAMPLES_PER_PARAM = 30


def valid_num(x):
    """Return 0 if invalid result, else given x."""
    if np.isinf(x):
        return 0
    return x


def get_samples(parameters, num_samples):
    """Return num_samples taken from the given parameter ranges."""
    parameters["num_vars"] = len(parameters["names"])
    samples = saltelli.sample(parameters, num_samples)
    return samples


def run(b1, b2, b3, b4, class_size, model_iterations, comparison_methods,
        fixed_class_size=None):
    """
    Run the model with given class size and beta coefficients.
    Return a list containing a result for each comparison measure.
    Each of these results is the sum of e.g. `get_entropy(final_state)`.

    Args:
        b1-b4: float, the coefficients for each beta term.
        class_size: float, class size that will be cast to an int.
        fixed_class_size: float, convenience for overriding class_size.
        model_iterations: int, amount of iterations to run each model.
        comparison_methods: dict of string to comparison function.
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
    return list(map(valid_num, comparison_values))


def run_sobol_analysis(parameters=PARAMETERS, num_samples=NUM_SAMPLES,
                       model_iterations=MODEL_ITERATIONS,
                       comparison_methods=COMPARISONS,
                       results_filename=RESULTS_FILENAME,
                       fixed_class_size=None):
    """Run, print and save sensitivity analysis.

    Args:
        parameters: dict of parameter ranges as expected by salib.
        num_samples: int, amount of samples, as per the argument to salib.
        model_iterations: int, amount of iterations to run each model.
        comparison_methods: dict of string to comparison function.
        results_filename: str, where to save the sobol results.
        fixed_class_size: int, fix class size to given number (note that in
            this case class size must not be in the given parameters)
    """
    samples = get_samples(parameters, num_samples)
    print("\nTotal runs: {}".format(samples.shape[0]))

    def sample_measures(sample):
        """Return output measures for one sample."""
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
    """Run OFAT for each of the parameters.

    Return a (samples_per_param, num_params, num_measures) size matrix.

    """
    # Set up results matrix and print some tasty info.
    results = np.empty(
        (samples_per_param, len(parameters["names"]), len(comparison_methods)))
    default_params = parameters["_defaults"]
    for i, param_name in enumerate(parameters["names"]):
        print("Default {} for parameter {}".format(
            default_params[i], param_name))
    print("\nTotal runs: {}".format(
        len(parameters["names"]) * samples_per_param * runs_per_sample))

    # First we iterate through each parameter, then the samples for that
    # parameter, then the amount of runs for that sample (averaged).
    run_count = 0
    for j, param_name in enumerate(parameters["names"]):
        bounds = parameters["bounds"][j]
        print("\nRunning OFAT on {}, bounds {}".format(param_name, bounds))
        param_values = np.linspace(*bounds, samples_per_param)
        for i, param_value in enumerate(param_values):
            sample_params = default_params[:]
            sample_params[j] = param_value
            sample_measures = []
            for _ in range(runs_per_sample):
                measures = run(
                    *sample_params,
                    model_iterations=model_iterations,
                    comparison_methods=comparison_methods)
                sample_measures.append(measures)
                run_count += 1
                print("\nRun: {}\nparameters: {}\nmeasures: {}".format(
                    run_count, sample_params, measures))
            # TODO: Why is this axis=0 and not axis=1 :s ? But it works so..
            sample_measures = np.array(sample_measures).mean(axis=0)
            print("avg measure: {}".format(sample_measures))
            # Get the mean of all sample values.
            results[i][j] = sample_measures
    return results


def ofat_single_comparison_results(results, comparison_index):
    """Return a new array like the given `run_ofat_analysis` results but each
    element is reduced from a list to a single value for one comparison method.

    """
    single_comparison_results = np.empty(results.shape[:2])
    for i in range(results.shape[0]):
        for j in range(results.shape[1]):
            single_comparison_results[i][j] = results[i][j][comparison_index]
    return single_comparison_results


def display_ofat_results(results, parameters=PARAMETERS,
                         comparison_methods=COMPARISONS):
    """Display OFAT results as returned from `run_ofat_analysis`.

    Args:
        results: the returned value from `run_ofat_analysis`.
        ...: the other two are the same parameters as to `run_ofat_analysis`.

    """
    for k, comparison_method in enumerate(comparison_methods):
        plot_data = ofat_single_comparison_results(results, k)
        # We plot all the betas on one graph and class size separately.
        f, (axis1, axis2) = plt.subplots(2)
        for j, param_name in enumerate(parameters["names"]):
            if param_name == "class_size":
                axis = axis2
            else:
                axis = axis1
            bounds = parameters["bounds"][j]
            axis.scatter(np.linspace(*bounds, results.shape[0]),
                         plot_data[:, j],
                         label=param_name)
            axis.set_title(comparison_method)
            axis.set_ylim([0, 30])
            axis.legend()
        plt.show()


if __name__ == "__main__":
    # Run with default settings.
    # run_sobol_analysis()

    # Run with fixed class size.
    run_sobol_analysis(fixed_class_size=200)

    # OFAT!
    # results = run_ofat_analysis()
    # display_ofat_results(results)
