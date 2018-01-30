from collections import OrderedDict
import pickle

from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
from pprint import pprint

import run_model
import model_comparison

MODEL_ITERATIONS = 300
NUM_SAMPLES = 1000  # Total samples using Saltelli sampling: `NUM_SAMPLES` * 12
COMPARISONS = OrderedDict({
    "clusters": model_comparison.count_clusters,
    "entropy": model_comparison.get_entropy,
    "lbp": model_comparison.count_lbp
})
RESULTS_FILENAME = "sensitivity_results.pickle"
PARAMETERS = {
    "names": ["b1", "b2", "b3", "b4", "class_size"],
    "bounds": [
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [1, 240],
    ]
}


def get_samples(parameters, num_samples):
    """Return num_samples taken from the given parameter ranges."""
    parameters["num_vars"] = len(parameters["names"])
    samples = saltelli.sample(parameters, num_samples)
    print("Parameters shape: {0}".format(samples.shape))
    return samples


def run(b1, b2, b3, b4, class_size, model_iterations, comparison_methods):
    """
    Run the model with given class size and beta coefficients.
    Return a list containing a result for each comparison measure.
    Each of these results is the sum of e.g. `get_entropy(final_state)`.

    Args:
        b1-b4: float, the coefficients for each beta term.
        class_size: float, class size that will be cast to an int.
        model_iterations: int, amount of iterations to run each model.
        comparison_methods: dict of string to comparison function.
    """
    # Setup parameters for model.
    class_size = int(class_size)
    coefficients = [b1, b2, b3, b4]
    print("coefs: {} class_size: {}".format(coefficients, class_size))
    global _RUN_COUNTER
    print("\nRun: {} \nN: {} coefficients: {}".format(
        _RUN_COUNTER, class_size, coefficients))
    _RUN_COUNTER += 1

    # Setup initial model and run it
    model = run_model.init_default_model(coefficients, class_size)
    final_state = run_model.final_model_state(model, model_iterations)

    # Collect comparison measures and return them.
    comparison_values = []
    for comparison_method, comparison_f in comparison_methods.items():
        results = comparison_f(final_state)
        results_sum = sum(results)
        print("{} comparison: {}".format(comparison_method, results_sum))
        comparison_values.append(results_sum)
    return comparison_values


def run_sensitivity_analysis(parameters=PARAMETERS, num_samples=NUM_SAMPLES,
                             model_iterations=MODEL_ITERATIONS,
                             comparison_methods=COMPARISONS,
                             results_filename=RESULTS_FILENAME,
                             fixed_class_size=None):
    """Run, print and save sensitivity analysis.

    Args:
        parameters: dict of parameter ranges expected by salib.
        num_samples: int, amount of samples that salib will take.
        model_iterations: int, amount of iterations to run each model.
        comparison_methods: dict of string to comparison function.
        results_filename: str, where to save the sensitivity results.
        fixed_class_size: int, fix class size to given number (note that in
            this case class size must not be in the given parameters)
    """
    # Get samples and calculate results for each sample.
    samples = get_samples(parameters, num_samples)
    global _RUN_COUNTER
    _RUN_COUNTER = 0
    results = np.array(list(map(
        lambda run_params: run(
            *run_params,
            # Class size may be passed in as a parameter sample via
            # `run_params`. However we can also fix the class size by passing
            # in `fixed_class_size` as a keyword argument.
            **({} if fixed_class_size is None
               else {"class_size": fixed_class_size}),
            model_iterations=model_iterations,
            comparison_methods=comparison_methods),
        samples)))

    # Calculate and print sensitivity analysis results.
    for i, comparison_method in enumerate(comparison_methods):
        print("\nSensitivity using {0}:\n".format(comparison_method))
        sensitivity = sobol.analyze(
            parameters, np.array(list(map(lambda x: x[i], results))))
        pprint(sensitivity)

    # Finally save results to given filename.
    with open(results_filename, "wb") as f:
        f.write(pickle.dumps(sensitivity))
    print("Saved results to {0}".format(results_filename))


if __name__ == "__main__":
    # Run with default settings.
    # run_sensitivity_analysis()

    # Run with fixed class size.
    parameters = PARAMETERS
    del parameters["names"][0]
    del parameters["bounds"][0]
    run_sensitivity_analysis(
        fixed_class_size=200, parameters=parameters
    )
