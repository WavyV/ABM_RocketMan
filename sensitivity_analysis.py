from collections import OrderedDict
import pickle

from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
from pprint import pprint

import run_model
import model_comparison

MODEL_ITERATIONS = 300
SAMPLES = 100  # Total samples using Saltelli sampling: `SAMPLES` * 12
COMPARISONS = OrderedDict({
    "clusters": model_comparison.count_clusters,
    "entropy": model_comparison.get_entropy,
    "lbp": model_comparison.count_lbp
})
RESULTS_FILENAME = "sensitivity_results.pickle"
_RUN_COUNTER = 0


def run(class_size, b1, b2, b3, b4):
    """
    Run the model with given class size and beta coefficients.
    Return a list containing a value for each comparison method.
    Each of these values is the sum of e.g. `get_entropy(final_state)`.
    """
    class_size = int(class_size)
    global _RUN_COUNTER
    print("\nRun: {} \nN: {} b1: {} b2: {} b3: {} b4: {}".format(
        _RUN_COUNTER, class_size, b1, b2, b3, b4))
    _RUN_COUNTER += 1
    model = run_model.init_default_model([b1, b2, b3, b4], class_size)
    final_state = run_model.final_model_state(model, MODEL_ITERATIONS)
    comparison_values = []
    for comparison, comparison_f in COMPARISONS.items():
        comparison_value = sum(comparison_f(final_state))
        print("{} comparison: {}".format(comparison, comparison_value))
        comparison_values.append(comparison_value)
    return comparison_values


# Define parameters and sample using the Saltelli method.
parameters = {
    "names": ["N", "b1", "b2", "b3", "b4"],
    "bounds": [
        [1, 240],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
    ]
}
parameters["num_vars"] = len(parameters["names"])
samples = saltelli.sample(parameters, SAMPLES)
print("Parameters shape: {0}".format(samples.shape))

# Calculate model results and sensitivity and print.
results = np.array(list(map(lambda args: run(*args), samples)))
for i, comparison in enumerate(COMPARISONS):
    print("\nSensitivity using {0}:\n".format(comparison))
    sensitivity = sobol.analyze(
        parameters, np.array(list(map(lambda x: x[i], results))))
    pprint(sensitivity)

# Finally save results.
with open(RESULTS_FILENAME, "wb") as f:
    f.write(pickle.dumps(sensitivity))
print("Saved results to {0}".format(RESULTS_FILENAME))
