from classroom_seating import *
import network
import model_comparison
import matplotlib.pyplot as plt
import numpy as np
from noisyopt import minimizeSPSA

"""
Initialize the ClassroomModel

Args:
    coefs: coefficients for the utility function
    class_size: total number of students to be included into the social network
    seed: for random number generation

Returns:
    model: the created model instance
"""
def init_model(coefs, class_size, seed=0):

    # The (fixed) classroom layout
    blocks = [6, 14, 0]
    num_rows = 14
    width = sum(blocks) + len(blocks) - 1
    entrances = [(width-1, 0)]
    pos_utilities = np.zeros((width, num_rows))
    classroom = ClassroomDesign(blocks, num_rows, pos_utilities, entrances)

    # the probability distribution of sociability
    # TODO: Replace this by the distribution determined from the data
    s_distr = [0.2, 0.6, 0.2]

    # The social network of Students
    # TODO: Replace this by a network representing the real distribution of friendship!
    social_network = network.erdos_renyi(class_size, 0.2)

    # create the model
    model = ClassroomModel(classroom, coefs, sociability_distr=s_distr, social_network=social_network, seed=seed)

    return model

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
    deviation: mean deviation between model outputs and target outputs
"""
def objective_function(coefs, class_size, num_iterations, target_output, method):

    # assure that the coefficients sum up to one
    coefs = [(c/sum(coefs) if sum(coefs) > 0 else 0) for c in coefs]
    print("run the model with coefficients [{:.4f} {:.4f} {:.4f} {:.4f}]".format(coefs[0],coefs[1],coefs[2],coefs[3]))

    # run the model several times to handle stochasticity
    model_outputs = []
    aisles_x = []
    for seed in range(10):
        print("repetition {}".format(seed + 1))
        model = init_model(coefs, class_size, seed)
        for n in range(num_iterations):
            model.step()
        model_output = model.get_binary_model_state()
        model_outputs.append(model_output)
        aisles_x = model.classroom.aisles_x

        #plt.figure()
        #plt.imshow(model_output.T)
        #plt.show()

    # compute the error between model output and target output (averaged over the set of runs)
    deviation = np.mean([model_comparison.compare(m, target_output, method=method, aisles=aisles_x) for m in model_outputs])

    print("mean error = {:.4f}".format(deviation))
    return deviation

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

    class_size = 200 # a social network with 200 students is used
    num_iterations = 25 # 25 students are sampled from this network and enter the classroom one by one
    target_output = create_random_test_output(20, 13, num_iterations) # dummpy seating distribution for comparison
    method = 'lbp' # the method used for comparison

    # bounds for the parameters to be estimated
    bounds = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]

    # initial guess
    x0 = np.array([0.25, 0.25, 0.25, 0.25])

    # simultaneous perturbation stochastic approximation algorithm
    result = minimizeSPSA(objective_function, x0, args=(class_size, num_iterations, target_output, method), bounds=bounds, niter=50, paired=False)

    print(result)
