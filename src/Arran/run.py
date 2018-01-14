# from mesa import *
from mesa.batchrunner import BatchRunner
from model import TheaterModel
import matplotlib.pyplot as plt

# fixed_params = {'students': 288,
#                 'num_of_rows': 12,
#                 'doors': [0],
#                 'sparsity': 0.9}
#
# variable_params = {'rows': ([0, 24, 0], [8, 8, 8], [12, 12])}
#
# batch_run = BatchRunner(TheaterModel,
#                         fixed_parameters=fixed_params,
#                         variable_parameters=variable_params,
#                         iterations=50,
#                         max_steps=288,
#                         model_reporters={'Density': compute_density})
#
# batch_run.run_all()
#
# run_data = batch_run.get_model_vars_dataframe()
# plt.scatter()

model = TheaterModel(blocks=[8, 8, 8], sparsity=0.0)

for i in range(288):
    model.step()

model.print_theater()
print(len(model.seated_students))
