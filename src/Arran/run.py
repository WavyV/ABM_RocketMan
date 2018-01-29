# from mesa import *
from mesa.batchrunner import BatchRunner
from model import TheaterModel
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

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

model1 = TheaterModel(blocks=[0, 8, 8, 8, 0], sparsity=0.0)
model2 = TheaterModel(blocks=[0, 8, 8, 8, 0], sparsity=0.1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4))
im1 = ax1.imshow(model1.plan, vmin=-1, vmax=1)
ax1.axis('off')
ax1.set_title('No Network')
im2 = ax2.imshow(model2.plan, vmin=-1, vmax=1)
ax2.axis('off')
ax2.set_title('10%s Network'%'%')

annot = ax2.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax2:
        cont, ind = im2.contains(event)
        if cont:
            print(np.round(event.xdata), np.round(event.ydata))

            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

def animate(i):
    model1.step()
    model2.step()
    im1.set_data(model1.plan)
    im2.set_data(model2.plan)
    # print(i)
    return im1, im2

anim = animation.FuncAnimation(fig, animate, frames=100, interval=50)

fig.canvas.mpl_connect("motion_notify_event", hover)

plt.show()


# for i in range(288):
#     model.step()
#
# plt.imshow(model.theater)
# plt.show()

# model.print_theater()
# print(len(model.seated_students))
