import sys
from os import path

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.text import OffsetFrom

MODEL_DATA_PATH = "animation_data"
FILE_NAME = "model_data.json"

"""
Animate the seating process of previously generated simulation data

Usage:
    python3 animation.py [file_name_for_animation_data]
"""



"""
Mouse cursor interactivity
"""
def hover(event):
    for i, ax in enumerate(fig.axes):
        a = annotes[i]
        if event.inaxes == ax:
            cond, ind = images[i].contains(event)
            if cond:
                x, y = int(np.round(event.xdata)), int(np.round(event.ydata))
                value = images[i].get_array()[y][x]
                seat_utility, student_id, sociability, initial_happiness = model_data[i][y,x]

                text = ""
                if value == -0.8:
                    text = "AISLE"
                elif value == -2:
                    text = "DOOR"
                elif value <= -1:
                    text = "EMPTY SEAT\nSeat attractivity: {:.2f}\nAccessibility: {:.2f}".format(seat_utility, value + 2)
                elif value >= 0:
                    text = "FILLED SEAT\n {} {:.2f} \n {} {} \n {} {:.2f} \n {} {:.2f}".format("Seat attractivity:", seat_utility,
                                "Student sociability:", int(sociability), "Initial happiness (t = {}):".format(int(student_id)),
                                initial_happiness, "Current happiness:", value - 1)

                a.set_text(text)
                a.xy = (x, y)
                a.set_visible(True)
                fig.canvas.draw_idle()

        else:
            a.set_visible(False)
            fig.canvas.draw_idle()




"""
Returns images for a given iteration, from the given model data.
"""
def iteration_images(iteration, all_model_states):
    fig.canvas.set_window_title("Iteration: {}".format(iteration))
    for i in range(len(fig.axes)):
        image, info = all_model_states[i][iteration]
        images[i].set_data(image)
        model_data[i] = info

    return tuple(images)


"""
Animate the models using data loaded from a file.
"""
def animate_models(num_iterations):
    # First load the previously generated model data.
    #with open(MODEL_DATA_PATH, "rb") as f:
    #    model_coefs, model_names, all_model_states = pickle.load(f)
    # Initial animation state.
    iteration = 0
    running = True

    # Manages the logic deciding what the next iteration is.
    # Returns a tuple of images for the decided upon iteration.
    def next_iteration(_):
        nonlocal iteration
        if running:
            iteration += 1
        iteration = max(0, min(iteration, num_iterations - 1))
        return iteration_images(iteration, all_model_states)

    # Alter animation procedure based on user input.
    def key_press_handler(event):
        nonlocal iteration
        nonlocal running
        # Pause/resume animation on "p" key.
        if event.key == "p":
            running ^= True
        # Move to last frame on "e" key.
        elif event.key == "e":
            iteration = num_iterations - 1
        # Move to initial frame on "0" key.
        elif event.key == "0":
            iteration = -1
        # Move left on "←" key.
        elif event.key == "left":
            iteration -= 1
        # Move right on "→" key.
        elif event.key == "right":
            iteration += 1
        # Move 10 left on "a" key.
        elif event.key == "a":
            iteration -= 10
        # Move 10 right on "d" key.
        elif event.key == "d":
            iteration += 10

    fig.canvas.mpl_connect("key_press_event", key_press_handler)
    fig.canvas.mpl_connect("motion_notify_event", hover)

    anim = animation.FuncAnimation(fig, next_iteration, frames=None,
                                   interval=300)
    fig.tight_layout()
    plt.show()




if __name__ == "__main__":
    try:
        with open(path.join(MODEL_DATA_PATH, sys.argv[1]), "rb") as f:
            model_coefs, model_names, all_model_states = pickle.load(f)
    except:
        with open(path.join(MODEL_DATA_PATH, FILE_NAME), "rb") as f:
            model_coefs, model_names, all_model_states = pickle.load(f)

    """
    Initialize the plots
    """
    fig, axs = plt.subplots(1, len(model_names), figsize=(5*len(model_names),5))
    min_value, max_value = -2, 2
    images, model_data, annotes = [], [], []

    for i, ax in enumerate(fig.axes):

        image, info = all_model_states[i][0]
        images.append(ax.imshow(image, vmin=min_value, vmax=max_value, cmap = "RdYlGn", interpolation=None))
        ax.axis("off")
        ax.set_title(model_names[i])
        model_data.append(info)
        helper = ax.annotate("", xy=(0.5, 0), xycoords="axes fraction",
                      va="bottom", ha="center")
        offset_from = OffsetFrom(helper, (0.5, 0))
        a = ax.annotate("seat", xy=(0,0), xycoords="data",
                      xytext=(0, -10), textcoords=offset_from,
                      va="top", ha="center",
                      bbox=dict(boxstyle="round", fc="w"),
                      arrowprops=dict(arrowstyle="->"), alpha=1)
        a.set_visible(False)
        annotes.append(a)

    animate_models(len(all_model_states[0]))
