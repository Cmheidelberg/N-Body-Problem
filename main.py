import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from utils import *
import os

try:
    config = configparser.ConfigParser()
    config.sections()
    config.read(CONFIG_FILE_NAME) 

    # Frame
    WINDOW_SIZE = int(config["figure"]["window_size"])
    BACKGROUND_COLOR = literal_eval(config["figure"]["background_color"])
    DRAW_AXIS = config["figure"]["draw_axis"]
    
    # Animation
    DPI = int(config["animation"]["dpi"])
    FPS = int(config["animation"]["fps"])
    STEP = int(config["animation"]["step"])
    FRAMES = int(config["animation"]["frames"])

    # Name
    OUTPUT_NAME = config["file"]["name"]

except KeyError as e:
    print(f"Malconfigured {CONFIG_FILE_NAME}. KeyError: {e}")
    quit(1)
except ValueError as e:
    print(f"Value error in {CONFIG_FILE_NAME}: {e}")
    quit(1)
except SyntaxError as e:
    print(f"Syntax error in {CONFIG_FILE_NAME}: {e}")
    print("This probably means a Tuple value in the config is malformed")
    quit(1)

def render_cube(number, save_location):
    fig = plt.figure()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(BACKGROUND_COLOR)

    gradient = np.linspace(0, 1, number)
    X,Y,Z = np.meshgrid(gradient, gradient, gradient)
    colors=np.stack((X.flatten(),Y.flatten(),Z.flatten()),axis=1)
    ax.scatter(X,Y,Z,alpha=1.0,s=50,c=colors,marker='o',linewidth=0)
    plt.axis(DRAW_AXIS)
    fig.set_size_inches(WINDOW_SIZE, WINDOW_SIZE)
    
    
    anim = FuncAnimation(fig, update, frames=np.arange(0, FRAMES, STEP), repeat=True, fargs=(fig, ax))
    anim.save(save_file_name, dpi=DPI, fps=FPS,progress_callback=lambda i, n: print(f'Rendering: {int((i/n)*100)}%') if (i/n)*100 % 10 == 0 else False)
    print("done rendering.")
    print(f"File saved to: {save_file_name}")

def update(i, fig, ax):
    ax.view_init(elev=20., azim=i)
    return fig, ax


# MAIN
if __name__ == "__main__":

    # Create output file if it does not exist
    save_file_name = os.path.join(OUTPUT_DIR_LOCATION,OUTPUT_NAME)
    
    if not os.path.exists(OUTPUT_DIR_LOCATION):
        print("No output directory found. Making new one")
        os.mkdir(OUTPUT_DIR_LOCATION)

    inp = input("do you want to render new cube? [y/n]")
    if inp.lower() == 'y':
        amnt = input("how many?")
        amnt = int(amnt)
        render_cube(amnt,save_file_name)
    show_gif(save_file_name)
    print("Success")

