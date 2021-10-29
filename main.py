import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import os
import scipy
import scipy.integrate
import configparser

from utils import *


try:
    config = configparser.ConfigParser()
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

    #simulation
    TIME = int(config["simulation"]["time"])
    RESOLUTION = int(config["simulation"]["resolution"])
    
    if TIME <= 200:
    
        if RESOLUTION%TIME != 0:
            RESOLUTION = int(np.round(RESOLUTION/TIME,decimals=0) * TIME)
            print(f"Warning: rounding up resolution to {RESOLUTION}.")            

    else:
        if RESOLUTION%(TIME*0.005) != 0:
            RESOLUTION = int(np.round(RESOLUTION/(TIME*0.005),decimals=0) * TIME)
            print(f"Warning: rounding up resolution to {RESOLUTION}.") 

    K1=TIME
    K2 = 36*K1

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

try:
    config = configparser.ConfigParser()
    config.read(BODY_FILE_NAME)
    BODIE_NAMES = config.sections()
    NUMBER_OF_BODIES = len(BODIE_NAMES)
    LOCATIONS = np.ones(len(BODIE_NAMES)*3, dtype="float64")
    VELOCITIES = np.ones(len(BODIE_NAMES)*3, dtype="float64")
    MASSES = np.ones(len(BODIE_NAMES), dtype="float64")
    COLORS = []

    for i,name in enumerate(BODIE_NAMES):
        LOCATIONS[i*3] = config[name]["location-x"]
        LOCATIONS[i*3+1] = config[name]["location-y"]
        LOCATIONS[i*3+2] = config[name]["location-z"] 

        VELOCITIES[i*3] = config[name]["velocity-x"]
        VELOCITIES[i*3+1] = config[name]["velocity-y"]
        VELOCITIES[i*3+2] = config[name]["velocity-z"] 

        MASSES[i] = config[name]["mass"]
        COLORS.append(config[name]["color"])        

except KeyError as e:
    print(f"Malconfigured {BODY_FILE_NAME}. KeyError: {e}")
    quit(1)
except ValueError as e:
    print(f"Value error in {BODY_FILE_NAME}: {e}")
    quit(1)
except SyntaxError as e:
    print(f"Syntax error in {BODY_FILE_NAME}: {e}")
    print("I dont know how you got a syntax error from the {BODY_FILENAME}")
    quit(1)

def render_animation(figure, axis,save_location):
    anim = FuncAnimation(figure, update, frames=np.arange(0, FRAMES, STEP), repeat=True, fargs=(figure, axis))
    anim.save(save_location, dpi=DPI, fps=FPS,progress_callback=lambda i, n: print(f'Rendering: {int((i/n)*100)}%') if (i/n)*100 % 10 == 0 else False)
    print("done rendering.")
    print(f"File saved to: {save_location}")

def ThreeBodyEquations(w,t):

    rads = [[0 for x in range(NUMBER_OF_BODIES)] for y in range(NUMBER_OF_BODIES)]
    for i in range(0,NUMBER_OF_BODIES):
        for j in range(i+1,NUMBER_OF_BODIES):
            if i != j:
                rads[i][j] = np.linalg.norm(w[j*3:j*3+3] - w[i*3:i*3+3])
                rads[j][i] = np.linalg.norm(w[j*3:j*3+3] - w[i*3:i*3+3])

    tmp = 0
    dvarr = []
    for i in range(0,NUMBER_OF_BODIES):
        for j in range(0,NUMBER_OF_BODIES):
            if i != j:
                tmp += K1*MASSES[j]*(w[j*3:j*3+3]-w[i*3:i*3+3])/rads[i][j]**3
        dvarr.append(tmp)
        tmp = 0

    drarr = []
    for i in range(0,NUMBER_OF_BODIES):
        drarr.append(K2*w[(i*3)+(NUMBER_OF_BODIES*3):(i*3)+(NUMBER_OF_BODIES*3)+3])

    r_derivatives = drarr[0]
    v_derivatives = dvarr[0]
    for i in range(1,NUMBER_OF_BODIES):
        r_derivatives = np.concatenate((r_derivatives,drarr[i])) 
        v_derivatives = np.concatenate((v_derivatives,dvarr[i]))
    
    flat_derivatives = np.concatenate((r_derivatives,v_derivatives))

    return flat_derivatives

# def calculate_center_of_mass:
#     #Update COM formula
#     position_com = 0
#     for p in MASSES:
#         curr_r = 
#     r_com=(m1*r1+m2*r2+m3*r3)/(m1+m2+m3)
#     #Update velocity of COM formula
#     v_com=(m1*v1+m2*v2+m3*v3)/(m1+m2+m3)
#     return 

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

        #Package initial parameters
        init_params=np.array([LOCATIONS,VELOCITIES]) #Initial parameters
        init_params=init_params.flatten() #Flatten to make 1D array
        time_span=np.linspace(0,0.005*TIME,RESOLUTION) #20 orbital periods and 500 points

        #Ode
        three_body_sol=scipy.integrate.odeint(ThreeBodyEquations,init_params,time_span)
        bodie_solutions = []
        for index in range(NUMBER_OF_BODIES):
            print(index)
            bodie_solutions.append(three_body_sol[:,index*3:index*3+3])
        
        #Create figure
        fig=plt.figure(figsize=(7,7))
        #Create 3D axes
        ax=fig.add_subplot(111,projection="3d")
        #Plot the orbits

        #Draw lines of motion
        for i in range(0,NUMBER_OF_BODIES):
            ax.plot((bodie_solutions[i])[:,0],(bodie_solutions[i])[:,1],(bodie_solutions[i])[:,2],color=COLORS[i])

        #Place a point on the end location
        for i in range(0,NUMBER_OF_BODIES):
            ax.scatter((bodie_solutions[i])[-1,0],(bodie_solutions[i])[-1,1],(bodie_solutions[i])[-1,2],color=COLORS[i],marker="o",s=50,label=BODIE_NAMES[i])

        #Labels
        ax.set_xlabel("x",fontsize=14)
        ax.set_ylabel("y",fontsize=14)
        ax.set_zlabel("z",fontsize=14)
        ax.set_facecolor(BACKGROUND_COLOR)
        #ax.set_title("Visualization of orbits of stars in a two-body system\n",fontsize=14)
        #ax.legend(loc="upper left",fontsize=14)
        plt.show()
    #show_gif(save_file_name)
    print("Success")

