import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import os
import argparse
from numpy.lib.npyio import save
import scipy
import scipy.integrate
import configparser
from datetime import datetime

from utils import *

def update_config_parameters(path):
    '''Returns an array of variables with the values defined in the nbp config.'''
    try:
        config = configparser.ConfigParser()
        config.read(path) 

        # Frame
        WINDOW_SIZE = int(config["figure"]["window_size"])
        BACKGROUND_COLOR = literal_eval(config["figure"]["background_color"])
        DRAW_AXIS = config["figure"]["draw_axis"]
        DRAW_LEGEND = config["figure"]["draw_legend"]
        FIGURE_TITLE = config["figure"]["title"]
        


        # Name
        OUTPUT_NAME = config["outputs"]["name"]

        #simulation
        TIME = int(config["simulation"]["time"])
        RESOLUTION = int(config["simulation"]["resolution"])
        CENTER_ON_BODY = config["simulation"]["center_camera_on_body"]
        
        if TIME <= 200:
        
            if RESOLUTION%TIME != 0:
                RESOLUTION = int(np.round(RESOLUTION/TIME,decimals=0) * TIME)
                print(f"Warning: rounding up resolution to {RESOLUTION}.")            

        else:
            if RESOLUTION%(TIME*0.005) != 0:
                RESOLUTION = int(np.round(RESOLUTION/(TIME*0.005),decimals=0) * TIME)
                print(f"Warning: rounding up resolution to {RESOLUTION}.") 

        K1=TIME
        K2 = VELOCITY_SCALER*K1
        return WINDOW_SIZE,BACKGROUND_COLOR,DRAW_AXIS,FIGURE_TITLE,DRAW_LEGEND,OUTPUT_NAME,TIME,RESOLUTION,CENTER_ON_BODY,K1,K2

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


def update_body_parameters(path):
    '''Returns an aray of variables from their values defined in the body config'''
    try:
        config = configparser.ConfigParser()
        config.read(path)
        BODY_NAMES = config.sections()
        NUMBER_OF_BODIES = len(BODY_NAMES)
        LOCATIONS = np.ones(len(BODY_NAMES)*3, dtype="float64")
        VELOCITIES = np.ones(len(BODY_NAMES)*3, dtype="float64")
        MASSES = np.ones(len(BODY_NAMES), dtype="float64")
        COLORS = []
        for i,name in enumerate(BODY_NAMES):

            tmp_locations = config[name]["locations"].strip('][').split(",")
            tmp_velocities = config[name]["velocities"].strip('][').split(",")

            LOCATIONS[i*3] = tmp_locations[0]
            LOCATIONS[i*3+1] = tmp_locations[1]
            LOCATIONS[i*3+2] = tmp_locations[2]

            VELOCITIES[i*3] = tmp_velocities[0]
            VELOCITIES[i*3+1] = tmp_velocities[1]
            VELOCITIES[i*3+2] = tmp_velocities[2] 

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
    return BODY_NAMES,NUMBER_OF_BODIES,LOCATIONS,VELOCITIES,MASSES,COLORS


# ======= Global Variables =======
start = 0
end = 100

print(f"DEBUG: {CONFIG_FILE_PATH}")
BODY_NAMES,NUMBER_OF_BODIES,LOCATIONS,VELOCITIES,MASSES,COLORS = update_body_parameters(BODY_FILE_PATH)
WINDOW_SIZE,BACKGROUND_COLOR,DRAW_AXIS,FIGURE_TITLE,DRAW_LEGEND,OUTPUT_NAME,TIME,RESOLUTION,CENTER_ON_BODY,K1,K2 = update_config_parameters(CONFIG_FILE_PATH)
save_equation = None
load_equation = None  

def parse_timestep(input_string):
    if not len(input_string) > 0:
        print("Could not parse --display_timesteps argument. using [0,100]")
        return 0, 100
    
    if input_string[0] == '[':
        input_string = input_string[1:]
    
    if input_string[-1] == ']':
        input_string = input_string[:-1]
    
    split = input_string.split("-")
    split = input_string.split(",")

    if split[0].isnumeric() and split[-1].isnumeric():
        start = int(split[0])
        end = int(split[-1])

        if start <= end:
            return start,end
        else:
            print(f"WARNING: Cannot have a start value larger than the end value. Using [{end},{start}] instead")
            return end,start
    else:
        print("Could not parse --display_timesteps argument correctly. Setting range to [0,100]")
        return 0, 100

parser = argparse.ArgumentParser(description='N-Body Simulator command line arguments. An interactable user menu is given if no arguments are given. By default the programs config and the body config used are located in the root of the project\'s directory')
# parser.add_argument('--time','-t', type=int, nargs=1, 
#                     help='Set the total time to run the simulation for. (unitless)')

# parser.add_argument('--resolution','-r', type=int, nargs=1, 
#                     help='Set the resolution value for the calculations. (unitless)')

parser.add_argument('--center_on','-c', type=str, nargs=1, 
                    help='Name of the body to center the frame of reference on. If body cannot be found no reference is set.')

parser.add_argument('--display_timesteps','-ts', type=parse_timestep,
                    help="Specify a percentage range of the figure to display. For example, to graph only the first half input [0,50].")

parser.add_argument('--config_path','-p', type=str, nargs=1, 
                    help="Specify path to config file")

parser.add_argument('--body_config_path','-b', type=str, nargs=1, 
                    help="Specify the body config path. (start position and velocity of bodies)")
                  
parser.add_argument('--no_prompt', '-np', action='store_true',
                    help="Dissable interactive menu prompt.")

parser.add_argument('--save_output','-s', type=str, nargs=1, 
                    help='Output directory to save solved equation to (saves as .nbp)')

parser.add_argument('--load_equation','-l', type=str, nargs=1, 
                    help='Load solved equation. (.nbp file)')

# def render_animation(figure, axis,save_location):
#     anim = FuncAnimation(figure, update, frames=np.arange(0, FRAMES, STEP), repeat=True, fargs=(figure, axis))
#     anim.save(save_location, dpi=DPI, fps=FPS,progress_callback=lambda i, n: print(f'Rendering: {int((i/n)*100)}%') if (i/n)*100 % 10 == 0 else False)
#     print("done rendering.")
#     print(f"File saved to: {save_location}")


def offset_solition_by_body_location(body_solutions, index):
    '''
    Set center of reference frame to the body in a given index. Note: to set the reference frame to the center of mass pass -1 as index.
    
    :param body_solutions: Body solutions for the nbp
    :type body_solutions: list

    :param index: Index of body name to center on. If value is -1 then center of c.o.m of system
    :type index: int

    :return offset: offset body solutions centered around body in index (or center of mass)

    '''
    offset = body_solutions
    for curr in range(len(body_solutions[index])):
        
        # If we are given a specific body index to center on
        if index != -1:
            x = body_solutions[index][curr,0]
            y = body_solutions[index][curr,1]
            z = body_solutions[index][curr,2]
        
        # If we are not given -1 center on center of mass of system
        else:
            x,y,z = calculate_center_of_mass(body_solutions, curr)

        x_offset = -x
        y_offset = -y
        z_offset = -z

        for i in range(len(body_solutions)):  
            offset[i][curr,0] += x_offset
            offset[i][curr,1] += y_offset
            offset[i][curr,2] += z_offset
    return offset


def calculate_center_of_mass(body_solutions, curr):
    '''Given the body_solutions and current index return an x,y, and z position for center of mass'''

    com_x = 0
    com_y = 0
    com_z = 0
    num_of_bodies = len(body_solutions)

    for i in range(num_of_bodies):
        com_x += body_solutions[i][curr,0] * MASSES[i]
        com_y += body_solutions[i][curr,1] * MASSES[i]
        com_z += body_solutions[i][curr,2] * MASSES[i]
    
    com_x = com_x/num_of_bodies
    com_y = com_y/num_of_bodies 
    com_z= com_z/num_of_bodies

    return com_x, com_y, com_z


def NBodySimulation(w,t):

    # Prime with array of 0's
    rads = [[0 for x in range(NUMBER_OF_BODIES)] for y in range(NUMBER_OF_BODIES)]
    
    for i in range(0,NUMBER_OF_BODIES):
        for j in range(i+1,NUMBER_OF_BODIES):
            if i != j:
                rads[i][j] = np.linalg.norm(w[j*3:j*3+3] - w[i*3:i*3+3])
                rads[j][i] = np.linalg.norm(w[j*3:j*3+3] - w[i*3:i*3+3])

    dvarr = []
    drarr = []
    for i in range(0,NUMBER_OF_BODIES):
        tmp = 0
        for j in range(0,NUMBER_OF_BODIES):
            if i != j:
                tmp += K1*MASSES[j]*(w[j*3:j*3+3]-w[i*3:i*3+3])/rads[i][j]**3
        dvarr.append(tmp)
        drarr.append(K2*w[(i*3)+(NUMBER_OF_BODIES*3):(i*3)+(NUMBER_OF_BODIES*3)+3])

    r_derivatives = drarr[0]
    v_derivatives = dvarr[0]
    for i in range(1,NUMBER_OF_BODIES):
        r_derivatives = np.concatenate((r_derivatives,drarr[i])) 
        v_derivatives = np.concatenate((v_derivatives,dvarr[i]))
    
    flat_derivatives = np.concatenate((r_derivatives,v_derivatives))

    return flat_derivatives


def get_nbp_solutions():
    #Package initial parameters
    init_params=np.array([LOCATIONS,VELOCITIES])
    init_params=init_params.flatten()
    time_span=np.linspace(0,0.005*TIME,RESOLUTION)
    #Solve or load ODEs
    if load_equation is not None:
        with open(load_equation, 'rb') as f:
            three_body_sol = np.load(f)
    else:
        three_body_sol=scipy.integrate.odeint(NBodySimulation,init_params,time_span)
    
    return three_body_sol


def parse_save_load_name(filename):
    '''Add .nbp file extension to the end of the given filename string if file extension does not exist.'''
    tmp = filename.split('.')
    if len(tmp) == 1:
        return filename + ".nbp"
    return filename


def interactive_menu():
    global BODY_NAMES,NUMBER_OF_BODIES,LOCATIONS,VELOCITIES,MASSES,COLORS
    global WINDOW_SIZE,BACKGROUND_COLOR,DRAW_AXIS,FIGURE_TITLE,DRAW_LEGEND,OUTPUT_NAME,TIME,RESOLUTION,CENTER_ON_BODY,K1,K2
    global start, end
    dirs = None
    dir_index = 0
    print("====== N Body Problem ======") 

    title = "Select how you would like to run the simulation"
    options = []
    options.append("Use preset config")         # 1
    options.append("Input path to my config")   # 2
    options.append("Use default configs")       # 3
    options.append("Load solved problem")       # 4 
    options.append("Quit")                      # 5
    help_text = ("\n[SELECTION INFORMATION]\n" 
                "1. Use preset config: present a list of preset N Body Problems to simulate. " 
                "This is intended to show possible usecases of the simulator. \n2. input path to my config: "
                "prompt you to enter the path for your simulator config and body config files.\n3. Use "
                "default configs will use the config.cfg and bodies.cfg located in the program's root\n4. Load "
                "solved problem: lets you select a .nbp file (from a previous saved problem) to quickly view "
                "the results or edit some post processing settings.\n5. Quit program\n6. Show this text\n")

    selection = simple_menu_print(title, options,help_text)
    body_path = None
    curr_config_path = None
    nbp_solutions = None

    if(selection == 1):
        title = "Please select a preset to simulate"
        dirs = os.listdir(PRESETS_DIR)
        dir_index = directory_select_menu(dirs)
        body_path = os.path.join(PRESETS_DIR, dirs[dir_index])
        curr_config_path = CONFIG_FILE_PATH
    elif(selection == 2):
        slow_print_text("Please enter a path to your body config file")
        body_path = input()
        
        slow_print_text("Please enter a path to your N Body Problem config file")
        curr_config_path = input()
    elif(selection == 3):
        body_path = None
        curr_config_path = None
    elif(selection == 4):
        title = "How would you like to load?"
        options = []
        options.append("Select run from default save folder")
        options.append("Input path to file")
        options.append("Quit")
        selection = simple_menu_print(title, options)
        
        # TODO: save/load needs to save and load config and body cfg with the .nbp file so the figure can be remade form it
        if(selection == 1):
            dirs = os.listdir(SAVED_RUNS_DIR)
            dir_index = directory_select_menu(dirs)
            solved_dir = os.path.join(SAVED_RUNS_DIR, dirs[dir_index])
            load_equation = parse_save_load_name(solved_dir)
            with open(load_equation, 'rb') as f:
                nbp_solutions = np.load(f)

            print("Load successful")

        elif(selection == 2):
            slow_print_text("Please enter path to the file: ")
            solved_dir = input()
            load_equation = parse_save_load_name(solved_dir)
            with open(load_equation, 'rb') as f:
                nbp_solutions = np.load(f)
        else:
            print("Quitting...")
            exit(1)


    else:
        print("Quitting...")
        exit(1)

    if(body_path is not None):
        BODY_NAMES, NUMBER_OF_BODIES,LOCATIONS,VELOCITIES,MASSES,COLORS = update_body_parameters(body_path)
    if(curr_config_path is not None):
        WINDOW_SIZE,BACKGROUND_COLOR,DRAW_AXIS,FIGURE_TITLE,DRAW_LEGEND,OUTPUT_NAME,TIME,RESOLUTION,CENTER_ON_BODY,K1,K2 = update_config_parameters(curr_config_path)

    if(nbp_solutions is None):
        time_before_run = datetime.now()
        print("Running simulation. This may take a while...")
        nbp_solutions = get_nbp_solutions()
        time_after_run = datetime.now()
        print(f"Done! Time elapsed {time_after_run - time_before_run}")

    # Flatten the solutions array for drawing
    body_solutions = []
    for index in range(int(len(nbp_solutions[0])/6)):
        body_solutions.append(nbp_solutions[:,index*3:index*3+3])

    title = "What would you like to do?"
    options = []
    options.append("Show figure")                              # 1
    options.append("Choose reference frame")                   # 2
    options.append("Select start/end time")                    # 3
    options.append("Save run")            # 4
    options.append("Figure Options")                           # 5
    options.append("Rerun Simulation with different timestep") # 6
    options.append("Quit")                                     # 7

    help_text = ("\n[POST SIMULATION HELP]\n"
                 "1. Draws the figure showing path of bodies over time\n"
                 "2. Centers the reference frame on a selected body\n"
                 "3. Trim the start/end time of the lines drawn to the figure\n"
                 "4. Save the solution so it can be quickly loaded next time without needing to rerun the whole \n" 
                 "simulation. Note: This only saves the bodies positions for the simulator to load later list so \n"
                 "reference frame and figure settings are not saved.\n"
                 "5. Modify how the figure looks when shown\n"
                 "6. Quit simulator"
                )
    
    curr_solution = body_solutions
    while(True):
        # Post processing section
        selection = simple_menu_print(title, options,help_text)
        if(selection == 1):
            draw_figure(curr_solution)
        if(selection == 2):
            center_reference_title = "Select what to center the reference on"
            center_reference_options = []
            center_reference_options.append("Center of mass")
            for body_name in BODY_NAMES:
                center_reference_options.append(body_name)
            center_reference_selection = simple_menu_print(center_reference_title,center_reference_options)
            curr_solution = offset_solition_by_body_location(body_solutions, center_reference_selection-2)
        if(selection == 3):
            print("What percentage of the run should the figure start drawing at (between 0 and 100). For example, to start one third of the way through the full simulation input 33")
            start = get_float_input(f"Input a start value. (current value {start})",low=0,high=100)
            print(f"What percentage of the run should the figure stop drawing at (between {start} and 100)")
            end = get_float_input(f"Input an end value. (current value {end})",low=start,high=100)
        if(selection == 4):
            slow_print_text("What would you like to name the run?")
            name = input()

            if name is None or len(name) == 0:
                print("Invalid name. Using \"default\" instead")
                name = "default"
            save_path = os.path.join(SAVED_RUNS_DIR,name)
            save_equation = parse_save_load_name(save_path)
            print(f"Saving run to: {save_path}")
            with open(save_equation, 'wb') as f:
                np.save(f, nbp_solutions)  
            print("done!\n")

        if(selection == 5):
            figure_options_title = "What would you like to do?"
            figure_options_options = []
            figure_options_options.append("Toggle axis on/off")      # 1
            figure_options_options.append("Toggle legend on/off")    # 2
            figure_options_options.append("Change background color") # 3
            figure_options_options.append("Set title")               # 4
            figure_options_options.append("Back")                    # 5
            figure_options_done = False

            while(not figure_options_done):
                figure_options_selection = simple_menu_print(figure_options_title, figure_options_options)
                if(figure_options_selection == 1):
                    axis_options_title = "Would you like the axis on or off?"
                    axis_options_options = []
                    axis_options_options.append("Set on")
                    axis_options_options.append("Set off")
                    axis_options_selection = simple_menu_print(axis_options_title,axis_options_options)
                    if(axis_options_selection == 1):
                        DRAW_AXIS = "on"
                    else:
                        DRAW_AXIS = "off"
                elif(figure_options_selection == 2):
                    legend_options_title = "Would you like the legend on or off?"
                    legend_options_options = []
                    legend_options_options.append("Set on")
                    legend_options_options.append("Set off")
                    legend_options_selection = simple_menu_print(legend_options_title,legend_options_options)
                    if(legend_options_selection == 1):
                        DRAW_LEGEND = "on"
                    else:
                        DRAW_LEGEND = "off"
                elif(figure_options_selection == 3):
                    formatted_background_color_entered = False
                    while(not formatted_background_color_entered):
                        slow_print_text("Please enter a background color tupe each element ranging between 0 and 1 for RGB background. ie: (0.9,0.9,0.9)")
                        bgc = input()
                        try:
                            BACKGROUND_COLOR = literal_eval(bgc)
                            formatted_background_color_entered = True
                        except Exception as e:
                            print("Invalid background color")
                elif(figure_options_selection == 4):
                    slow_print_text("Enter a title for the figure. (or leave blank to remve current title)")
                    FIGURE_TITLE = input()
                else:
                    print("Saving figure options")
                    figure_options_done = True
        elif(selection == 6):
            TIME = get_int_input(f"What time number would you like to use? Current run used {TIME}",low=0)
            RESOLUTION = get_int_input(f"What resolution number would you like to use? Current run used {RESOLUTION}",low = 0)

            time_before_run = datetime.now()
            print("Re-Running simulation. This may take a while...")
            nbp_solutions = get_nbp_solutions()
            time_after_run = datetime.now()
            print(f"Done! Time elapsed {time_after_run - time_before_run}")

            # Flatten the solutions array for drawing
            body_solutions = []
            for index in range(int(len(nbp_solutions[0])/6)):
                body_solutions.append(nbp_solutions[:,index*3:index*3+3])
            curr_solution = body_solutions

        elif(selection == 7):
            print("Quitting...")
            quit(0)
                

def draw_figure(nbp_solutions):
        #Create figure
        fig=plt.figure(figsize=(WINDOW_SIZE,WINDOW_SIZE))
        #Create 3D axes
        ax=fig.add_subplot(111,projection="3d")
        #Plot the orbits

        #Draw lines of motion
        length = len(nbp_solutions[0][:,0])
        start_index = int(start*0.01*length)
        end_index = int(end*0.01*length)-1
        for i in range(0,NUMBER_OF_BODIES):
            x = (nbp_solutions[i])[start_index:end_index,0]
            y = (nbp_solutions[i])[start_index:end_index,1]
            z = (nbp_solutions[i])[start_index:end_index,2]
            ax.plot(x,y,z,color=COLORS[i])

        #Place a point on the end location
        for i in range(0,NUMBER_OF_BODIES):
            ax.scatter((nbp_solutions[i])[end_index,0],
                    (nbp_solutions[i])[end_index,1],
                    (nbp_solutions[i])[end_index,2],
                    color=COLORS[i],marker="o",s=50,label=BODY_NAMES[i])

        #Labels
        ax.set_xlabel("x",fontsize=14)
        ax.set_ylabel("y",fontsize=14)
        ax.set_zlabel("z",fontsize=14)
        ax.set_facecolor(BACKGROUND_COLOR)
        
        
        if(FIGURE_TITLE is not None and len(FIGURE_TITLE.strip("\"")) > 0):
            FIGURE_TITLE.strip("\"")
            ax.set_title(FIGURE_TITLE,fontsize=14)
        
        if(DRAW_LEGEND is not None and DRAW_LEGEND.lower() == "on"):
            ax.legend(loc="upper left",fontsize=14)

        plt.axis(DRAW_AXIS)
        plt.show()

# ==============================================================================

# MAIN
if __name__ == "__main__":

    # Create output file if it does not exist
    save_file_name = os.path.join(OUTPUT_DIR_LOCATION,OUTPUT_NAME)
    args = parser.parse_args()
    tmp = vars(args)
    
    # -------------------------------------------------
    # Parse command line args
    if tmp['no_prompt'] is not None:
        print("Use interactive menu? [y/n]")
        choice = input()
        if choice == 'y' or choice == 'Y' or choice.lower() == "yes":
            interactive_menu()


    if tmp['center_on'] is not None:
        CENTER_ON_BODY = tmp['center_on'][0]

    if tmp['save_output'] is not None:
        save_equation = tmp['save_output'][0]
        save_equation = parse_save_load_name(save_equation)

    if tmp['display_timesteps'] is not None:
        start,end = tmp['display_timesteps']

    if tmp['load_equation'] is not None:
        load_equation = tmp['load_equation'][0]
        load_equation = parse_save_load_name(load_equation)
        warning_with_no_effect = ['body_config_path','save_output']
        for element in warning_with_no_effect:
            if tmp[element] is not None:
                print(f"WARNING: setting {element} will have no effect when loading a previously solved problme.")

    if tmp['body_config_path'] is not None:
        BODY_NAMES,NUMBER_OF_BODIES,LOCATIONS,VELOCITIES,MASSES,COLORS = update_body_parameters(tmp['body_config_path'][0])    

    if tmp['config_path'] is not None:
        WINDOW_SIZE,BACKGROUND_COLOR,DRAW_AXIS,FIGURE_TITLE,DRAW_LEGEND,OUTPUT_NAME,TIME,RESOLUTION,CENTER_ON_BODY,K1,K2 = update_config_parameters(tmp['config_path'][0])

    # if tmp['body_config'] is None and tmp['center_on'] is not None:
    #     inp = input("do you want to use a preset config? [y/n]")
    #     if inp.lower() == 'y':
    #         dirs = os.listdir("bodies-presets")
    #         dir_index = directory_select_menu(dirs)
    #         BODY_NAMES,NUMBER_OF_BODIES,LOCATIONS,VELOCITIES,MASSES,COLORS = update_body_parameters(os.path.join("bodies-presets", dirs[dir_index]))

    if not os.path.exists(OUTPUT_DIR_LOCATION):
        print("No output directory found. Making new one")
        os.mkdir(OUTPUT_DIR_LOCATION)


    three_body_sol = get_nbp_solutions()

    if save_equation is not None and load_equation is None:
        with open(os.path.join(SAVED_RUNS_DIR,save_equation), 'wb') as f:
            np.save(f, three_body_sol)  

    body_solutions = []
    for index in range(NUMBER_OF_BODIES):
        body_solutions.append(three_body_sol[:,index*3:index*3+3])

    # Offset reference fraim to center on body/location?
    for index,name in enumerate(BODY_NAMES):
        if name.lower() == CENTER_ON_BODY.lower():
            body_solutions = offset_solition_by_body_location(body_solutions, index)
    
    try:
        if int(CENTER_ON_BODY) == -1:
            body_solutions = offset_solition_by_body_location(body_solutions, -1)
    except ValueError as v:
        print("Ignoring value err")
    
    draw_figure(body_solutions)
    #show_gif(save_file_name)
    print("Success")

