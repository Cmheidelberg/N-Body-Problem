
import pyglet
from constants import *
from ast import literal_eval
import time
import sys

def directory_select_menu(directories):


    menu_title = "Please select a file"

    slow_print_text(menu_title)
    
    for i,d in enumerate(directories):
        print(f"{i+1}. {d}")
        time.sleep(0.15)
    
    while True:
        selection = input("Selection: ")

        if selection.isdecimal():
            num = int(selection)
            if num-1 >= 0 and num-1 < len(directories):
                return num-1
        else:
            for i in range(len(directories)):
                if selection.lower() == directories[i].lower():
                    return i
        
        print("invalid selection.\n")


def simple_menu_print(menu_title, options_list, help_text=""):
    
    has_help_text = len(help_text) > 0
    while(True):
        
        # Print menu
        slow_print_text(menu_title)

        i = 0
        for i in range(len(options_list)):
            print(f"{i+1}. {options_list[i]}")
            time.sleep(0.15)
        
        if(has_help_text):
            print(f"{i+2}. Help")

        inp = input()
        if(inp.isnumeric()):
            inp = int(inp)
            if(has_help_text and inp == len(options_list)+1):
                slow_print_text(help_text + "\n")
            elif(inp > 0 and inp <= len(options_list)):
                return inp
            else:
                print("Invalid selection\n")
        else:
            print("Input must be a number\n")


def slow_print_text(text):
    ''' Print the text to the screen with a slight pause between each character.'''
    
    # Modify delay time by length of message
    delay_time = 1/(2*len(text))
    
    # Print multiple characters at once for messages that require delay times shorter than thread.sleep supports
    sleep_mod_value = int(0.01/delay_time)
    
    count = 0
    for c in text:
        count+=1
        sys.stdout.write(c)
        sys.stdout.flush()
        
        # for long messages pring multiple characters at once to speed up
        if(sleep_mod_value == 0 or count % sleep_mod_value == 0):
            time.sleep(delay_time)
        
    print("")


def take_user_input(prompt):
    slow_print_text(prompt)
    return input()

def get_int_input(prompt,low=None,high=None):
    '''
    Ask user to input a number. Repeat prompt until user gives a valid input. Optional low and high bounds
    Specify a min and max accepted range
    
    :param prompt: Message to ask the user before taking their input
    :type prompt: string

    :param low: lowest accepted number (inclusive)
    :type low: int,None

    :param high: highest accepted number (inclusive)
    :type high: int,None

    :return input_num: Number input by user
    '''

    while(True):
        print(prompt)
        input_num = input()

        if input_num.isnumeric:
            try:
                input_num = int(input_num)
                if (low is None or input_num >= low) and (high is None or input_num <= high):
                    return input_num
                else:
                    print(f"Input number must be between the range of {low} and {high}")
            except:
                print("Invalid value. Please enter an integer")

        else:
            print("Input must be a number.")


def get_float_input(prompt,low=None,high=None):
    '''
    Ask user to input a number. Repeat prompt until user gives a valid input. Optional low and high bounds
    Specify a min and max accepted range
    
    :param prompt: Message to ask the user before taking their input
    :type prompt: string

    :param low: lowest accepted number (inclusive)
    :type low: int,None

    :param high: highest accepted number (inclusive)
    :type high: int,None

    :return input_num: Number input by user
    '''

    while(True):
        print(prompt)
        input_num = input()

        if input_num.isnumeric:
            try:
                input_num = float(input_num)
                if (low is None or input_num >= low) and (high is None or input_num <= high):
                    return input_num
                else:
                    print(f"Input number must be between the range of {low} and {high}")
            except:
                print("Invalid value. Please enter a float")

        else:
            print("Input must be a number.")
        

def show_gif(filename):
    if os.path.exists(filename): 
        animation = pyglet.image.load_animation(filename)
        animSprite = pyglet.sprite.Sprite(animation)

        w = animSprite.width
        h = animSprite.height

        window = pyglet.window.Window(width=w, height=h)

        r,g,b,alpha = 0.5,0.5,0.8,0.5

        pyglet.gl.glClearColor(r,g,b,alpha)

        @window.event
        def on_draw():
            window.clear()
            animSprite.draw()

        pyglet.app.run()
    else:
        print(f"Error showing gif: \"{filename}\" does not exist.")
        print("Aborting")
        exit(1)

