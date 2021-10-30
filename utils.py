
import pyglet
from constants import *
from ast import literal_eval

def directory_select_menu(directories):


    print("Please select a configuration file")
    for i,d in enumerate(directories):
        print(f"{i+1}. {d}")
    
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

