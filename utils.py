
import pyglet
from constants import *
import configparser
from ast import literal_eval

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

