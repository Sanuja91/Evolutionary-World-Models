import gym 
import cv2
import numpy as np
from pyglet.window import key

from models.vae import ConvVAE

def create_images(frames):
    """Creates training images for VAE"""
    env = gym.make('CarRacing-v0')
    obs = env.reset()
    count = 0
    a = np.array([0.0, 0.0, 0.0])

    def key_press(k, mod):
        global restart
        if k == 0xff0d: restart = True
        if k == key.LEFT:  a[0] = -1.0
        if k == key.RIGHT: a[0] = +1.0
        if k == key.UP:    a[1] = +1.0
        if k == key.DOWN:  a[2] = +0.8   # set 1.0 for wheels to block to zero rotation
    def key_release(k, mod):
        if k == key.LEFT  and a[0] == -1.0: a[0] = 0
        if k == key.RIGHT and a[0] == +1.0: a[0] = 0
        if k == key.UP:    a[1] = 0
        if k == key.DOWN:  a[2] = 0

    for _ in range(frames):
        env.render()
        env.viewer.window.on_key_press = key_press
        env.viewer.window.on_key_release = key_release
        cv2.imwrite(f'images/{count}.png', obs)
        obs, reward, done, info = env.step(a)
        count += 1
    env.close()
