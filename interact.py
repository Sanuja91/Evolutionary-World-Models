import gym, torch, cv2 
import numpy as np
from pyglet.window import key
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.vae import ConvVAE, train_vae


def create_vae_training_data(frames):
    """Creates training images for CNN-VAE"""
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

    path = f'data\\inputs\\images'
    if not os.path.exists(path):
        os.makedirs(path)

    for _ in range(frames):
        env.render()
        env.viewer.window.on_key_press = key_press
        env.viewer.window.on_key_release = key_release
        cv2.imwrite(f'{path}\\{count}.png', obs)
        obs, reward, done, info = env.step(a)
        count += 1
    env.close()

def create_rnn_training_data(name):
    """Create training data for MDN-RNN"""
    model = ConvVAE(name, None, 'Latest')
    model.eval()
    train_loader = DataLoader(
        datasets.ImageFolder(
            'data\\inputs', 
            transform = transforms.ToTensor()),
        batch_size = model.batch_size, 
        shuffle = False
    )

    path = f'data\\inputs\\tensors'
    if not os.path.exists(path):
        os.makedirs(path)

    mus = []
    logvars = []
    for batch_idx, (inputs, _) in enumerate(train_loader):
        mu, logvar = model.encode(inputs)
        mus.append(mu)
        logvars.append(logvar)
        if batch_idx > 3:
            break
    mus = torch.cat(mus)
    logvars = torch.cat(logvars)

    torch.save(mus, f'{path}\\mus.pt')
    torch.save(logvars, f'{path}\\logvars.pt')

def train_vae_(name):
    """Trains the CNN-VAE"""
    params = {
        'z_size' : 128,
        'batch_size' : 64,
        'learning_rate' : 0.0001,
        'kl_tolerance' : 0.5,
        'batch_norm' : True
    }

    model = ConvVAE(name, params, False)
    train_vae(model, 100, 500)


# create_vae_training_data(100)
create_rnn_training_data('Alpha - BN')
# train_vae_('Alpha - TEST')
