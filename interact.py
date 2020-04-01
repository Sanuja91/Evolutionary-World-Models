import gym, torch, cv2, os
import numpy as np
from pyglet.window import key
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.vae import CNN_VAE, train_vae
from models.mdn import MDN_RNN, train_mdn


def create_vae_training_data(frames):
    """Creates training images for CNN-VAE"""
    env = gym.make('CarRacing-v0')
    obs = env.reset()
    action = np.array([0.0, 0.0, 0.0])

    def key_press(k, mod):
        global restart
        if k == 0xff0d: restart = True
        if k == key.LEFT:  action[0] = -1.0
        if k == key.RIGHT: action[0] = +1.0
        if k == key.UP:    action[1] = +1.0
        if k == key.DOWN:  action[2] = +0.8   # set 1.0 for wheels to block to zero rotation
    def key_release(k, mod):
        if k == key.LEFT  and action[0] == -1.0: action[0] = 0
        if k == key.RIGHT and action[0] == +1.0: action[0] = 0
        if k == key.UP:    action[1] = 0
        if k == key.DOWN:  action[2] = 0

    path = f'data\\inputs\\images'
    if not os.path.exists(path):
        os.makedirs(path)
    
    action_path = f'data\\inputs\\tensors'
    if not os.path.exists(action_path):
        os.makedirs(action_path) 
    
    actions = []

    for count in range(frames):
        env.render()
        env.viewer.window.on_key_press = key_press
        env.viewer.window.on_key_release = key_release
        cv2.imwrite(f'{path}\\{count}.png', obs)
        obs, reward, done, info = env.step(action)
        actions.append(action)
    
    torch.save(torch.tensor(actions), f'{action_path}\\actions.pt')
    env.close()

def create_rnn_training_data(name):
    """Create training data for MDN-RNN"""
    model = CNN_VAE(name, None, 'Latest')
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
    zs = []
    for batch_idx, (inputs, _) in enumerate(train_loader):
        mu, logvar = model.encode(inputs)
        z = model.reparameterize(mu, logvar)
        mus.append(mu)
        logvars.append(logvar)
        zs.append(z)

    mus = torch.cat(mus)
    logvars = torch.cat(logvars)
    zs = torch.cat(zs)

    torch.save(mus, f'{path}\\mus.pt')
    torch.save(logvars, f'{path}\\logvars.pt')
    torch.save(zs.squeeze(-1), f'{path}\\zs.pt')

def train_vae_(name):
    """Trains the CNN-VAE"""
    params = {
        'z_size' : 128,
        'batch_size' : 64,
        'learning_rate' : 0.0001,
        'kl_tolerance' : 0.5,
        'batch_norm' : True
    }

    model = CNN_VAE(name, params, False)
    train_vae(model, 500, 100)

def train_mdn_(name):
    """Trains the MDN-RNN"""
    params = {
        'z_size' : 128,
        'action_size' : 3,
        'n_hidden' : 256,
        'learning_rate' : 0.0001,
        'n_gaussians' : 5,
        'n_layers' : 1,
        'seq_len' : 50,
        'grad_clip' : 1.0,
        'batch_size' : 200,
        'batch_norm' : True
    }

    model = MDN_RNN(name, params, False)
    train_mdn(model, 300, 10)

# create_vae_training_data(3000)
# train_vae_('Alpha - BN SMALL')
# create_rnn_training_data('Alpha - BN SMALL')
# train_mdn_('Alpha - TEST')
