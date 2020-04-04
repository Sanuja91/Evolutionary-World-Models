import gym, torch, cv2, os
import numpy as np
from pyglet.window import key
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.vae import CNN_VAE, train_vae
from models.mdn import MDN_RNN, train_mdn
from models.controller import Controller, evolve_controller
from evolve.base import Simple_Asexual_Evolution

Z_SIZE = 128
ACTION_SIZE = 3

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

def create_mdn_training_data(name):
    """Create training data for MDN-RNN"""
    vae = CNN_VAE(name, None, 'Latest')
    vae.eval()
    train_loader = DataLoader(
        datasets.ImageFolder(
            'data\\inputs', 
            transform = transforms.ToTensor()),
        batch_size = vae.batch_size, 
        shuffle = False
    )

    path = f'data\\inputs\\tensors'
    if not os.path.exists(path):
        os.makedirs(path)

    mus = []
    logvars = []
    zs = []
    for batch_idx, (inputs, _) in enumerate(train_loader):
        mu, logvar = vae.encode(inputs)
        z = vae.reparameterize(mu, logvar)
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
        'z_size' : Z_SIZE,
        'batch_size' : 64,
        'learning_rate' : 0.0001,
        'kl_tolerance' : 0.5,
        'batch_norm' : True
    }

    vae = CNN_VAE(name, params, False)
    train_vae(vae, 500, 100)

def train_mdn_(name):
    """Trains the MDN-RNN"""
    params = {
        'z_size' : Z_SIZE,
        'action_size' : ACTION_SIZE,
        'n_hidden' : 256,
        'learning_rate' : 0.0001,
        'n_gaussians' : 5,
        'n_layers' : 1,
        'seq_len' : 50,
        'grad_clip' : 1.0,
        'batch_size' : 200,
        'batch_norm' : True
    }

    mdn = MDN_RNN(name, params, False)
    train_mdn(mdn, 300, 10)

def train_controller_(name, vae_name, mdn_name):
    """Trains / Evolves the controller"""
    params = {
        'z_size' : Z_SIZE,
        'action_size' : ACTION_SIZE,
        'n_hidden' : 256,
        'batch_norm' : True
    }

    env = gym.make('CarRacing-v0')
    vae = CNN_VAE(vae_name, None, 'Latest')
    mdn = MDN_RNN(mdn_name, None, 'Latest')
    controller = Controller(name, params, False)    

    params = {
        'num_agents' : 500, # number of random agents to create
        'runs' : 3, # number of runs to evaluate fitness score
        'timesteps': 1000, # timesteps to run in environment to get cumulative reward
        'top_parents' : 20, # number of parents from agents
        'generations' : 1000, # run evolution for X generations
        'mutation_power' : 0.2, # strength of mutation, set from https://arxiv.org/pdf/1712.06567.pdf
        'vae' : vae,
        'mdn' : mdn
    }

    evolution = Simple_Asexual_Evolution(env, controller, interact, params)

    evolution.evolve()

    # evolve_controller(model, vae, mdn, evolve, 300, 10)


def interact(agent, env, params):
    """Runs the agents in the given environment and collects rewards"""    
    timesteps = params['timesteps']
    vae = params['vae']
    mdn = params['mdn']
    
    hidden = mdn.init_hidden(1)
    vae.eval()
    mdn.eval()
    agent.eval()
    observation = env.reset()
    cumulative_rewards = 0
    s = 0
    for _ in range(timesteps):
        print(observation.shape)
        observation = transforms.ToTensor()(observation.copy()).unsqueeze(0)
        
        mu, logvar = vae.encode(observation)
        z = vae.reparameterize(mu, logvar)
        action = agent(z, hidden[0])
        _, hidden = mdn(z, hidden)

        print(action.shape)
        exit()

        output_probabilities = agent(inp).detach().numpy()[0]
        action = np.random.choice(range(game_actions), 1, p = output_probabilities).item()
        observation, reward, done, info = env.step(action)
        cumulative_rewards += reward
        s = s + 1
        if done:
            break
    return cumulative_rewards


# create_vae_training_data(3000)
# train_vae_('Alpha - BN SMALL')
# create_mdn_training_data('Alpha - BN SMALL')
# train_mdn_('Alpha - TEST')
train_controller_('Alpha - TEST', 'Alpha - BN SMALL', 'Alpha - TEST')
