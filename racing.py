import gym, torch, cv2, os, time, concurrent
import numpy as np
from pyglet.window import key
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.vae import CNN_VAE, train_vae
from models.mdn import MDN_RNN, train_mdn
from models.controller import Controller
from evolve.simple import Simple_Asexual_Evolution, Elite_Evolution, Natural_Evolution
from utilities import one_hot_encode_actions

Z_SIZE = 128

ACTIONS = [
    np.array([0.0, 0.0, 0.0]),       # do nothing
    np.array([0.0, 1.0, 0.0]),       # forward
    np.array([0.0, 0.0, 1.0]),       # stop
    np.array([1.0, 0.0, 0.0]),       # right
    np.array([-1.0, 0.0, 0.0]),      # left
    np.array([1.0, 1.0, 0.0]),       # forward right
    np.array([-1.0, 1.0, 0.0]),      # forward left
    np.array([1.0, 0.0, 1.0]),       # stop right
    np.array([-1.0, 0.0, 1.0]),      # stop left
    np.array([0.0, 1.0, 1.0]),       # forward stop
    np.array([-1.0, 1.0, 1.0]),      # forward stop left
    np.array([1.0, 1.0, 1.0])        # forward stop right
]

ACTION_LABELS, ENCODED_ACTIONS = one_hot_encode_actions(ACTIONS)
ACTION_SIZE = len(ENCODED_ACTIONS[0])

def create_vae_training_data(frames):
    """Creates training images for CNN-VAE"""
    env = gym.make('CarRacing-v0')
    obs = env.reset()
    action = np.array([0.0, 0.0, 0.0])

    def key_press(k, mod):
        global restart
        if k == 0xff0d: restart = True
        if k == key.LEFT:  action[0] = - 1.0
        if k == key.RIGHT: action[0] = + 1.0
        if k == key.UP:    action[1] = + 1.0
        if k == key.DOWN:  action[2] = + 1.0   

    def key_release(k, mod):
        if k == key.LEFT  and action[0] == - 1.0: action[0] = 0.0
        if k == key.RIGHT and action[0] == + 1.0: action[0] = 0.0
        if k == key.UP:    action[1] = 0.0
        if k == key.DOWN:  action[2] = 0.0

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
        action_label = [''.join(str(action_)) for action_ in action]

        for idx in range(len(ACTION_LABELS)):
            if str(action) == ACTION_LABELS[idx]:
                actions.append(ENCODED_ACTIONS[idx])
                break
            if idx == len(ACTION_LABELS) - 1:
                raise Exception(f'All actions not encoded!! Failed Action = {action_label}')
    
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
        'hidden_size' : 256,
        'learning_rate' : 0.0001,
        'gaussian_size' : 5,
        'stacked_layers' : 1,
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
        'hidden_size' : 256,
        'batch_norm' : True
    }

    vae = CNN_VAE(vae_name, None, 'Latest')
    mdn = MDN_RNN(mdn_name, None, 'Latest')
    controller = Controller(name, params, False)  
    vae.to(vae.device)
    mdn.to(mdn.device)  
    controller.to(controller.device)

    params = {
        'render': False,
        'gym': 'CarRacing-v0',
        'num_agents' : 10, # number of random agents to create
        'runs' : 1, # number of runs to evaluate fitness score
        'timesteps': 1000, # timesteps to run in environment to get cumulative reward.. Random actions till LSTM starts working
        'top_parents' : 3, # number of parents from agents
        'generations' : 1000, # run evolution for X generations
        'mutation_power' : 0.2, # strength of mutation, set from https://arxiv.org/pdf/1712.06567.pdf
        'gene_splits': [0.5, 0.7, 0.8],
        'vae' : vae,
        'mdn' : mdn
    }

    evolution = Natural_Evolution(controller, interact, params)

    evolution.evolve()

    # evolve_controller(model, vae, mdn, evolve, 300, 10)


def interact(agent, env, params):
    """Runs the agents in the given environment and collects rewards"""    
    timesteps = params['timesteps']
    vae = params['vae']
    mdn = params['mdn']
    render = params['render']
    mdn.batch_size = 1

    hidden = mdn.init_hidden(mdn.batch_size)
    vae.eval()
    mdn.eval()
    agent.eval()
    observation = env.reset()
    cumulative_rewards = 0
    s = 0
    za = None
    
    for timestep in range(timesteps):
        if render:
            env.render()
        hidden = (hidden[0].detach(), hidden[1].detach())
        observation = transforms.ToTensor()(observation.copy()).unsqueeze(0).to(agent.device)
        
        mu, logvar = vae.encode(observation)
        z = vae.reparameterize(mu, logvar).squeeze(-1)
        action_probabilities = agent(z, hidden[0].squeeze(0)).squeeze(0).detach().cpu().numpy()

        action_idx = np.random.choice(list(ENCODED_ACTIONS.keys()), 1, p = action_probabilities)[0]
        encoded_action = torch.tensor(ENCODED_ACTIONS[action_idx]).float().to(agent.device).unsqueeze(0)
        action = ACTIONS[action_idx]

        za_ = torch.cat((z, encoded_action), dim = 1).unsqueeze(1)

        if za is None:
            za = za_
        else:
            za = torch.cat((za, za_), dim = 1)
            if za.shape[1] > mdn.seq_len:
                # truncates old data if sequence length is too long
                (za, _) = torch.split(za, mdn.seq_len, dim = 1) 
        
        _, hidden = mdn(za, hidden)

        observation, reward, done, info = env.step(action)
        cumulative_rewards += reward
        s = s + 1
        if done:
            break
    print("CUMULATIVE REWARD", cumulative_rewards)
    return cumulative_rewards


# create_vae_training_data(3000)
# train_vae_('Alpha - BN SMALL')
# create_mdn_training_data('Alpha - BN SMALL')
# train_mdn_('Alpha - TEST')
train_controller_('Alpha - TEST', 'Alpha - BN SMALL', 'Alpha - TEST')

# import ray
# def create_env():
#     return gym.make('CarRacing-v0')

# @ray.remote
# def run_env(id, steps):
#     env = gym.make('CarRacing-v0')
#     env.reset()
#     for step in range(steps):
#         action = env.action_space.sample()
#         env.step(action)
#         print(f'ENV {id} STEP {step}')
#     return id


# ray.init()
# results = []
# for id in range(20):
#     results.append(run_env.remote(id, 1000))
# ray.get(results)
