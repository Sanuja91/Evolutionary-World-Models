from sklearn.preprocessing import minmax_scale, LabelEncoder, OneHotEncoder
import numpy as np
import pandas as pd
import os

def one_hot_encode_actions(actions):
    """One hot encodes given actions and returns labelled dictionaries"""
    actions = np.array([''.join(str(action)) for action in actions])

    # create one hot encoding
    enc = OneHotEncoder(sparse = False)
    encoded = pd.DataFrame(enc.fit_transform(actions.reshape(len(actions), 1)))
    
    # create dicts of encodings and actions
    actions = dict(zip(encoded.index.values, actions))
    encoded = dict(zip(encoded.index.values, encoded.values))
    
    return actions, encoded

def create_path(path):
    """Creates path string depending on OS"""
    if os.name != 'nt':
        path = path.replace('\\', '/')
    return path