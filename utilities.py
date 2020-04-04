from sklearn.preprocessing import minmax_scale, LabelEncoder, OneHotEncoder
import numpy as np
import pandas as pd

def one_hot_encode_actions(actions):
    actions = [''.join(str(action)) for action in actions]
    actions = pd.DataFrame(actions, columns = ['actions'])

    # creating instance of labelencoder
    labelencoder = LabelEncoder()
    actions['labels'] = labelencoder.fit_transform(actions['actions'])

    # creating instance of one-hot-encoder
    enc = OneHotEncoder(handle_unknown = 'ignore')
    encoded = pd.DataFrame(enc.fit_transform(actions).toarray())
    encoded = dict(zip(encoded.index.values, encoded.values))
    actions.set_index('labels', inplace = True)
    actions = dict(zip(actions.index.values, np.ndarray.flatten(actions.values)))
    return actions, encoded