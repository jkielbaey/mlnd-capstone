from __future__ import print_function # future proof
import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from io import StringIO
from six import BytesIO

CONTENT_TYPE_CSV = 'text/csv'
CONTENT_TYPE_NPY = 'application/x-npy'

# import model
from model import Net

scalers = {}

def model_fn(model_dir):
    print("Loading model.")
    
    global scalers

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(model_info['input_dim'], 
                model_info['hidden_dim'])
    
    scalers = model_info['scalers']

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))
    
    return model.to(device)

def apply_scaling(df):
    ['lepton_phi', 'missing_energy_phi', 'jet_3_phi', 'jet_4_phi']
    column_names = [
        'lepton_pt', 'lepton_eta', 'missing_energy_magnitude', 
        'jet_1_pt', 'jet_1_eta', 'jet_1_phi', 'jet_1_b-tag',
        'jet_2_pt', 'jet_2_eta', 'jet_2_phi', 'jet_2_b-tag',
        'jet_3_pt', 'jet_3_eta', 'jet_3_b-tag',
        'jet_4_pt', 'jet_4_eta', 'jet_4_b-tag',
    ]
    df.columns = column_names
    
    # Apply a PowerTransformer on the f_dist_columns in order to make convert them more into a normal distribution.
    f_dist_columns = ['lepton_pt', 'jet_1_pt', 'jet_2_pt', 'jet_3_pt', 'jet_4_pt', 'missing_energy_magnitude']
    power_scaler = scalers['power']
    df[f_dist_columns] = pd.DataFrame(power_scaler.transform(df[f_dist_columns]))

    # Apply MinMaxScaler on all features to align the range of the different features.
    min_max_scaler = scalers['minmax']
    df = pd.DataFrame(min_max_scaler.transform(df))

    return df

def input_fn(serialized_input_data, content_type):
    print('Deserializing the input data: {}'.format(str(serialized_input_data)[:200]))
    if content_type == CONTENT_TYPE_NPY:
        stream = BytesIO(serialized_input_data)
        data = np.load(stream)
        print("Numpy {}: {}".format(type(data), str(data)[:200]))
        return data
    elif content_type == CONTENT_TYPE_CSV:
        print("CSV input {}: {}".format(type(serialized_input_data), str(serialized_input_data)[:200]))
        data = pd.read_csv(StringIO(serialized_input_data), header=None)
        #print("CSV before scaling {}: {}".format(type(data), str(data)[:200]))
        data = apply_scaling(data).values
        print("CSV for prediction {}: {}".format(type(data), np.array2string(data)[:200]))
        return data
    
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)

def output_fn(prediction_output, accept):
    print('Serializing the generated output: {} {} to format {}'.format(type(prediction_output), str(prediction_output)[:200], accept))
    if accept == CONTENT_TYPE_NPY:
        buffer = BytesIO()
        np.save(buffer, prediction_output)
        return buffer.getvalue(), accept
    elif accept == CONTENT_TYPE_CSV:
        s = StringIO()
        np.savetxt(s, prediction_output, fmt='%d')
        return s.getvalue(), accept
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)

def predict_fn(input_data, model):
    print('Predicting class labels for the input data...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Process input_data so that it is ready to be sent to our model
    # convert data to numpy array then to Tensor
    data = torch.from_numpy(input_data.astype('float32'))
    data = data.to(device)

    # Put model into evaluation mode
    model.eval()

    # Compute the result of applying the model to the input data.
    out = model(data)
    
    # The variable `result` should be a numpy array; a single value 0-1
    result = np.round(out.cpu().detach().numpy())

    print("Predictions: {}".format(list(result)))

    return result