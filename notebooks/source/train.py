from __future__ import print_function # future proof
import argparse
import boto3
import sys
import os
import json
import re

import numpy as np
import pandas as pd
import joblib

# pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

# import model
from model import Net

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


# Load the data from a csv file
def _get_data_loader(batch_size, data_dir, data_file):
    print("Get data loader for file {}.".format(data_file))

    data = pd.read_csv(os.path.join(data_dir, data_file), header=None, names=None)
    y = torch.from_numpy(data[[0]].values).float().squeeze()
    x = torch.from_numpy(data.drop([0], axis=1).values).float()
    ds = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    
def train_epoch(model, device, train_loader, optimizer, criterion, epoch):
    """ Perform training for a single epoch.
    
    :param model: PyTorch model to train
    :param device: Where the model and data should be loaded (cpu or gpu)
    :param train_loader: PyTorch DataLoader used during training
    :param optimizer: The optimizer used during training
    :param criterion: The loss function used for training
    :param epoch: Epoch that the model is being trained for
    """
    model.train()
    model.to(device)
    print("  Train: model on {}".format(device))
    
    
    total_loss = 0

    for batch_idx, (batch_X, batch_y) in enumerate(train_loader, 1):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        output = model(batch_X).view(-1)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.data.item()
        
    avg_loss = total_loss/len(train_loader)
    print('  Train: Average loss: {:.4f}'.format(avg_loss))
    
    return avg_loss

def test_model(model, device, test_loader, criterion):
    model.eval()
    model.to(device)
    print("  Validation: model on {}".format(device))

    test_losses = []
    total_loss = 0

    with torch.no_grad():
        for batch_idx, (batch_X, batch_y) in enumerate(test_loader, 1):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            output = model(batch_X).view(-1)

            test_loss = criterion(output, batch_y)
            total_loss += test_loss.data.item()

    avg_test_loss = total_loss / len(test_loader)
    print('  Validation: Average loss: {:.4f}'.format(avg_test_loss))

    return avg_test_loss

def train_model(model, device, train_loader, test_loader, optimizer, criterion, epochs, model_dir):
    best_val_loss = None
    best_val_loss_epoch = -1
    best_train_loss = None
    best_train_loss_epoch = -1
    for epoch in range(1, epochs + 1):
        print("Epoch {}".format(epoch))
        train_loss = train_epoch(model, device, train_loader, optimizer, criterion, epoch)
        val_loss = test_model(model, device, test_loader, criterion)

        if best_train_loss is None or train_loss < best_train_loss:
            best_train_loss = train_loss
            best_train_loss_epoch = epoch
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_loss_epoch = epoch
            save_model(model, model_dir)
            print("  Model performed better on validation...Saved.")
            
        if val_loss > best_val_loss and epoch - best_val_loss_epoch >= 10:
            print("  Validation loss didn't improve over last 10 epochs. Stopping")
            break

    print("Best training loss: {:.4f} in epoch {}".format(best_train_loss, best_train_loss_epoch))
    print("Best validation loss: {:.4f} in epoch {}".format(best_val_loss, best_val_loss_epoch))


# Provided model saving functions
def save_model(model, model_dir):
    path = os.path.join(model_dir, 'model.pth')

    # save state dictionary
    torch.save(model.cpu().state_dict(), path)


def save_model_params(model_info, model_dir):
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    print("Saving model params in {}".format(model_info_path))
    with open(model_info_path, 'wb') as f:
        torch.save(model_info, f)


if __name__ == '__main__':
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--validation-dir', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument('--scalers-config', type=str, default="")
    
    # Training Parameters
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--optimizer', type=str, default='adam', metavar='OPTIM',
                        help='optimizer (default: adam)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0, metavar='M',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--weigth-init', type=str, default='none', metavar='S',
                        help='weight init method (default: pytorch default)')
  
    # Model parameters
    parser.add_argument('--input-dim', type=int, default=2, metavar='IN',
                        help='input batch size for training (default: 2)')
    parser.add_argument('--hidden-dim', type=int, default=50, metavar='H',
                        help='input batch size for training (default: 50)')
    parser.add_argument('--dropout-rate', type=float, default=0.2, metavar='DR',
                        help='dropout rate (default: 0.2)')
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Model will be trained on {}".format(device))
    
    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        
    m = re.match('s3://([^/]+)/(.+)$', args.scalers_config)
    if m is None:
        print("Scaler config param invalid ({}).".format(args.scalers_config))
        sys.exit(1)

    bucket = m.group(1)
    key = m.group(2)

    boto3.resource('s3').Bucket(bucket).download_file(key, 'scaler.pkl')
    with open('scaler.pkl', 'rb') as f:
        scalers = joblib.load(f)
        
    # get train loader
    train_loader = _get_data_loader(args.batch_size, args.train_dir, 'train.csv')
    test_loader = _get_data_loader(args.batch_size, args.validation_dir, 'val.csv')
    
    model = Net(args.input_dim, args.hidden_dim, init_weights=args.weigth_init, dropout_rate=args.dropout_rate).to(device)
    
    print("Show model:")
    print(model)

    model_info = {
        'input_dim': args.input_dim,
        'hidden_dim': args.hidden_dim,
        'dropout_rate': args.dropout_rate,
        'scalers': scalers
    }
    save_model_params(model_info, args.model_dir)

    optimzers = {
        'adam': optim.Adam(model.parameters(), lr=args.lr),
        'sgd': optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    }
    optimizer = optimzers[args.optimizer]
    criterion = nn.BCELoss()
        
    train_model(model, device, train_loader, test_loader, optimizer, criterion, args.epochs, args.model_dir)