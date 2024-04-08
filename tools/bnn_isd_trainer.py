import numpy as np
import random

from sklearn.model_selection import train_test_split
matplotlib_style = 'fivethirtyeight'
import matplotlib.pyplot as plt; plt.style.use(matplotlib_style)

from utility.training import get_data_loader, scale_data, split_time_event, make_stratified_split
from utility.survival import make_time_bins, mtlr_survival, reformat_survival
from utility.bnn_isd_models import BayesMtlr, BayesEleCox, BayesLinCox, BayesianBaseModel, BayesEleMtlr
import math
import torch
import torch.optim as optim
import torch.nn as nn
from typing import List, Tuple, Optional, Union
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from tqdm import trange
from pycox.evaluation import EvalSurv

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
Numeric = Union[float, int, bool]
NumericArrayLike = Union[List[Numeric], Tuple[Numeric], np.ndarray, pd.Series, pd.DataFrame, torch.Tensor]

def train_bnn_model(
        model: nn.Module,
        data_train: pd.DataFrame,
        data_val: pd.DataFrame,
        time_bins: NumericArrayLike,
        config: dotdict,
        random_state: int,
        reset_model: bool = True,
        device: torch.device = torch.device("cuda")
) -> nn.Module:
    if config.verbose:
        print(f"Training {model.get_name()}: reset mode is {reset_model}, number of epochs is {config.num_epochs}, "
              f"learning rate is {config.lr}, C1 is {config.c1}, "
              f"batch size is {config.batch_size}, device is {device}.")
    #data_train, _, data_val = train_val_test_stratified_split(data_train, stratify_colname='both',
    #                                                          frac_train=0.9, frac_test=0.1,
    #                                                          random_state=random_state)
    train_size = data_train.shape[0]
    val_size = data_val.shape[0]
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    if reset_model:
        model.reset_parameters()

    model = model.to(device)
    model.train()
    best_val_nll = np.inf
    best_ep = -1

    pbar = trange(config.num_epochs, disable=not config.verbose)

    start_time = datetime.now()
    if isinstance(model, BayesEleCox) or isinstance(model, BayesLinCox):
        x_train, t_train, e_train = (torch.tensor(data_train.drop(["time", "event"], axis=1).values, dtype=torch.float),
                                     torch.tensor(data_train["time"].values, dtype=torch.float),
                                     torch.tensor(data_train["event"].values, dtype=torch.float))
        x_val, t_val, e_val = (torch.tensor(data_val.drop(["time", "event"], axis=1).values, dtype=torch.float).to(device),
                               torch.tensor(data_val["time"].values, dtype=torch.float).to(device),
                               torch.tensor(data_val["event"].values, dtype=torch.float).to(device))

        train_loader = DataLoader(TensorDataset(x_train, t_train, e_train), batch_size=train_size, shuffle=True)
        model.config.batch_size = train_size

        for i in pbar:
            total_loss = 0
            total_log_likelihood = 0
            total_kl_divergence = 0
            for xi, ti, ei in train_loader:
                xi, ti, ei = xi.to(device), ti.to(device), ei.to(device)
                optimizer.zero_grad()
                loss, log_prior, log_variational_posterior, log_likelihood = model.sample_elbo(xi, ti, ei, train_size)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_log_likelihood += log_likelihood.item()
                total_kl_divergence += log_variational_posterior.item() - log_prior.item()

            val_loss, _, _, val_log_likelihood = model.sample_elbo(x_val, t_val, e_val, dataset_size=val_size)
            pbar.set_description(f"[epoch {i + 1: 4}/{config.num_epochs}]")
            pbar.set_postfix_str(f"Train: Total = {total_loss:.4f}, "
                                 f"KL = {total_kl_divergence:.4f}, "
                                 f"nll = {total_log_likelihood:.4f}; "
                                 f"Val: Total = {val_loss.item():.4f}, "
                                 f"nll = {val_log_likelihood.item():.4f}; ")
            if config.early_stop:
                if best_val_nll > val_loss:
                    best_val_nll = val_loss
                    best_ep = i
                if (i - best_ep) > config.patience:
                    print(f"Validation loss converges at {best_ep}-th epoch.")
                    break
    elif isinstance(model, BayesEleMtlr):
        x, y = reformat_survival(data_train, time_bins)
        x_val, y_val = reformat_survival(data_val, time_bins)
        x_val, y_val = x_val.to(device), y_val.to(device)
        train_loader = DataLoader(TensorDataset(x, y), batch_size=config.batch_size, shuffle=True)
        for i in pbar:
            total_loss = 0
            total_log_likelihood = 0
            total_kl_divergence = 0
            for xi, yi in train_loader:
                xi, yi = xi.to(device), yi.to(device)
                optimizer.zero_grad()
                loss, log_prior, log_variational_posterior, log_likelihood = model.sample_elbo(xi, yi, train_size)

                loss.backward()
                optimizer.step()

                total_loss += loss.item() / train_size
                total_log_likelihood += log_likelihood.item() / train_size
                total_kl_divergence += (log_variational_posterior.item() -
                                        log_prior.item()) * config.batch_size / train_size**2

            val_loss, _, _, val_log_likelihood = model.sample_elbo(x_val, y_val, dataset_size=val_size)
            val_loss /= val_size
            val_log_likelihood /= val_size
            pbar.set_description(f"[epoch {i + 1: 4}/{config.num_epochs}]")
            pbar.set_postfix_str(f"Train: Total = {total_loss:.4f}, "
                                 f"KL = {total_kl_divergence:.4f}, "
                                 f"nll = {total_log_likelihood:.4f}; "
                                 f"Val: Total = {val_loss.item():.4f}, "
                                 f"nll = {val_log_likelihood.item():.4f}; ")
            if config.early_stop:
                if best_val_nll > val_loss:
                    best_val_nll = val_loss
                    best_ep = i
                if (i - best_ep) > config.patience:
                    print(f"Validation loss converges at {best_ep}-th epoch.")
                    break            
    
    else:
        raise TypeError("Model type cannot be identified.")
    end_time = datetime.now()
    training_time = end_time - start_time
    print(f"Training time: {training_time.total_seconds()}")
    # model.eval()
    if isinstance(model, BayesEleCox) or isinstance(model, BayesLinCox):
        model.calculate_baseline_survival(x_train.to(device), t_train.to(device), e_train.to(device))
    return model