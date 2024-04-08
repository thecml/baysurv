"""
tune_mlp_model.py
====================================
Tuning script for mlp model
--dataset: Dataset name, one of "SUPPORT", "NHANES", "GBSG2", "WHAS500", "FLCHAIN", "METABRIC"
"""

import numpy as np
import os
import tensorflow as tf
from tools.baysurv_builder import make_mlp_model
from utility.risk import InputFunction
from utility.loss import CoxPHLoss
from tools import baysurv_trainer, data_loader
import os
from utility.tuning import get_mlp_sweep_config
import argparse
import numpy as np
import os
import argparse
from tools import data_loader
import pandas as pd
from utility.training import split_time_event
from utility.survival import calculate_event_times, compute_deterministic_survival_curve
import config as cfg
from utility.training import make_stratified_split
from utility.survival import convert_to_structured
from utility.training import make_stratified_split, scale_data
from pycox.evaluation import EvalSurv

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

os.environ["WANDB_SILENT"] = "true"
import wandb

N_RUNS = 100
PROJECT_NAME = "baysurv_bo_mlp"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        required=True,
                        default=None)
    args = parser.parse_args()
    global dataset_name
    if args.dataset:
        dataset_name = args.dataset
    
    sweep_config = get_mlp_sweep_config()
    sweep_id = wandb.sweep(sweep_config, project=PROJECT_NAME)
    wandb.agent(sweep_id, train_model, count=N_RUNS)

def train_model():
    config_defaults = cfg.MLP_DEFAULT_PARAMS

    # Initialize a new wandb run
    wandb.init(config=config_defaults, group=dataset_name)
    config = wandb.config
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    early_stop = config['early_stop']
    patience = config['patience']
    l2_reg = config['l2_reg']
    n_samples_train = config['n_samples_train']
    n_samples_valid = config['n_samples_valid']
    n_samples_test = config['n_samples_test']

    # Load data
    if dataset_name == "SUPPORT":
        dl = data_loader.SupportDataLoader().load_data()
    elif dataset_name == "GBSG2":
        dl = data_loader.GbsgDataLoader().load_data()
    elif dataset_name == "WHAS500":
        dl = data_loader.WhasDataLoader().load_data()
    elif dataset_name == "FLCHAIN":
        dl = data_loader.FlchainDataLoader().load_data()
    elif dataset_name == "METABRIC":
        dl = data_loader.MetabricDataLoader().load_data()
    elif dataset_name == "SEER":
        dl = data_loader.SeerDataLoader().load_data()
    elif dataset_name == "MIMIC":
        dl = data_loader.MimicDataLoader().load_data()
    else:
        raise ValueError("Dataset not found")

    num_features, cat_features = dl.get_features()
    df = dl.get_data()
    
    # Split data
    df_train, df_valid, df_test = make_stratified_split(df, stratify_colname='both', frac_train=0.7,
                                                        frac_valid=0.1, frac_test=0.2, random_state=0)
    X_train = df_train[cat_features+num_features]
    X_valid = df_valid[cat_features+num_features]
    X_test = df_test[cat_features+num_features]
    y_train = convert_to_structured(df_train["time"], df_train["event"])
    y_valid = convert_to_structured(df_valid["time"], df_valid["event"])
    y_test = convert_to_structured(df_test["time"], df_test["event"])
    
    # Scale data
    X_train, X_valid, X_test = scale_data(X_train, X_valid, X_test, cat_features, num_features)
    
    # Convert to array
    X_train = np.array(X_train)
    X_valid = np.array(X_valid)

    # Make time/event split
    t_train, e_train = split_time_event(y_train)
    t_valid, e_valid = split_time_event(y_valid)

    # Calculate event times
    event_times = calculate_event_times(t_train, e_train)
    
    # Make datasets
    train_ds = InputFunction(X_train, t_train, e_train, batch_size=batch_size,
                             drop_last=True, shuffle=True)()
    valid_ds = InputFunction(X_valid, t_valid, e_valid, batch_size=batch_size)()
    
    model_name = "MLP"
    model = make_mlp_model(input_shape=X_train.shape[1:],
                           output_dim=1,
                           layers=config['network_layers'],
                           activation_fn=config['activation_fn'],
                           dropout_rate=config['dropout'],
                           regularization_pen=l2_reg)
    
    # Define optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=wandb.config.learning_rate,
                                            weight_decay=wandb.config.weight_decay)
    
    # Train model
    trainer = baysurv_trainer.Trainer(model=model,
                                      model_name=model_name,
                                      train_dataset=train_ds,
                                      valid_dataset=valid_ds,
                                      test_dataset=None,
                                      optimizer=optimizer,
                                      loss_function=CoxPHLoss(),
                                      num_epochs=num_epochs,
                                      early_stop=early_stop,
                                      patience=patience,
                                      n_samples_train=n_samples_train,
                                      n_samples_valid=n_samples_valid,
                                      n_samples_test=n_samples_test)
    trainer.train_and_evaluate()

    # Compute survival function
    surv_preds = pd.DataFrame(compute_deterministic_survival_curve(
        model, X_train, X_valid, e_train, t_train, event_times, model_name), columns=event_times)
    
    # Compute CI
    try:
        ev = EvalSurv(surv_preds.T, t_valid, e_valid, censor_surv="km")
        ci = ev.concordance_td()
    except:
        ci = np.nan

    # Log to wandb
    wandb.log({"val_ci": ci})

if __name__ == "__main__":
    main()