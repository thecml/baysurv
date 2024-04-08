import numpy as np
import os
from utility.tuning import *
import argparse
from tools import data_loader
import pandas as pd
from utility.training import split_time_event
import config as cfg
from utility.survival import make_time_bins, calculate_event_times, compute_deterministic_survival_curve
from tools.sota_builder import make_cox_model, make_coxnet_model, make_coxboost_model
from tools.sota_builder import make_rsf_model, make_dsm_model, make_dcm_model, make_dcph_model
from tools.sota_builder import make_baycox_model, make_baymtlr_model
from tools.bnn_isd_trainer import train_bnn_model
from utility.bnn_isd_models import make_ensemble_cox_prediction, make_ensemble_mtlr_prediction
from utility.training import make_stratified_split, scale_data
from utility.survival import convert_to_structured
from tools.Evaluations.util import make_monotonic
from pycox.evaluation import EvalSurv
import torch

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

os.environ["WANDB_SILENT"] = "true"
import wandb

N_RUNS = 10
PROJECT_NAME = "baysurv_bo"

# Setup device
device = "cpu" # use CPU
device = torch.device(device)

def main():
    global model_name
    global dataset_name
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        required=True,
                        default=None)
    parser.add_argument('--model', type=str,
                        required=True,
                        default=None)
    args = parser.parse_args()
    
    if args.dataset:
        dataset_name = args.dataset
    if args.model:
        model_name = args.model
        
    if model_name == "cox":
        sweep_config = get_cox_sweep_config()
    elif model_name == "coxboost":
        sweep_config = get_coxboost_sweep_config()
    elif model_name == "coxnet":
        sweep_config = get_coxnet_sweep_config()
    elif model_name == "dcph":
        sweep_config = get_dcph_sweep_config()
    elif model_name == "dcm":
        sweep_config = get_dcm_sweep_config()
    elif model_name == "dsm":
        sweep_config = get_dsm_sweep_config()
    elif model_name == "rsf":
        sweep_config = get_rsf_sweep_config()
    elif model_name == "baycox":
        sweep_config = get_baycox_sweep_config()
    elif model_name == "baymtlr":
        sweep_config = get_baymtlr_sweep_config()
    else:
        raise ValueError("Model not found")
    
    sweep_id = wandb.sweep(sweep_config, project=f'{PROJECT_NAME}_{model_name}')
    wandb.agent(sweep_id, train_model, count=N_RUNS)

def train_model():
# Make and train mdoel
    if model_name == "cox":
        config_defaults = cfg.COX_DEFAULT_PARAMS
    elif model_name == "coxboost":
        config_defaults = cfg.COXBOOST_DEFAULT_PARAMS
    elif model_name == "coxnet":
        config_defaults = cfg.COXNET_DEFAULT_PARAMS
    elif model_name == "dcm":
        config_defaults = cfg.DCM_DEFAULT_PARAMS
    elif model_name == "dcph":
        config_defaults = cfg.DCPH_DEFAULT_PARAMS
    elif model_name == "dsm":
        config_defaults = cfg.DSM_DEFAULT_PARAMS
    elif model_name == "rsf":
        config_defaults = cfg.RSF_DEFAULT_PARAMS
    elif model_name == "baycox":
        config_defaults = cfg.BAYCOX_DEFAULT_PARAMS
    elif model_name == "baymtlr":
        config_defaults = cfg.BAYMTLR_DEFAULT_PARAMS
    else:
        raise ValueError("Model not found")

    # Initialize a new wandb run
    wandb.init(config=config_defaults, group=dataset_name)
    config = wandb.config

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
    
    # Make time/event split
    t_train, e_train = split_time_event(y_train)
    t_valid, e_valid = split_time_event(y_valid)

    # Make event times
    event_times = calculate_event_times(t_train, e_train)
    mtlr_times = make_time_bins(t_train, event=e_train)
    
    # Make and train mdoel
    if model_name == "cox":
        model = make_cox_model(config)
        model.fit(X_train, y_train)
    elif model_name == "coxboost":
        model = make_coxboost_model(config)
        model.fit(X_train, y_train)
    elif model_name == "coxnet":
        model = make_coxnet_model(config)
        model.fit(X_train, y_train)
    elif model_name == "dcph":
        model = make_dcph_model(config)
        model.fit(X_train, pd.DataFrame(y_train),
                  val_data=(X_valid, pd.DataFrame(y_valid)))
    elif model_name == "dcm":
        model = make_dcm_model(config)
        model.fit(X_train, pd.DataFrame(y_train),
                  val_data=(X_valid, pd.DataFrame(y_valid)))
    elif model_name == "dsm":
        model = make_dsm_model(config)
        model.fit(X_train, pd.DataFrame(y_train),
                  val_data=(X_valid, pd.DataFrame(y_valid)))
    elif model_name == "rsf":
        model = make_rsf_model(config)
        model.fit(X_train, y_train)
    elif model_name == "baycox":
        data_train = X_train.copy()
        data_train["time"] = pd.Series(y_train['time'])
        data_train["event"] = pd.Series(y_train['event']).astype(int)
        data_valid = X_valid.copy()
        data_valid["time"] = pd.Series(y_valid['time'])
        data_valid["event"] = pd.Series(y_valid['event']).astype(int)
        num_features = X_train.shape[1]
        model = make_baycox_model(num_features, config)
        model = train_bnn_model(model, data_train, data_valid, mtlr_times,
                                config=config, random_state=0, reset_model=True, device=device)
    elif model_name == "baymtlr":
        data_train = X_train.copy()
        data_train["time"] = pd.Series(y_train['time'])
        data_train["event"] = pd.Series(y_train['event']).astype(int)
        data_valid = X_valid.copy()
        data_valid["time"] = pd.Series(y_valid['time'])
        data_valid["event"] = pd.Series(y_valid['event']).astype(int)
        num_features = X_train.shape[1]
        model = make_baymtlr_model(num_features, mtlr_times, config)
        model = train_bnn_model(model, data_train, data_valid,
                                mtlr_times, config=config,
                                random_state=0, reset_model=True, device=device)
    else:
        raise ValueError("Model not found")
    
    # Compute survival function
    if model_name == "dsm":
        surv_preds = pd.DataFrame(model.predict_survival(X_valid, times=list(event_times)), columns=event_times)
    elif model_name == "dcph":
        surv_preds = pd.DataFrame(model.predict_survival(X_valid, t=list(event_times)), columns=event_times)
    elif model_name == "dcm":
        surv_preds = pd.DataFrame(model.predict_survival(X_valid, times=list(event_times)), columns=event_times)
    elif model_name == "cox":
        test_surv_fn = model.predict_survival_function(X_valid)
        surv_preds = np.row_stack([fn(event_times) for fn in test_surv_fn])
    elif model_name == "coxnet":
        test_surv_fn = model.predict_survival_function(X_valid)
        surv_preds = np.row_stack([fn(event_times) for fn in test_surv_fn])
    elif model_name == "rsf":
        test_surv_fn = model.predict_survival_function(X_valid)
        surv_preds = np.row_stack([fn(event_times) for fn in test_surv_fn])
    elif model_name == "coxboost":
        test_surv_fn = model.predict_survival_function(X_valid)
        surv_preds = np.row_stack([fn(event_times) for fn in test_surv_fn])
    elif model_name == "baycox":
        baycox_valid_data = torch.tensor(data_valid.drop(["time", "event"], axis=1).values,
                                        dtype=torch.float, device=device)
        survival_outputs, _, ensemble_outputs = make_ensemble_cox_prediction(model, baycox_valid_data, config)
        surv_preds = survival_outputs.numpy()
    elif model_name == "baymtlr":
        baycox_valid_data = torch.tensor(data_valid.drop(["time", "event"], axis=1).values,
                                        dtype=torch.float, device=device)
        survival_outputs, _, ensemble_outputs = make_ensemble_mtlr_prediction(model, baycox_valid_data, mtlr_times, config)
        surv_preds = survival_outputs.numpy()
    else:
        surv_preds = compute_deterministic_survival_curve(model, np.array(X_train), np.array(X_valid),
                                                          e_train, t_train, event_times, model_name)
        
    # Make DCM monotonic
    if model_name == "dcm":
        surv_preds = make_monotonic(surv_preds.to_numpy(), event_times, method='ceil')
        
    # Convert to DataFrame
    if model_name == "baymtlr":
        mtlr_times = torch.cat([torch.tensor([0]).to(mtlr_times.device), mtlr_times], 0)
        surv_preds = pd.DataFrame(surv_preds, columns=mtlr_times.numpy())
    else:
        surv_preds = pd.DataFrame(surv_preds, dtype=np.float64, columns=event_times)
    
    try:
        ev = EvalSurv(surv_preds.T, t_valid, e_valid, censor_surv="km")
        ci = ev.concordance_td()
    except:
        ci = np.nan
    
    # Log to wandb
    wandb.log({"val_ci": ci})

if __name__ == "__main__":
    main()


