import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sksurv.linear_model.coxph import BreslowEstimator
from utility.training import get_data_loader, scale_data, split_time_event
from utility.survival import survival_probability_calibration
from utility.model import load_mlp_model, load_sota_model, load_mcd_model, load_sngp_model, load_vi_model
from utility.survival import compute_nondeterministic_survival_curve
from utility.plot import plot_calibration_curves
from collections import defaultdict
from pathlib import Path
import paths as pt
from utility.survival import make_time_bins, calculate_event_times
from tools.preprocessor import Preprocessor
from utility.survival import (calculate_event_times, calculate_percentiles, convert_to_structured,
                              compute_deterministic_survival_curve, compute_nondeterministic_survival_curve)
from utility.training import make_stratified_split

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

N_SAMPLES_TRAIN = 10
N_SAMPLES_TEST = 100

if __name__ == "__main__":
    # Load data
    dataset_name = "METABRIC"
    dl = get_data_loader(dataset_name).load_data()
    num_features, cat_features = dl.get_features()
    df = dl.get_data()
    
    # Split data
    df_train, df_valid, df_test = make_stratified_split(df, stratify_colname='both', frac_train=0.7,
                                                        frac_valid=0.1, frac_test=0.2, random_state=0)
    X_train = df_train[cat_features+num_features]
    X_valid = df_valid[cat_features+num_features]
    X_test = df_test[cat_features+num_features]
    y_train = convert_to_structured(df_train["time"], df_train["event"])
    y_test = convert_to_structured(df_test["time"], df_test["event"])

    # Scale data
    X_train, X_valid, X_test = scale_data(X_train, X_valid, X_test, cat_features, num_features)
    
    # Convert to array
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # Make time/event split
    t_train, e_train = split_time_event(y_train)
    t_test, e_test = split_time_event(y_test)

    # Make event times
    event_times = calculate_event_times(t_train, e_train)
    
    # Calculate quantiles
    percentiles = dict()
    for q in [25, 50, 75, 90]:
        t = int(np.percentile(event_times, q))
        t_nearest = find_nearest(event_times, t)
        percentiles[q] = t_nearest

    # Load models
    n_input_dims = X_train.shape[1:]
    n_train_samples = X_train.shape[0]
    cox_model = load_sota_model(dataset_name, "cox")
    mlp_model = load_mlp_model(dataset_name, n_input_dims)
    sngp_model = load_sngp_model(dataset_name, n_input_dims)
    vi_model = load_vi_model(dataset_name, len(X_train), n_input_dims)
    mcd_model1 = load_mcd_model(dataset_name, "mcd1", n_input_dims)
    mcd_model2 = load_mcd_model(dataset_name, "mcd2", n_input_dims)
    mcd_model3 = load_mcd_model(dataset_name, "mcd3", n_input_dims)

    # Compute calibration curves
    pred_obs, predictions, deltas = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    models = {'mlp': mlp_model, "sngp": sngp_model, "vi": vi_model,
              'mcd1': mcd_model1, 'mcd2': mcd_model2, 'mcd3': mcd_model2}
    for t0 in percentiles.values():
        for model_name, model in models.items():
            if model_name in ["cox", "rsf", "mlp", "sngp"]:
                surv_preds = compute_deterministic_survival_curve(model, X_train, X_test, e_train, t_train, event_times, model_name)
            else:
                surv_preds = np.mean(compute_nondeterministic_survival_curve(model, X_train, X_test,
                                                                             e_train, t_train, event_times,
                                                                             N_SAMPLES_TRAIN, N_SAMPLES_TEST), axis=0)
            surv_preds = pd.DataFrame(surv_preds, columns=event_times)
            pred_t0, obs_t0, predictions_at_t0, deltas_t0 = survival_probability_calibration(surv_preds, t_test, e_test, t0)
            pred_obs[t0][model_name] = (pred_t0, obs_t0)
            predictions[t0][model_name] = predictions_at_t0
            deltas[t0][model_name] = deltas_t0

    # Plot calibration curves
    model_names = models.keys()
    plot_calibration_curves(percentiles, pred_obs, predictions, model_names, dataset_name)