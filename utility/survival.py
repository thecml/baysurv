import numpy as np
import pandas as pd
from sksurv.linear_model.coxph import BreslowEstimator
import matplotlib.pyplot as plt
from lifelines.utils import CensoringType
from lifelines.fitters import RegressionFitter
from lifelines import CRCSplineFitter
import warnings
import torch
import math
from typing import Optional
from typing import List, Tuple, Optional, Union
import rpy2.robjects as robjects
import scipy.integrate as integrate
from dataclasses import InitVar, dataclass, field

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

Numeric = Union[float, int, bool]
NumericArrayLike = Union[List[Numeric], Tuple[Numeric], np.ndarray, pd.Series, pd.DataFrame, torch.Tensor]

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def calculate_percentiles(times):
    times_pct = dict()
    for q in [25, 50, 75]:
        t = int(np.percentile(times, q))
        t_nearest = find_nearest(times, t)
        times_pct[q] = t_nearest
    return times_pct

def encode_survival(
        time: Union[float, int, NumericArrayLike],
        event: Union[int, bool, NumericArrayLike],
        bins: NumericArrayLike
) -> torch.Tensor:
    """Encodes survival time and event indicator in the format
    required for MTLR training.

    For uncensored instances, one-hot encoding of binned survival time
    is generated. Censoring is handled differently, with all possible
    values for event time encoded as 1s. For example, if 5 time bins are used,
    an instance experiencing event in bin 3 is encoded as [0, 0, 0, 1, 0], and
    instance censored in bin 2 as [0, 0, 1, 1, 1]. Note that an additional
    'catch-all' bin is added, spanning the range `(bins.max(), inf)`.

    Parameters
    ----------
    time
        Time of event or censoring.
    event
        Event indicator (0 = censored).
    bins
        Bins used for time axis discretisation.

    Returns
    -------
    torch.Tensor
        Encoded survival times.
    """
    # TODO this should handle arrays and (CUDA) tensors
    if isinstance(time, (float, int, np.ndarray)):
        time = np.atleast_1d(time)
        time = torch.tensor(time)
    if isinstance(event, (int, bool, np.ndarray)):
        event = np.atleast_1d(event)
        event = torch.tensor(event)

    if isinstance(bins, np.ndarray):
        bins = torch.tensor(bins)

    try:
        device = bins.device
    except AttributeError:
        device = "cpu"

    time = np.clip(time, 0, bins.max())
    # add extra bin [max_time, inf) at the end
    y = torch.zeros((time.shape[0], bins.shape[0] + 1),
                    dtype=torch.float,
                    device=device)
    # For some reason, the `right` arg in torch.bucketize
    # works in the _opposite_ way as it does in numpy,
    # so we need to set it to True
    bin_idxs = torch.bucketize(time, bins, right=True)
    for i, (bin_idx, e) in enumerate(zip(bin_idxs, event)):
        if e == 1:
            y[i, bin_idx] = 1
        else:
            y[i, bin_idx:] = 1
    return y.squeeze()

def reformat_survival(
        dataset: pd.DataFrame,
        time_bins: NumericArrayLike
) -> (torch.Tensor, torch.Tensor):
    x = torch.tensor(dataset.drop(["time", "event"], axis=1).values, dtype=torch.float)
    y = encode_survival(dataset["time"].values, dataset["event"].values, time_bins)
    return x, y

def mtlr_survival(
        logits: torch.Tensor,
        with_sample: bool = True
) -> torch.Tensor:
    """Generates predicted survival curves from predicted logits.

    Parameters
    ----------
    logits
        Tensor with the time-logits (as returned by the MTLR module)
        with size (n_samples, n_data, n_bins) or (n_data, n_bins).

    Returns
    -------
    torch.Tensor
        The predicted survival curves for each row in `pred` at timepoints used
        during training.
    """
    # TODO: do not reallocate G in every call
    if with_sample:
        assert logits.dim() == 3, "The logits should have dimension with with size (n_samples, n_data, n_bins)"
        G = torch.tril(torch.ones(logits.shape[2], logits.shape[2])).to(logits.device)
        density = torch.softmax(logits, dim=2)
        G_with_samples = G.expand(density.shape[0], -1, -1)

        # b: n_samples; i: n_data; j: n_bin; k: n_bin
        return torch.einsum('bij,bjk->bik', density, G_with_samples)
    else:   # no sampling
        assert logits.dim() == 2, "The logits should have dimension with with size (n_data, n_bins)"
        G = torch.tril(torch.ones(logits.shape[1], logits.shape[1])).to(logits.device)
        density = torch.softmax(logits, dim=1)
        return torch.matmul(density, G)

def cox_survival(
        baseline_survival: torch.Tensor,
        linear_predictor: torch.Tensor
) -> torch.Tensor:
    """
    Calculate the individual survival distributions based on the baseline survival curves and the liner prediction values.
    :param baseline_survival: (n_time_bins, )
    :param linear_predictor: (n_samples, n_data)
    :return:
    The invidual survival distributions. shape = (n_samples, n_time_bins)
    """
    n_sample = linear_predictor.shape[0]
    n_data = linear_predictor.shape[1]
    risk_score = torch.exp(linear_predictor)
    survival_curves = torch.empty((n_sample, n_data, baseline_survival.shape[0]), dtype=torch.float).to(linear_predictor.device)
    for i in range(n_sample):
        for j in range(n_data):
            survival_curves[i, j, :] = torch.pow(baseline_survival, risk_score[i, j])
    return survival_curves

def make_monotonic(
        array: Union[torch.Tensor, np.ndarray, list]
):
    for i in range(len(array) - 1):
        if not array[i] >= array[i + 1]:
            array[i + 1] = array[i]
    return array

def compute_unique_counts(
        event: torch.Tensor,
        time: torch.Tensor,
        order: Optional[torch.Tensor] = None):
    """Count right censored and uncensored samples at each unique time point.

    Parameters
    ----------
    event : array
        Boolean event indicator.

    time : array
        Survival time or time of censoring.

    order : array or None
        Indices to order time in ascending order.
        If None, order will be computed.

    Returns
    -------
    times : array
        Unique time points.

    n_events : array
        Number of events at each time point.

    n_at_risk : array
        Number of samples that have not been censored or have not had an event at each time point.

    n_censored : array
        Number of censored samples at each time point.
    """
    n_samples = event.shape[0]

    if order is None:
        order = torch.argsort(time)

    uniq_times = torch.empty(n_samples, dtype=time.dtype, device=time.device)
    uniq_events = torch.empty(n_samples, dtype=torch.int, device=time.device)
    uniq_counts = torch.empty(n_samples, dtype=torch.int, device=time.device)

    i = 0
    prev_val = time[order[0]]
    j = 0
    while True:
        count_event = 0
        count = 0
        while i < n_samples and prev_val == time[order[i]]:
            if event[order[i]]:
                count_event += 1

            count += 1
            i += 1

        uniq_times[j] = prev_val
        uniq_events[j] = count_event
        uniq_counts[j] = count
        j += 1

        if i == n_samples:
            break

        prev_val = time[order[i]]

    uniq_times = uniq_times[:j]
    uniq_events = uniq_events[:j]
    uniq_counts = uniq_counts[:j]
    n_censored = uniq_counts - uniq_events

    # offset cumulative sum by one
    total_count = torch.cat([torch.tensor([0], device=uniq_counts.device), uniq_counts], dim=0)
    n_at_risk = n_samples - torch.cumsum(total_count, dim=0)

    return uniq_times, uniq_events, n_at_risk[:-1], n_censored

def calculate_event_times(t_train, e_train):
    unique_times = compute_unique_counts(torch.Tensor(e_train), torch.Tensor(t_train))[0]
    if 0 not in unique_times:
        unique_times = torch.cat([torch.tensor([0]).to(unique_times.device), unique_times], 0)
    return unique_times.numpy() 

def make_time_bins(
        times: NumericArrayLike,
        num_bins: Optional[int] = None,
        use_quantiles: bool = True,
        event: Optional[NumericArrayLike] = None
) -> torch.Tensor:
    """
    Courtesy of https://ieeexplore.ieee.org/document/10158019
    
    Creates the bins for survival time discretisation.

    By default, sqrt(num_observation) bins corresponding to the quantiles of
    the survival time distribution are used, as in https://github.com/haiderstats/MTLR.

    Parameters
    ----------
    times
        Array or tensor of survival times.
    num_bins
        The number of bins to use. If None (default), sqrt(num_observations)
        bins will be used.
    use_quantiles
        If True, the bin edges will correspond to quantiles of `times`
        (default). Otherwise, generates equally-spaced bins.
    event
        Array or tensor of event indicators. If specified, only samples where
        event == 1 will be used to determine the time bins.

    Returns
    -------
    torch.Tensor
        Tensor of bin edges.
    """
    # TODO this should handle arrays and (CUDA) tensors
    if event is not None:
        times = times[event == 1]
    if num_bins is None:
        num_bins = math.ceil(math.sqrt(len(times)))
    if use_quantiles:
        # NOTE we should switch to using torch.quantile once it becomes
        # available in the next version
        bins = np.unique(np.quantile(times, np.linspace(0, 1, num_bins)))
    else:
        bins = np.linspace(times.min(), times.max(), num_bins)
    bins = torch.tensor(bins, dtype=torch.float)
    return bins

def survival_probability_calibration(surv_preds: pd.DataFrame,
                                     times,
                                     events,
                                     t0: float):
    def safe_log(x, eps=1e-10):
        result = np.where(x > eps, x, -10)
        np.log(result, out=result, where=result > 0)
        return result

    def ccl(p):
        return safe_log(-safe_log(1-p))
    
    T = "Survival_time"
    E = "Event"
        
    predictions_at_t0 = np.clip(1 - surv_preds[t0].squeeze(), 1e-10, 1 - 1e-10)

    # create new dataset with the predictions
    prediction_df = pd.DataFrame({"ccl_at_%d" % t0: ccl(predictions_at_t0), T: times, E: events})

    # fit new dataset to flexible spline model
    # this new model connects prediction probabilities and actual survival. It should be very flexible, almost to the point of overfitting. It's goal is just to smooth out the data!
    knots = 3
    regressors = {"beta_": ["ccl_at_%d" % t0], "gamma0_": "1", "gamma1_": "1", "gamma2_": "1"}

    # this model is from examples/royson_crowther_clements_splines.py
    crc = CRCSplineFitter(knots, penalizer=0.000001)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        crc.fit_right_censoring(prediction_df, T, E, regressors=regressors) # only support right-censoring for now

    # predict new model at values 0 to 1, but remember to ccl it!
    x = np.linspace(np.clip(predictions_at_t0.min() - 0.01, 0, 1), np.clip(predictions_at_t0.max() + 0.01, 0, 1), 100)
    y = 1 - crc.predict_survival_function(pd.DataFrame({"ccl_at_%d" % t0: ccl(x)}), times=[t0]).T.squeeze()

    deltas = ((1 - crc.predict_survival_function(prediction_df, times=[t0])).T.squeeze() - predictions_at_t0).abs()
    
    return x, y, predictions_at_t0, deltas

def compute_survival_scale(risk_scores, t_train, e_train):
    # https://pubmed.ncbi.nlm.nih.gov/15724232/
    rnd = np.random.RandomState()

    # generate hazard scale
    mean_survival_time = t_train[e_train].mean()
    baseline_hazard = 1. / mean_survival_time
    scale = baseline_hazard * np.exp(risk_scores)
    return scale

def compute_survival_times(risk_scores, t_train, e_train):
    # https://pubmed.ncbi.nlm.nih.gov/15724232/
    rnd = np.random.RandomState(0)
        
    # generate survival time
    mean_survival_time = t_train[e_train].mean()
    baseline_hazard = 1. / mean_survival_time
    scale = baseline_hazard * np.exp(risk_scores)
    u = rnd.uniform(low=0, high=1, size=risk_scores.shape[0])
    t = -np.log(u) / scale

    return t

def compute_survival_times_with_censoring(risk_scores, t_train, e_train):
    # https://pubmed.ncbi.nlm.nih.gov/15724232/
    rnd = np.random.RandomState(0)

    # generate survival time
    mean_survival_time = t_train[e_train].mean()
    baseline_hazard = 1. / mean_survival_time
    scale = baseline_hazard * np.exp(risk_scores)
    u = rnd.uniform(low=0, high=1, size=risk_scores.shape[0])
    t = -np.log(u) / scale

    # generate time of censoring
    prob_censored = 1 - e_train.sum()/len(e_train)
    qt = np.quantile(t, 1.0 - prob_censored)
    c = rnd.uniform(low=t.min(), high=qt)

    # apply censoring
    observed_event = t <= c
    observed_time = np.where(observed_event, t, c)
    return observed_time, observed_event

def convert_to_structured(T, E):
    # dtypes for conversion
    default_dtypes = {"names": ("event", "time"), "formats": ("bool", "i4")}

    # concat of events and times
    concat = list(zip(E, T))

    # return structured array
    return np.array(concat, dtype=default_dtypes)

def compute_deterministic_survival_curve(model, X_train, X_test, e_train, t_train,
                                         event_times, model_name):
    if model_name == "sngp":
        train_logits = model.predict(X_train, verbose=False)[0].reshape(-1)
        test_logits = model.predict(X_test, verbose=False)[0].reshape(-1)
    elif model_name == "mlp":
        train_logits = model.predict(X_train, verbose=False).reshape(-1)
        test_logits = model.predict(X_test, verbose=False).reshape(-1)
    else:
        train_logits = model.predict(X_train).reshape(-1)
        test_logits = model.predict(X_test).reshape(-1)
    breslow = BreslowEstimator().fit(train_logits, e_train, t_train)
    surv_fn = breslow.get_survival_function(test_logits)
    breslow_surv_times = np.row_stack([fn(event_times) for fn in surv_fn])
    return breslow_surv_times

def compute_nondeterministic_survival_curve(model, X_train, X_test, e_train, t_train,
                                            event_times, n_samples_train, n_samples_test):
    train_cpd = np.zeros((n_samples_train, len(X_train)))
    for i in range(0, n_samples_train):
        train_logits = model.predict(X_train, verbose=False)
        train_cpd[i,:] = np.reshape(train_logits, len(X_train))
    breslow = BreslowEstimator().fit(np.mean(train_cpd, axis=0), e_train, t_train)
    breslow_surv_times = np.zeros((n_samples_test, len(X_test), len(event_times)))
    for i in range(0, n_samples_test):
        test_logits = model.predict(X_test, verbose=False)
        surv_fn = breslow.get_survival_function(np.reshape(test_logits, len(X_test)))
        breslow_surv_times[i] = np.row_stack([fn(event_times) for fn in surv_fn])
    return breslow_surv_times

def coverage(time_bins, upper, lower, true_times, true_indicator) -> float:
    '''Courtesy of https://github.com/shi-ang/BNN-ISD/tree/main'''
    time_bins = check_and_convert(time_bins)
    upper, lower = check_and_convert(upper, lower)
    true_times, true_indicator = check_and_convert(true_times, true_indicator)
    true_indicator = true_indicator.astype(bool)
    covered = 0
    upper_median_times = predict_median_survival_times(upper, time_bins, round_up=True)
    lower_median_times = predict_median_survival_times(lower, time_bins, round_up=False)
    covered += 2 * np.logical_and(upper_median_times[true_indicator] >= true_times[true_indicator],
                                  lower_median_times[true_indicator] <= true_times[true_indicator]).sum()
    covered += np.sum(upper_median_times[~true_indicator] >= true_times[~true_indicator])
    total = 2 * true_indicator.sum() + (~true_indicator).sum()
    return covered / total

def coverage_curves(
        upper: torch.Tensor,
        lower: torch.Tensor,
        test_curves: torch.Tensor
) -> float:
    upper = upper.cpu().detach().numpy()
    lower = lower.cpu().detach().numpy()
    test_curves = test_curves.cpu().detach().numpy()
    return ((upper >= test_curves) & (lower <= test_curves)).mean()


def predict_median_survival_times(
        survival_curves: np.ndarray,
        times_coordinate: np.ndarray,
        round_up: bool = True
):
    median_probability_times = np.zeros(survival_curves.shape[0])
    max_time = times_coordinate[-1]
    slopes = (1 - survival_curves[:, -1]) / (0 - max_time)

    if round_up:
        # Find the first index in each row that are smaller or equal than 0.5
        times_indices = np.where(survival_curves <= 0.5, survival_curves, -np.inf).argmax(axis=1)
    else:
        # Find the last index in each row that are larger or equal than 0.5
        times_indices = np.where(survival_curves >= 0.5, survival_curves, np.inf).argmin(axis=1)

    need_extend = survival_curves[:, -1] > 0.5
    median_probability_times[~need_extend] = times_coordinate[times_indices][~need_extend]
    median_probability_times[need_extend] = (max_time + (0.5 - survival_curves[:, -1]) / slopes)[need_extend]

    return median_probability_times

def predict_mean_survival_time(
        survival_curve: np.ndarray,
        times_coordinate: np.ndarray
):
    # If all the predicted probabilities are 1 the integral will be infinite.
    if np.all(survival_curve == 1):
        warnings.warn("All the predicted probabilities are 1, the integral will be infinite.")
        return np.inf

    x = robjects.FloatVector(times_coordinate)
    y = robjects.FloatVector(survival_curve)
    r_splinefun = robjects.r['splinefun']  # extract splinefun method from R
    spline = r_splinefun(x, y, method='hyman')

    # predicting boundary
    max_time = max(times_coordinate.tolist())

    # simply calculate the slope by using the [0, 1] - [max_time, S(t|x)]
    slope = (1 - np.array(spline(max_time)).item()) / (0 - max_time)

    # zero_probability_time = min(times_coordinate[np.where(survival_curve == 0)],
    #                             max_time + (0 - np.array(spline(max_time)).item()) / slope)
    if 0 in survival_curve:
        zero_probability_time = min(times_coordinate[np.where(survival_curve == 0)])
    else:
        zero_probability_time = max_time + (0 - np.array(spline(max_time)).item()) / slope

    def _func_to_integral(time, maximum_time, slope_rate):
        return np.array(spline(time)).item() if time < maximum_time else (1 + time * slope_rate)
    # _func_to_integral = lambda time: spline(time) if time < max_time else (1 + time * slope)
    # limit controls the subdivision intervals used in the adaptive algorithm.
    # Set it to 1000 is consistent with Haider's R code
    mean_survival_time, *rest = integrate.quad(_func_to_integral, 0, zero_probability_time,
                                               args=(max_time, slope), limit=1000)
    return mean_survival_time

def predict_median_survival_time(
        survival_curve: np.ndarray,
        times_coordinate: np.ndarray
):
    # If all the predicted probabilities are 1 the integral will be infinite.
    if np.all(survival_curve == 1):
        warnings.warn("All the predicted probabilities are 1, the median survival time will be infinite.")
        return np.inf

    x = robjects.FloatVector(times_coordinate)
    y = robjects.FloatVector(survival_curve)
    r_splinefun = robjects.r['splinefun']  # extract splinefun method from R
    spline = r_splinefun(x, y, method='hyman')

    min_prob = min(spline(times_coordinate.tolist()))

    if 0.5 in survival_curve:
        median_probability_time = times_coordinate[np.where(survival_curve == 0.5)[0][0]]
    elif min_prob < 0.5:
        min_time_before_median = times_coordinate[np.where(survival_curve > 0.5)[0][-1]]
        max_time_after_median = times_coordinate[np.where(survival_curve < 0.5)[0][0]]

        prob_range = robjects.FloatVector(
            spline(np.linspace(min_time_before_median, max_time_after_median, num=1000).tolist()))
        time_range = robjects.FloatVector(np.linspace(min_time_before_median, max_time_after_median, num=1000))
        inverse_spline = r_splinefun(prob_range, time_range, method='hyman')
        # Need to convert the R floatvector to numpy array and use .item() to obtain the single value
        median_probability_time = np.array(inverse_spline(0.5)).item()
    else:
        max_time = max(times_coordinate.tolist())
        slope = (1 - np.array(spline(max_time)).item()) / (0 - max_time)
        median_probability_time = max_time + (0.5 - np.array(spline(max_time)).item()) / slope

    return median_probability_time

def predict_prob_from_curve(
        survival_curve: np.ndarray,
        times_coordinate: np.ndarray,
        target_time: float
) -> float:
    """
    Quote from ISDEvaluation/Evaluations/EvaluationHelperFunction.R
    We need some type of predict function for survival curves - here we build a spline to fit the survival model curve.
    This spline is the monotonic spline using the hyman filtering of the cubic Hermite spline method,
    see https://en.wikipedia.org/wiki/Monotone_cubic_interpolation. Also see help(splinefun).

    Note that we make an alteration to the method because if the last two time points
    have the same probability (y value) then the spline is constant outside of the training data.
    We need this to be a decreasing function outside the training data so instead we take the linear fit of (0,1)
    and the last time point we have (p,t*) and then apply this linear function to all points outside of our fit.
    """
    x = robjects.FloatVector(times_coordinate)
    y = robjects.FloatVector(survival_curve)
    r_splinefun = robjects.r['splinefun']  # extract splinefun method from R
    spline = r_splinefun(x, y, method='hyman')

    # predicting boundary
    max_time = float(max(times_coordinate))

    # simply calculate the slope by using the [0, 1] - [max_time, S(t|x)]
    # Need to convert the R floatvector to numpy array and use .item() to obtain the single value
    slope = (1 - np.array(spline(max_time)).item()) / (0 - max_time)

    # If the true event time is out of predicting boundary, then use the linear fit mentioned above;
    # Else if the true event time is in the boundary, then use the spline
    if target_time > max_time:
        # func: y = slope * x + 1, the minimum prob should be 0
        predict_probability = max(slope * target_time + 1, 0)
    else:
        predict_probability = np.array(spline(float(target_time))).item()

    return predict_probability


def predict_multi_probs_from_curve(
        survival_curve: np.ndarray,
        times_coordinate: np.ndarray,
        target_times: NumericArrayLike
) -> np.ndarray:
    """
    Quote from ISDEvaluation/Evaluations/EvaluationHelperFunction.R
    We need some type of predict function for survival curves - here we build a spline to fit the survival model curve.
    This spline is the monotonic spline using the hyman filtering of the cubic Hermite spline method,
    see https://en.wikipedia.org/wiki/Monotone_cubic_interpolation. Also see help(splinefun).

    Note that we make an alteration to the method because if the last two time points
    have the same probability (y value) then the spline is constant outside of the training data.
    We need this to be a decreasing function outside the training data so instead we take the linear fit of (0,1)
    and the last time point we have (p,t*) and then apply this linear function to all points outside of our fit.
    """
    target_times = check_and_convert(target_times).astype(float).tolist()

    x = robjects.FloatVector(times_coordinate)
    y = robjects.FloatVector(survival_curve)
    r_splinefun = robjects.r['splinefun']  # extract splinefun method from R
    spline = r_splinefun(x, y, method='hyman')

    # predicting boundary
    max_time = float(max(times_coordinate))

    # simply calculate the slope by using the [0, 1] - [maxtime, S(t|x)]
    # Need to convert the R floatvector to numpy array and use .item() to obtain the single value
    slope = (1 - np.array(spline(max_time)).item()) / (0 - max_time)

    # If the true event time is out of predicting boundary, then use the linear fit mentioned above;
    # Else if the true event time is in the boundary, then use the spline
    predict_probabilities = np.array(spline(target_times))
    for i, target_time in enumerate(target_times):
        if target_time > max_time:
            predict_probabilities[i] = max(slope * target_time + 1, 0)

    return predict_probabilities

def check_and_convert(*args):
    """ Makes sure that the given inputs are numpy arrays, list,
        tuple, panda Series, pandas DataFrames, or torch Tensors.

        Also makes sure that the given inputs have the same shape.

        Then convert the inputs to numpy array.

        Parameters
        ----------
        * args : tuple of objects
                 Input object to check / convert.

        Returns
        -------
        * result : tuple of numpy arrays
                   The converted and validated arg.

        If the input isn't numpy arrays, list or pandas DataFrames, it will
        fail and ask to provide the valid format.
    """

    result = ()
    last_length = ()
    for i, arg in enumerate(args):

        if len(arg) == 0:
            error = " The input is empty. "
            error += "Please provide at least 1 element in the array."
            raise IndexError(error)

        else:

            if isinstance(arg, np.ndarray):
                x = (arg.astype(np.double),)
            elif isinstance(arg, list):
                x = (np.asarray(arg).astype(np.double),)
            elif isinstance(arg, tuple):
                x = (np.asarray(arg).astype(np.double),)
            elif isinstance(arg, pd.Series):
                x = (arg.values.astype(np.double),)
            elif isinstance(arg, pd.DataFrame):
                x = (arg.values.astype(np.double),)
            elif isinstance(arg, torch.Tensor):
                x = (arg.cpu().numpy().astype(np.double),)
            else:
                error = """{arg} is not a valid data format. Only use 'list', 'tuple', 'np.ndarray', 'torch.Tensor', 
                        'pd.Series', 'pd.DataFrame'""".format(arg=type(arg))
                raise TypeError(error)

            if np.sum(np.isnan(x)) > 0.:
                error = "The #{} argument contains null values"
                error = error.format(i + 1)
                raise ValueError(error)

            if len(args) > 1:
                if i > 0:
                    assert x[0].shape == last_length, """Shapes between {}-th input array and 
                    {}-th input array are not consistent""".format(i - 1, i)
                result += x
                last_length = x[0].shape
            else:
                result = x[0]

    return result

@dataclass
class KaplanMeier:
    """
    This class is borrowed from survival_evaluation package.
    """
    event_times: InitVar[np.array]
    event_indicators: InitVar[np.array]
    survival_times: np.array = field(init=False)
    survival_probabilities: np.array = field(init=False)

    def __post_init__(self, event_times, event_indicators):
        index = np.lexsort((event_indicators, event_times))
        unique_times = np.unique(event_times[index], return_counts=True)
        self.survival_times = unique_times[0]
        population_count = np.flip(np.flip(unique_times[1]).cumsum())

        event_counter = np.append(0, unique_times[1].cumsum()[:-1])
        event_ind = list()
        for i in range(np.size(event_counter[:-1])):
            event_ind.append(event_counter[i])
            event_ind.append(event_counter[i + 1])
        event_ind.append(event_counter[-1])
        event_ind.append(len(event_indicators))
        events = np.add.reduceat(np.append(event_indicators[index], 0), event_ind)[::2]

        self.survival_probabilities = np.empty(population_count.size)
        survival_probability = 1
        counter = 0
        for population, event_num in zip(population_count, events):
            survival_probability *= 1 - event_num / population
            self.survival_probabilities[counter] = survival_probability
            counter += 1

    def predict(self, prediction_times: np.array):
        probability_index = np.digitize(prediction_times, self.survival_times)
        probability_index = np.where(
            probability_index == self.survival_times.size + 1,
            probability_index - 1,
            probability_index,
        )
        probabilities = np.append(1, self.survival_probabilities)[probability_index]

        return probabilities

@dataclass
class KaplanMeierArea(KaplanMeier):
    area_times: np.array = field(init=False)
    area_probabilities: np.array = field(init=False)
    area: np.array = field(init=False)
    km_linear_zero: float = field(init=False)

    def __post_init__(self, event_times, event_indicators):
        super().__post_init__(event_times, event_indicators)
        area_probabilities = np.append(1, self.survival_probabilities)
        area_times = np.append(0, self.survival_times)
        if self.survival_probabilities[-1] != 0:
            slope = (area_probabilities[-1] - 1) / area_times[-1]
            zero_survival = -1 / slope
            area_times = np.append(area_times, zero_survival)
            area_probabilities = np.append(area_probabilities, 0)

        area_diff = np.diff(area_times, 1)
        area = np.flip(np.flip(area_diff * area_probabilities[0:-1]).cumsum())

        self.area_times = np.append(area_times, np.inf)
        self.area_probabilities = area_probabilities
        self.area = np.append(area, 0)
        self.km_linear_zero = -1 / ((1 - min(self.survival_probabilities))/(0 - max(self.survival_times)))

    def best_guess(self, censor_times: np.array):
        surv_prob = self.predict(censor_times)
        censor_indexes = np.digitize(censor_times, self.area_times)
        censor_indexes = np.where(
            censor_indexes == self.area_times.size + 1,
            censor_indexes - 1,
            censor_indexes,
        )
        censor_area = (self.area_times[censor_indexes] - censor_times
                      ) * self.area_probabilities[censor_indexes - 1]
        censor_area += self.area[censor_indexes]
        return censor_times + censor_area / surv_prob

    def _km_linear_predict(self, times):
        slope = (1 - min(self.survival_probabilities)) / (0 - max(self.survival_times))

        predict_prob = np.empty_like(times)
        before_last_time_idx = times <= max(self.survival_times)
        after_last_time_idx = times > max(self.survival_times)
        predict_prob[before_last_time_idx] = self.predict(times[before_last_time_idx])
        predict_prob[after_last_time_idx] = np.clip(1 + times[after_last_time_idx] * slope, a_min=0, a_max=None)
        # if time <= max(self.survival_times):
        #     predict_prob = self.predict(time)
        # else:
        #     predict_prob = max(1 + time * slope, 0)
        return predict_prob

    def _compute_best_guess(self, time: float):
        """
        Given a censor time, compute the decensor event time based on the residual mean survival time on KM curves.
        :param time:
        :return:
        """
        # Using integrate.quad from Scipy should be more accurate, but also making the program unbearably slow.
        # The compromised method uses numpy.trapz to approximate the integral using composite trapezoidal rule.
        time_range = np.linspace(time, self.km_linear_zero, 2000)
        best_guess = time + np.trapz(self._km_linear_predict(time_range), time_range) / self.predict(time)
        # best_guess = time + integrate.quad(self._km_linear_predict, time, self.km_linear_zero,
        #                                    limit=2000)[0] / self.predict(time)
        return best_guess

    def best_guess_revise(self, censor_times: np.array):
        bg_times = np.zeros_like(censor_times)
        for i in range(len(censor_times)):
            bg_times[i] = self._compute_best_guess(censor_times[i])
        return bg_times
