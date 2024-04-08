import numpy as np
import pandas as pd
from typing import Optional
import scipy.integrate as integrate

from utility.survival import check_and_convert, predict_mean_survival_time, predict_median_survival_time
from utility.survival import KaplanMeier, KaplanMeierArea

def l1_loss(
        predicted_times: np.ndarray,
        event_times: np.ndarray,
        event_indicators: np.ndarray,
        train_event_times: Optional[np.ndarray] = None,
        train_event_indicators: Optional[np.ndarray] = None,
        method: str = "Hinge",
        log_scale: bool = False,
        base_age: np.ndarray = None
) -> float:

    event_indicators = event_indicators.astype(bool)
    if train_event_indicators is not None:
        train_event_indicators = train_event_indicators.astype(bool)

    if method == "Uncensored":
        if log_scale:
            scores = np.log(event_times[event_indicators]) - np.log(predicted_times[event_indicators])
        else:
            scores = event_times[event_indicators] - predicted_times[event_indicators]
        return np.mean(np.abs(scores))
    elif method == "Hinge":
        if log_scale:
            scores = np.log(event_times) - np.log(predicted_times)
        else:
            scores = event_times - predicted_times
        scores[~event_indicators] = np.maximum(scores[~event_indicators], 0)
        return np.mean(np.abs(scores))
    elif method == "Margin":
        if train_event_times is None or train_event_indicators is None:
            error = "If 'Margin' is chosen, training set values must be included."
            raise ValueError(error)

        # Calculate the best guess survival time given the KM curve and censoring time of that patient
        # Each best guess value has a confidence weight = 1 - KM(censoring time).
        # The earlier the patient got censored, the lower the confident weight is.
        km_model = KaplanMeierArea(train_event_times, train_event_indicators)
        km_linear_zero = -1 / ((1 - min(km_model.survival_probabilities))/(0 - max(km_model.survival_times)))
        if np.isinf(km_linear_zero):
            km_linear_zero = max(km_model.survival_times)
        predicted_times = np.clip(predicted_times, a_max=km_linear_zero, a_min=None)

        def _km_linear_predict(time):
            slope = (1 - min(km_model.survival_probabilities)) / (0 - max(km_model.survival_times))

            # predict_prob = np.empty_like(time)
            # before_last_time_idx = time <= max(km_model.survival_times)
            # after_last_time_idx = time > max(km_model.survival_times)
            # predict_prob[before_last_time_idx] = km_model.predict(time[before_last_time_idx])
            # predict_prob[after_last_time_idx] = np.clip(1 + time[after_last_time_idx] * slope, a_min=0, a_max=None)
            if time <= max(km_model.survival_times):
                predict_prob = km_model.predict(time)
            else:
                predict_prob = max(1 + time * slope, 0)
            return predict_prob

        def _compute_best_guess(time):
            return time + integrate.quad(_km_linear_predict, time, km_linear_zero,
                                         limit=2000)[0] / km_model.predict(time)

        censor_times = event_times[~event_indicators]
        weights = 1 - km_model.predict(censor_times)
        best_guesses = km_model.best_guess_revise(censor_times)
        best_guesses[censor_times > km_linear_zero] = censor_times[censor_times > km_linear_zero]

        scores = np.empty(predicted_times.size)
        if log_scale:
            scores[event_indicators] = np.log(event_times[event_indicators]) - np.log(predicted_times[event_indicators])
            scores[~event_indicators] = weights * (np.log(best_guesses) - np.log(predicted_times[~event_indicators]))
        else:
            scores[event_indicators] = event_times[event_indicators] - predicted_times[event_indicators]
            scores[~event_indicators] = weights * (best_guesses - predicted_times[~event_indicators])
        weighted_multiplier = 1 / (np.sum(event_indicators) + np.sum(weights))
        return weighted_multiplier * np.sum(np.abs(scores))
    elif method == "Margin_bound":
        if train_event_times is None or train_event_indicators is None or base_age is None:
            error = "If 'margin' is chosen, training set values or baseline age must be included."
            raise ValueError(error)

        # Calculate the best guess survival time given the KM curve and censoring time of that patient
        # Each best guess value has a confidence weight = 1 - KM(censoring time).
        # The earlier the patient got censored, the lower the confident weight is.
        km_model = KaplanMeierArea(train_event_times, train_event_indicators)
        avg_age = base_age.mean()
        km_linear_zero = (120 - avg_age) * 12
        km_model.survival_times = np.append(km_model.survival_times, km_linear_zero)
        km_model.survival_probabilities = np.append(km_model.survival_probabilities, 0)

        censor_times = event_times[~event_indicators]
        weights = 1 - km_model.predict(censor_times)
        best_guesses = km_model.best_guess_revise(censor_times)
        best_guesses[censor_times > km_linear_zero] = censor_times[censor_times > km_linear_zero]

        scores = np.empty(predicted_times.size)
        if log_scale:
            scores[event_indicators] = np.log(event_times[event_indicators]) - np.log(predicted_times[event_indicators])
            scores[~event_indicators] = weights * (np.log(best_guesses) - np.log(predicted_times[~event_indicators]))
        else:
            scores[event_indicators] = event_times[event_indicators] - predicted_times[event_indicators]
            scores[~event_indicators] = weights * (best_guesses - predicted_times[~event_indicators])
        weighted_multiplier = 1 / (np.sum(event_indicators) + np.sum(weights))
        return weighted_multiplier * np.sum(np.abs(scores))
    else:
        error = """Please enter one of 'Uncensored', 'Hinge', or 'Margin' for L1 loss type."""
        raise TypeError(error)