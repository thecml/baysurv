import numpy as np
import pandas as pd
import warnings
from typing import Optional, Callable
from scipy.integrate import trapezoid
from typing import Union, Tuple, List
import torch

from utility.survival import check_and_convert
from utility.survival import predict_mean_survival_time, predict_median_survival_time
from utility.survival import predict_prob_from_curve, predict_multi_probs_from_curve
from utility.evaluation import l1_loss

Numeric = Union[float, int, bool]
NumericArrayLike = Union[List[Numeric], Tuple[Numeric], np.ndarray, pd.Series, pd.DataFrame, torch.Tensor]

class BaseEvaluator:
    def __init__(
            self,
            predicted_survival_curves: NumericArrayLike,
            time_coordinates: NumericArrayLike,
            test_event_times: NumericArrayLike,
            test_event_indicators: NumericArrayLike,
            train_event_times: Optional[NumericArrayLike] = None,
            train_event_indicators: Optional[NumericArrayLike] = None
    ):
        """
        Initialize the Evaluator
        :param predicted_survival_curves: structured array, shape = (n_samples, n_time_points)
            Predicted survival curves for the testing samples.
        :param time_coordinates: structured array, shape = (n_time_points, )
            Time coordinates for the given curves.
        :param test_event_times: structured array, shape = (n_samples, )
            Actual event/censor time for the testing samples.
        :param test_event_indicators: structured array, shape = (n_samples, )
            Binary indicators of censoring for the testing samples
        :param train_event_times: structured array, shape = (n_train_samples, )
            Actual event/censor time for the training samples.
        :param train_event_indicators: structured array, shape = (n_train_samples, )
            Binary indicators of censoring for the training samples
        """
        self._predicted_curves = check_and_convert(predicted_survival_curves)
        self._time_coordinates = check_and_convert(time_coordinates)

        test_event_times, test_event_indicators = check_and_convert(test_event_times, test_event_indicators)
        self.event_times = test_event_times
        self.event_indicators = test_event_indicators

        if (train_event_times is not None) and (train_event_indicators is not None):
            train_event_times, train_event_indicators = check_and_convert(train_event_times, train_event_indicators)
        else:
            warnings.warn("Train set information is missing. Evaluator cannot perform single time Brier score, "
                          "integrated Brier score, and L1-margin loss analysis")
        self.train_event_times = train_event_times
        self.train_event_indicators = train_event_indicators

    @property
    def predicted_curves(self):
        return self._predicted_curves

    @predicted_curves.setter
    def predicted_curves(self, val):
        print("Setter called. Resetting predicted curves for this evaluator.")
        self._predicted_curves = val

    @property
    def time_coordinates(self):
        return self._time_coordinates

    @time_coordinates.setter
    def time_coordinates(self, val):
        print("Setter called. Resetting time coordinates for this evaluator.")
        self._time_coordinates = val

    def predict_time_from_curve(
            self,
            predict_method: Callable,
    ) -> np.ndarray:
        """

        :param predict_method:
        :return:
        """
        if (predict_method is not predict_mean_survival_time) and (predict_method is not predict_median_survival_time):
            error = "Prediction method must be 'predict_mean_survival_time' or 'predict_median_survival_time', " \
                    "got '{}' instead".format(predict_method.__name__)
            raise TypeError(error)

        predicted_times = []
        for i in range(self.predicted_curves.shape[0]):
            predicted_time = predict_method(self.predicted_curves[i, :], self.time_coordinates)
            predicted_times.append(predicted_time)
        predicted_times = np.array(predicted_times)
        return predicted_times

    def predict_time_from_curve_bound(
            self,
            base_age: np.ndarray,
            predict_method: Callable,
            unit: str = "Month"
    ) -> np.ndarray:
        """
        Predict time from curve, while upper bounded the survival curve with 120-year-old.
        :param base_age:
        :param predict_method:
        :param unit: Unit for the survival curves. Year, Month, or, Day
        :return:
        """
        if (predict_method is not predict_mean_survival_time) and (predict_method is not predict_median_survival_time):
            error = "Prediction method must be 'predict_mean_survival_time' or 'predict_median_survival_time', " \
                    "got '{}' instead".format(predict_method.__name__)
            raise TypeError(error)

        # if base_age.shape != self.predicted_curves.shape:
        #     error = "Baseline age must have same shape as 'self.predicted_curves'".format(predict_method.__name__)
        #     raise TypeError(error)

        if unit == "Year":
            scale_factor = 1
        elif unit == "Month":
            scale_factor = 12
        elif unit == "Day":
            scale_factor = 365
        else:
            raise ValueError("The Unit parameter is not desired type.")

        predicted_times = []
        for i in range(self.predicted_curves.shape[0]):
            predicted_time = predict_method(np.append(self.predicted_curves[i, :], 0),
                                            np.append(self.time_coordinates, (120-base_age[i]) * scale_factor))
            predicted_times.append(predicted_time)
        predicted_times = np.array(predicted_times)
        return predicted_times

    def predict_probability_from_curve(
            self,
            target_time: float
    ) -> np.ndarray:
        """

        :param target_time:
        :return:
        """
        predict_probs = []
        for i in range(self.predicted_curves.shape[0]):
            predict_prob = predict_prob_from_curve(self.predicted_curves[i, :], self.time_coordinates, target_time)
            predict_probs.append(predict_prob)
        predict_probs = np.array(predict_probs)
        return predict_probs

    def predict_multi_probabilities_from_curve(
            self,
            target_times: np.ndarray
    ) -> np.ndarray:
        """

        :param target_times:
        :return:
        """
        predict_probs_mat = []
        for i in range(self.predicted_curves.shape[0]):
            predict_probs = predict_multi_probs_from_curve(self.predicted_curves[i, :], self.time_coordinates,
                                                           target_times).tolist()
            predict_probs_mat.append(predict_probs)
        predict_probs_mat = np.array(predict_probs_mat)
        return predict_probs_mat

    def plot_survival_curves(
            self,
            curve_indices,
            color=None,
            ordered: bool=True,
            xlim=None,
            title: str=''
    ):
        raise NotImplementedError

    def l1_loss(
            self,
            method: str = "Hinge",
            log_scale: bool = False,
            predicted_time_method: str = "Median",
            base_age: np.ndarray = None
    ) -> float:
        """
        Calculate the L1 loss for the test set.
        :param method: string, default: "Hinge"
        :param log_scale: boolean, default: False
        :param predicted_time_method: string, default: "Median"
        :return:
            Value for the calculated L1 loss.
        """
        if predicted_time_method == "Median":
            predict_method = predict_median_survival_time
        elif predicted_time_method == "Mean":
            predict_method = predict_mean_survival_time
        else:
            error = "Please enter one of 'Median' or 'Mean' for calculating predicted survival time."
            raise TypeError(error)

        # get median/mean survival time from the predicted curve
        # predicted_times = self.predict_time_from_curve_bound(base_age, predict_method)
        predicted_times = self.predict_time_from_curve(predict_method)
        return l1_loss(predicted_times, self.event_times, self.event_indicators, self.train_event_times,
                       self.train_event_indicators, method, log_scale, base_age)

class PycoxEvaluator(BaseEvaluator):
    def __init__(
            self,
            surv: pd.DataFrame,
            test_event_times: NumericArrayLike,
            test_event_indicators: NumericArrayLike,
            train_event_times: Optional[NumericArrayLike] = None,
            train_event_indicators: Optional[NumericArrayLike] = None
    ):
        """

        :param surv: pd.DataFrame, shape = (n_time_points, n_samples)
            Predicted survival curves for the testing samples
            DataFrame index represents the time coordinates for the given curves.
            DataFrame value represents transpose of the survival probabilities.
        :param test_event_times:
        :param test_event_indicators:
        :param train_event_times:
        :param train_event_indicators:
        """
        time_coordinates = surv.index.values
        predicted_survival_curves = surv.values.T
        # Pycox models can sometimes obtain -0 as survival probabilities. Need to convert that to 0.
        predicted_survival_curves[predicted_survival_curves < 0] = 0
        super(PycoxEvaluator, self).__init__(predicted_survival_curves, time_coordinates, test_event_times,
                                             test_event_indicators, train_event_times, train_event_indicators)

class LifelinesEvaluator(PycoxEvaluator):
    def __init__(
            self,
            surv: pd.DataFrame,
            test_event_times: NumericArrayLike,
            test_event_indicators: NumericArrayLike,
            train_event_times: Optional[NumericArrayLike] = None,
            train_event_indicators: Optional[NumericArrayLike] = None
    ):
        super(LifelinesEvaluator, self).__init__(surv, test_event_times, test_event_indicators, train_event_times,
                                                 train_event_indicators)


class ScikitSurvivalEvaluator(BaseEvaluator):
    def __init__(
            self,
            surv: np.ndarray,
            test_event_times: NumericArrayLike,
            test_event_indicators: NumericArrayLike,
            train_event_times: Optional[NumericArrayLike] = None,
            train_event_indicators: Optional[NumericArrayLike] = None,
            with_drop=None
    ):
        """

        :param surv:
        :param test_event_times:
        :param test_event_indicators:
        :param train_event_times:
        :param train_event_indicators:
        """
        time_coordinates = surv[0].x
        predict_curves = []
        for i in range(len(surv)):
            predict_curve = surv[i].y
            if False in (time_coordinates == surv[i].x):
                raise KeyError("{}-th survival curve does not have same time coordinates".format(i))
            predict_curves.append(predict_curve)
        predicted_curves = np.array(predict_curves)
        if time_coordinates[0] != 0:
            time_coordinates = np.concatenate([np.array([0]), time_coordinates], 0)
            predicted_curves = np.concatenate([np.ones([len(predicted_curves), 1]), predicted_curves], 1)
        # If some survival curves are all ones, we should do something.
        if np.any(predicted_curves[:, len(time_coordinates) - 1] == 1):
            idx_need_fix = predicted_curves[:, len(time_coordinates) - 1] == 1
            max_prob_at_end = np.max(predicted_curves[~idx_need_fix,
                                                      len(time_coordinates) - 1])
            # max_prob_at_end + (1 - max_prob_at_end) * 0.9
            predicted_curves[idx_need_fix, len(time_coordinates) - 1] = max(0.1 * max_prob_at_end + 0.9, 0.99)
        if with_drop is not None:
            # This is a experimental test, will be deleted afterwards.
            time_coordinates = np.concatenate([time_coordinates, np.array([with_drop])], 0)
            predicted_curves = np.concatenate([predicted_curves, np.zeros([len(predicted_curves), 1])], 1)
        super(ScikitSurvivalEvaluator, self).__init__(predicted_curves, time_coordinates, test_event_times,
                                                      test_event_indicators, train_event_times, train_event_indicators)