import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
from scipy.interpolate import interp1d
from typing import Union
import argparse
from utility.survival import compute_unique_counts, make_monotonic

from utility.loss import mtlr_nll, cox_nll

from utility.bnn_isd_distributions import ParametrizedGaussian, ScaleMixtureGaussian, InverseGamma
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
from datetime import datetime
from utility.survival import cox_survival, mtlr_survival

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

Numeric = Union[float, int, bool]
NumericArrayLike = Union[List[Numeric], Tuple[Numeric], np.ndarray, pd.Series, pd.DataFrame, torch.Tensor]

class CoxPH(nn.Module):
    """Cox proportional hazard model for individualised survival prediction."""

    def __init__(self, in_features: int, config: argparse.Namespace):
        super().__init__()
        if in_features < 1:
            raise ValueError("The number of input features must be at least 1")
        self.config = config
        self.in_features = in_features
        self.time_bins = None
        self.cum_baseline_hazard = None
        self.baseline_survival = None
        self.l1 = nn.Linear(self.in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.l1(x)
        return outputs

    def calculate_baseline_survival(self, x, t, e):
        outputs = self.forward(x)
        self.time_bins, self.cum_baseline_hazard, self.baseline_survival = baseline_hazard(outputs, t, e)

    def reset_parameters(self):
        self.l1.reset_parameters()
        return self

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.in_features}"

    def get_name(self):
        return self._get_name()

class mtlr(nn.Module):
    """Multi-task logistic regression for individualised
    survival prediction.

    The MTLR time-logits are computed as:
    `z = sum_k x^T w_k + b_k`,
    where `w_k` and `b_k` are learnable weights and biases for each time
    interval.

    Note that a slightly more efficient reformulation is used here, first
    proposed in [2]_.

    References
    ----------
    ..[1] C.-N. Yu et al., ‘Learning patient-specific cancer survival
    distributions as a sequence of dependent regressors’, in Advances in neural
    information processing systems 24, 2011, pp. 1845–1853.
    ..[2] P. Jin, ‘Using Survival Prediction Techniques to Learn
    Consumer-Specific Reservation Price Distributions’, Master's thesis,
    University of Alberta, Edmonton, AB, 2015.
    """

    def __init__(self, in_features: int, num_time_bins: int, config: argparse.Namespace):
        """Initialises the module.

        Parameters
        ----------
        in_features
            Number of input features.
        num_time_bins
            The number of bins to divide the time axis into.
        """
        super().__init__()
        if num_time_bins < 1:
            raise ValueError("The number of time bins must be at least 1")
        if in_features < 1:
            raise ValueError("The number of input features must be at least 1")
        self.config = config
        self.in_features = in_features
        self.num_time_bins = num_time_bins + 1  # + extra time bin [max_time, inf)

        self.mtlr_weight = nn.Parameter(torch.Tensor(self.in_features,
                                                     self.num_time_bins - 1))
        self.mtlr_bias = nn.Parameter(torch.Tensor(self.num_time_bins - 1))

        # `G` is the coding matrix from [2]_ used for fast summation.
        # When registered as buffer, it will be automatically
        # moved to the correct device and stored in saved
        # model state.
        self.register_buffer(
            "G",
            torch.tril(
                torch.ones(self.num_time_bins - 1,
                           self.num_time_bins,
                           requires_grad=True)))
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass on a batch of examples.

        Parameters
        ----------
        x : torch.Tensor, shape (num_samples, num_features)
            The input data.

        Returns
        -------
        torch.Tensor, shape (num_samples, num_time_bins - 1)
            The predicted time logits.
        """
        out = torch.matmul(x, self.mtlr_weight) + self.mtlr_bias
        return torch.matmul(out, self.G)

    def reset_parameters(self):
        """Resets the model parameters."""
        nn.init.xavier_normal_(self.mtlr_weight)
        nn.init.constant_(self.mtlr_bias, 0.)

    def __repr__(self):
        return (f"{self.__class__.__name__}(in_features={self.in_features},"
                f" num_time_bins={self.num_time_bins})")

    def get_name(self):
        return self._get_name()

class BayesianBaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def reset_parameters(self):
        pass

    @abstractmethod
    def log_prior(self):
        pass

    @abstractmethod
    def log_variational_posterior(self):
        pass

    def get_name(self):
        return self._get_name()

def make_cox_prediction(
        model: CoxPH,
        x: torch.Tensor,
        config: argparse.Namespace
):
    model.eval()
    start_time = datetime.now()
    with torch.no_grad():
        pred = model.forward(x)
        end_time = datetime.now()
        inference_time = end_time - start_time
        print(f"Inference time: {inference_time.total_seconds()}")
        survival_curves = cox_survival(model.baseline_survival, pred)
        survival_curves = survival_curves.squeeze()

    time_bins = model.time_bins
    return survival_curves, time_bins, survival_curves.unsqueeze(0).repeat(config.n_samples_test, 1, 1)

def make_mtlr_prediction(
        model: mtlr,
        x: torch.Tensor,
        time_bins: NumericArrayLike,
        config: argparse.Namespace
):
    model.eval()
    start_time = datetime.now()
    with torch.no_grad():
        pred = model.forward(x)
        end_time = datetime.now()
        inference_time = end_time - start_time
        print(f"Inference time: {inference_time.total_seconds()}")
        survival_curves = mtlr_survival(pred, with_sample=False)

    time_bins = torch.cat([torch.tensor([0]), time_bins], dim=0).to(survival_curves.device)
    return survival_curves, time_bins, survival_curves.unsqueeze(0).repeat(config.n_samples_test, 1, 1)

def make_ensemble_cox_prediction(
        model: BayesianBaseModel,
        x: torch.Tensor,
        config: dotdict
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    model.eval()
    start_time = datetime.now()
    with torch.no_grad():
        logits_outputs = model.forward(x, sample=True, n_samples=config.n_samples_test)
        end_time = datetime.now()
        inference_time = end_time - start_time
        print(f"Inference time: {inference_time.total_seconds()}")
        survival_outputs = cox_survival(model.baseline_survival, logits_outputs)
        mean_survival_outputs = survival_outputs.mean(dim=0)

    time_bins = model.time_bins
    return mean_survival_outputs, time_bins, survival_outputs

def make_ensemble_mtlr_prediction(
        model: BayesianBaseModel,
        x: torch.Tensor,
        time_bins: NumericArrayLike,
        config: dotdict
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    model.eval()
    start_time = datetime.now()

    with torch.no_grad():
        # ensemble_output should have size: n_samples * dataset_size * n_bin
        logits_outputs = model.forward(x, sample=True, n_samples=config.n_samples_test)
        end_time = datetime.now()
        inference_time = end_time - start_time
        print(f"Inference time: {inference_time.total_seconds()}")
        survival_outputs = mtlr_survival(logits_outputs, with_sample=True)
        mean_survival_outputs = survival_outputs.mean(dim=0)

    time_bins = time_bins.to(survival_outputs.device)
    return mean_survival_outputs, time_bins, survival_outputs

class BayesEleMtlr(BayesianBaseModel):
    def __init__(self, in_features: int, num_time_bins: int, config: argparse.Namespace):
        super().__init__()
        if num_time_bins < 1:
            raise ValueError("The number of time bins must be at least 1")
        self.config = config
        self.in_features = in_features
        self.hidden_size = in_features
        self.num_time_bins = num_time_bins + 1  # + extra time bin [max_time, inf)
        self.l1 = BayesianElementwiseLinear(self.in_features, config)
        self.l2 = BayesianLinear(self.in_features, self.num_time_bins - 1, config)
        self.register_buffer(
            "G",
            torch.tril(
                torch.ones(self.num_time_bins - 1,
                           self.num_time_bins,
                           requires_grad=True)))

    def forward(self, x: torch.Tensor, sample: bool, n_samples) -> torch.Tensor:
        this_batch_size = x.shape[0]    # because the last batch may not be a complete batch.
        x = F.dropout(F.relu(self.l1(x, n_samples=n_samples)), p=self.config.dropout)
        outputs = self.l2(x, sample, n_samples)
        outputs = outputs.reshape(n_samples, this_batch_size, self.num_time_bins - 1)    # this can be deleted, just for the safety

        # forward only returns (w * x + b) for computing nll loss
        # survival curves will be generated using mtlr_survival() function.
        # return outputs
        G_with_samples = self.G.expand(n_samples, -1, -1)
        # b: n_samples; i: n_data; j: n_bin - 1; k: n_bin
        return torch.einsum('bij,bjk->bik', outputs, G_with_samples)

    def log_prior(self):
        return self.l1.log_prior + self.l2.log_prior

    def log_variational_posterior(self):
        return self.l1.log_variational_posterior + self.l2.log_variational_posterior

    def sample_elbo(
            self,
            x,
            y,
            dataset_size
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        num_batch = dataset_size / self.config.batch_size
        n_samples = self.config.n_samples_train
        outputs = self(x, sample=True, n_samples=n_samples)
        log_prior = self.log_prior() / n_samples
        log_variational_posterior = self.log_variational_posterior() / n_samples
        # remark if average is needed or not
        nll = mtlr_nll(outputs.mean(dim=0), y, model=self, C1=0, average=False)
        # Shouldn't here be batch_size instead?
        loss = (log_variational_posterior - log_prior) / num_batch + nll
        return loss, log_prior, log_variational_posterior, nll

    def reset_parameters(self):
        """Reinitialize the model."""
        self.l1.reset_parameters()
        self.l2.reset_parameters()
        return self

    def __repr__(self):
        return (f"{self.__class__.__name__}(in_features={self.in_features}, "
                f"hidden_size={self.hidden_size}), "
                f"num_time_bins={self.num_time_bins})")

class BayesMtlr(BayesEleMtlr):
    def __init__(self, in_features: int, num_time_bins: int, config: argparse.Namespace):
        """Initialises the module.

        Parameters
        ----------
        in_features
            Number of input features.
        num_time_bins
            The number of bins to divide the time axis into.
        config
            Configuration/hyper-parameters of the network.
        """
        super(BayesEleMtlr, self).__init__()
        if num_time_bins < 1:
            raise ValueError("The number of time bins must be at least 1")
        self.config = config
        self.in_features = in_features
        self.hidden_size = config.hidden_size
        self.num_time_bins = num_time_bins + 1  # + extra time bin [max_time, inf)
        self.l1 = BayesianLinear(self.in_features, self.hidden_size, config)
        self.l2 = BayesianLinear(self.hidden_size, self.num_time_bins - 1, config)
        self.register_buffer(
            "G",
            torch.tril(
                torch.ones(self.num_time_bins - 1,
                           self.num_time_bins,
                           requires_grad=True)))

class BayesEleCox(BayesianBaseModel):
    def __init__(self, in_features: int, config: argparse.Namespace):
        super().__init__()
        if in_features < 1:
            raise ValueError("The number of input features must be at least 1")
        self.config = config
        self.in_features = in_features
        self.hidden_size = in_features
        self.time_bins = None
        self.cum_baseline_hazard = None
        self.baseline_survival = None
        self.l1 = BayesianElementwiseLinear(self.in_features, config)
        self.l2 = BayesianLinear(self.in_features, 1, config)

    def forward(self, x: torch.Tensor, sample: bool, n_samples) -> torch.Tensor:
        x = F.dropout(F.relu(self.l1(x, n_samples=n_samples)), p=self.config.dropout)
        outputs = self.l2(x, sample, n_samples)

        outputs = outputs.squeeze(dim=-1)
        return outputs

    def calculate_baseline_survival(self, x, t, e):
        outputs = self(x, sample=True, n_samples=self.config.n_samples_train).mean(dim=0)
        self.time_bins, self.cum_baseline_hazard, self.baseline_survival = baseline_hazard(outputs, t, e)

    def log_prior(self):
        return self.l1.log_prior + self.l2.log_prior

    def log_variational_posterior(self):
        return self.l1.log_variational_posterior + self.l2.log_variational_posterior

    def sample_elbo(
            self,
            x,
            t: torch.Tensor,
            e: torch.Tensor,
            dataset_size: int
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        num_batch = dataset_size / self.config.batch_size
        n_samples = self.config.n_samples_train
        outputs = self(x, sample=True, n_samples=n_samples)
        log_prior = self.log_prior() / n_samples
        log_variational_posterior = self.log_variational_posterior() / n_samples
        # remark if average is needed or not
        nll = cox_nll(outputs.mean(dim=0), t, e, model=self, C1=0)

        # Shouldn't here be batch_size instead?
        loss = (log_variational_posterior - log_prior) / (32 * dataset_size) + nll
        return loss, log_prior, log_variational_posterior / dataset_size, nll

    def reset_parameters(self):
        """Reinitialize the model."""
        self.l1.reset_parameters()
        self.l2.reset_parameters()
        return self

    def __repr__(self):
        return (f"{self.__class__.__name__}(in_features={self.in_features}, "
                f"hidden_size={self.hidden_size})")

class BayesianLinear(nn.Module):
    """
    Single linear layer of a mixture gaussian prior.
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            config: argparse.Namespace,
            use_mixture: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Scale to initialize weights
        self.config = config
        if self.config.mu_scale is None:
            self.weight_mu = nn.init.xavier_uniform_(nn.Parameter(torch.Tensor(out_features, in_features)))
        else:
            self.weight_mu = nn.init.uniform_(nn.Parameter(torch.Tensor(out_features, in_features)),
                                              -self.config.mu_scale, self.config.mu_scale)

        self.weight_rho = nn.Parameter(torch.ones([out_features, in_features]) * self.config.rho_scale)
        self.weight = ParametrizedGaussian(self.weight_mu, self.weight_rho)
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.zeros(1, out_features))
        self.bias_rho = nn.Parameter(torch.ones([1, out_features]) * self.config.rho_scale)
        self.bias = ParametrizedGaussian(self.bias_mu, self.bias_rho)
        # Prior distributions
        if use_mixture:
            pi = config.pi
        else:
            pi = 1
        self.weight_prior = ScaleMixtureGaussian(pi, config.sigma1, config.sigma2)
        self.bias_prior = ScaleMixtureGaussian(pi, config.sigma1, config.sigma2)

        # Initial values of the different parts of the loss function
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(
            self,
            x: torch.Tensor,
            sample: bool = True,
            n_samples: int = 1
    ):
        if self.training or sample:
            weight = self.weight.sample(n_samples=n_samples)
            bias = self.bias.sample(n_samples=n_samples)
        else:
            print("No sampling")
            weight = self.weight.mu.expand(n_samples, -1, -1)
            bias = self.bias.mu.expand(n_samples, -1, -1)

        if self.training:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        # For a single layer network, x would have 2 dimension [n_data, n_feature]
        # But sometime x would be the sampled output from the previous layer,
        # which will have 3 dimension [n_samples, n_data, n_feature]
        n_data = x.shape[-2]
        bias = bias.repeat(1, n_data, 1)
        # If x is 3-d, this expand command will make x remains the same.
        x = x.expand(n_samples, -1, -1)
        # b: n_samples; i: n_data; j: input features size; k: output size
        return torch.einsum('bij,bkj->bik', x, weight) + bias

    def reset_parameters(self):
        """Reinitialize parameters"""
        nn.init.xavier_uniform_(self.weight_mu)
        nn.init.constant_(self.weight_rho, self.config.rho_scale)
        nn.init.constant_(self.bias_mu, 0)
        nn.init.constant_(self.bias_rho, self.config.rho_scale)
        self.weight = ParametrizedGaussian(self.weight_mu, self.weight_rho)
        self.bias = ParametrizedGaussian(self.bias_mu, self.bias_rho)

class BayesianElementwiseLinear(nn.Module):
    """
    Single elementwise linear layer of a mixture gaussian prior.
    """

    def __init__(
            self,
            input_output_size: int,
            config: argparse.Namespace
    ):
        """
        Initialize gaussian layer using reparameterization.

        :param input_output_size: number of input features
        :param config: hyperparameters
        """
        super().__init__()
        self.input_output_size = input_output_size
        self.config = config
        if self.config.mu_scale is None:
            self.config.mu_scale = 1. * np.sqrt(6. / input_output_size)

        self.weight_mu = nn.init.uniform_(nn.Parameter(torch.Tensor(input_output_size)),
                                          -self.config.mu_scale, self.config.mu_scale)
        self.weight_rho = nn.Parameter(torch.ones([input_output_size]) * self.config.rho_scale)
        self.weight = ParametrizedGaussian(self.weight_mu, self.weight_rho)

        self.weight_prior = ScaleMixtureGaussian(config.pi, config.sigma1, config.sigma2)

        # Initial values of the different parts of the loss function
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(
            self,
            x: torch.tensor,
            sample: bool = True,
            n_samples: int = 1
    ):
        if self.training or sample:
            weight = self.weight.sample(n_samples=n_samples)
        else:
            print("No sampling")
            weight = self.weight.mu.expand(n_samples, -1, -1)

        if self.training:
            self.log_prior = self.weight_prior.log_prob(weight)
            self.log_variational_posterior = self.weight.log_prob(weight)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        # b: n_samples; i: n_data; j: input output size; k: input output size
        weight = torch.einsum('bj, jk->bjk', weight,
                              torch.eye(weight.shape[1], dtype=weight.dtype, device=weight.device))
        x = x.expand(n_samples, -1, -1)
        return torch.einsum('bij,bjk->bik', x, weight)

    def reset_parameters(self):
        """Reinitialize parameters"""
        nn.init.uniform_(self.weight_mu, -self.config.mu_scale, self.config.mu_scale)
        nn.init.constant_(self.weight_rho, self.config.rho_scale)
        self.weight = ParametrizedGaussian(self.weight_mu, self.weight_rho)

class BayesCox(BayesEleCox):
    def __init__(self, in_features: int, config: argparse.Namespace):
        super(BayesEleCox, self).__init__()
        if in_features < 1:
            raise ValueError("The number of input features must be at least 1")
        self.config = config
        self.in_features = in_features
        self.hidden_size = config.hidden_size
        self.time_bins = None
        self.cum_baseline_hazard = None
        self.baseline_survival = None
        self.l1 = BayesianLinear(self.in_features, self.hidden_size, config)
        self.l2 = BayesianLinear(self.hidden_size, 1, config)
        
def baseline_hazard(
        logits: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Calculate the baseline cumulative hazard function and baseline survival function using Breslow estimator
    :param logits: logit outputs calculated from the Cox-based network using training data.
    :param time: Survival time of training data.
    :param event: Survival indicator of training data.
    :return:
    uniq_times: time bins correspond of the baseline hazard/survival.
    cum_baseline_hazard: cumulative baseline hazard
    baseline_survival: baseline survival curve.
    """
    risk_score = torch.exp(logits)
    order = torch.argsort(time)
    risk_score = risk_score[order]
    uniq_times, n_events, n_at_risk, _ = compute_unique_counts(event, time, order)

    divisor = torch.empty(n_at_risk.shape, dtype=torch.float, device=n_at_risk.device)
    value = torch.sum(risk_score)
    divisor[0] = value
    k = 0
    for i in range(1, len(n_at_risk)):
        d = n_at_risk[i - 1] - n_at_risk[i]
        value -= risk_score[k:(k + d)].sum()
        k += d
        divisor[i] = value

    assert k == n_at_risk[0] - n_at_risk[-1]

    hazard = n_events / divisor
    # Make sure the survival curve always starts at 1
    if 0 not in uniq_times:
        uniq_times = torch.cat([torch.tensor([0]).to(uniq_times.device), uniq_times], 0)
        hazard = torch.cat([torch.tensor([0]).to(hazard.device), hazard], 0)
    # TODO: torch.cumsum with cuda array will generate a non-monotonic array. Need to update when torch fix this bug
    # See issue: https://github.com/pytorch/pytorch/issues/21780
    cum_baseline_hazard = torch.cumsum(hazard.cpu(), dim=0).to(hazard.device)
    baseline_survival = torch.exp(- cum_baseline_hazard)
    if baseline_survival.isinf().any():
        print(f"Baseline survival contains \'inf\', need attention. \n"
              f"Baseline survival distribution: {baseline_survival}")
        last_zero = torch.where(baseline_survival == 0)[0][-1].item()
        baseline_survival[last_zero + 1:] = 0
    baseline_survival = make_monotonic(baseline_survival)
    return uniq_times, cum_baseline_hazard, baseline_survival

class BayesLinCox(BayesianBaseModel):
    def __init__(self, in_features: int, config: argparse.Namespace):
        super().__init__()
        if in_features < 1:
            raise ValueError("The number of input features must be at least 1")
        self.config = config
        self.in_features = in_features
        self.time_bins = None
        self.cum_baseline_hazard = None
        self.baseline_survival = None

        self.l1 = BayesianLinear(self.in_features, 1, config)

    def forward(self, x: torch.Tensor, sample: bool, n_samples) -> torch.Tensor:
        outputs = self.l1(x, sample, n_samples)
        return outputs

    def calculate_baseline_survival(self, x, t, e):
        outputs = self.forward(x, sample=True, n_samples=self.config.n_samples_train).mean(dim=0)
        self.time_bins, self.cum_baseline_hazard, self.baseline_survival = baseline_hazard(outputs, t, e)

    def log_prior(self):
        return self.l1.log_prior

    def log_variational_posterior(self):
        return self.l1.log_variational_posterior

    def sample_elbo(self, x, t, e, dataset_size) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        n_samples = self.config.n_samples_train
        outputs = self(x, sample=True, n_samples=n_samples)
        log_prior = self.log_prior() / n_samples
        log_variational_posterior = self.log_variational_posterior() / n_samples
        # remark if average is needed or not
        nll = cox_nll(outputs.mean(dim=0), t, e, model=self, C1=0)

        # Shouldn't here be batch_size instead?
        loss = (log_variational_posterior - log_prior) / (dataset_size) + nll
        return loss, log_prior, log_variational_posterior, nll

    def reset_parameters(self):
        self.l1.reset_parameters()
        return self

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.in_features}"