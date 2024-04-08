from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from auton_survival.estimators import SurvivalModel
from auton_survival import DeepCoxPH
from utility.bnn_isd_models import BayesCox
from utility.bnn_isd_models import BayesMtlr
from pycox.models import DeepHitSingle

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def make_baycox_model(num_features, config):
    return BayesCox(in_features=num_features, config=config)
    
def make_baymtlr_model(num_features, time_bins, config):
    num_time_bins = len(time_bins)
    return BayesMtlr(in_features=num_features, num_time_bins=num_time_bins, config=config)

def make_cox_model(config):
    n_iter = config['n_iter']
    tol = config['tol']
    return CoxPHSurvivalAnalysis(alpha=0.0001, n_iter=n_iter, tol=tol)

def make_rsf_model(config):
    n_estimators = config['n_estimators']
    max_depth = config['max_depth']
    min_samples_split = config['min_samples_split']
    min_samples_leaf =  config['min_samples_leaf']
    max_features = config['max_features']
    return RandomSurvivalForest(random_state=0,
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                max_features=max_features)

def make_coxnet_model(config):
    l1_ratio = config['l1_ratio']
    alpha_min_ratio = config['alpha_min_ratio']
    n_alphas = config['n_alphas']
    normalize = config['normalize']
    tol = config['tol']
    max_iter = config['max_iter']
    return CoxnetSurvivalAnalysis(fit_baseline_model=True,
                                  l1_ratio=l1_ratio,
                                  alpha_min_ratio=alpha_min_ratio,
                                  n_alphas=n_alphas,
                                  normalize=normalize,
                                  tol=tol,
                                  max_iter=max_iter)
    
def make_coxboost_model(config):
    n_estimators = config['n_estimators']
    learning_rate = config['learning_rate']
    max_depth = config['max_depth']
    loss = config['loss']
    min_samples_split = config['min_samples_split']
    min_samples_leaf = config['min_samples_leaf']
    max_features = config['max_features']
    dropout_rate = config['dropout_rate']
    subsample = config['subsample']
    return GradientBoostingSurvivalAnalysis(n_estimators=n_estimators,
                                            learning_rate=learning_rate,
                                            max_depth=max_depth,
                                            loss=loss,
                                            min_samples_split=min_samples_split,
                                            min_samples_leaf=min_samples_leaf,
                                            max_features=max_features,
                                            dropout_rate=dropout_rate,
                                            subsample=subsample,
                                            random_state=0)

def make_dsm_model(config):
    layers = config['network_layers']
    n_iter = config['n_iter']
    learning_rate = config['learning_rate']
    return SurvivalModel('dsm', random_seed=0, iters=n_iter,
                         layers=layers, distribution='Weibull',
                         max_features='sqrt', learning_rate=learning_rate, batch_size=32)

def make_dcph_model(config):
    layers = config['network_layers']
    n_iter = config['n_iter']
    learning_rate = config['learning_rate']
    return SurvivalModel('dcph', random_seed=0, iters=n_iter, layers=layers, learning_rate=learning_rate, batch_size=32)

def make_dcm_model(config):
    layers = config['network_layers']
    n_iter = config['n_iter']
    learning_rate = config['learning_rate']
    return SurvivalModel('dcm', random_seed=0, iters=n_iter, layers=layers, learning_rate=learning_rate, batch_size=32)