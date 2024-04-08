import numpy as np

def get_baymtlr_sweep_config():
    return {
        "method": "bayes",
        "metric": {
            "name": "val_ci",
            "goal": "maximize"
        },
        "parameters": {
            "hidden_size": {
                "values": [32, 64, 128]
            },
            "dropout": {
                "values": [0, 0.25, 0.5]
            },
            "num_epochs": {
                "values": [100, 500, 1000]
            },
            "early_stop": {
                "values": [False]
            },
            "patience": {
                "values": [0]
            },
            "batch_size": {
                "values": [32, 64, 128]
            }
        }
    }

def get_baycox_sweep_config():
    return {
        "method": "bayes",
        "metric": {
            "name": "val_ci",
            "goal": "maximize"
        },
        "parameters": {
            "hidden_size": {
                "values": [32, 64, 128]
            },
            "dropout": {
                "values": [0, 0.25, 0.5]
            },
            "num_epochs": {
                "values": [100, 500, 1000]
            },
            "early_stop": {
                "values": [False]
            },
            "batch_size": {
                "values": [32, 64, 128]
            },
            "patience": {
                "values": [0]
            },
        }
    }

def get_coxboost_sweep_config():
    return {
        "method": "bayes",
        "metric": {
            "name": "val_ci",
            "goal": "maximize"
        },
        "parameters": {
            "n_estimators": {
                "values": [100, 200, 400]
            },
            "learning_rate": {
                "values": [0.1, 0.5, 1.0]
            },
            "max_depth": {
                "values": [3, 5, 7]
            },
            "loss": {
                "values": ['coxph']
            },
            "min_samples_split": {
                "values": [float(x) for x in np.linspace(0.1, 0.9, 5, endpoint=True)]
            },
            "min_samples_leaf": {
                "values": [float(x) for x in np.linspace(0.1, 0.5, 5, endpoint=True)]
            },
            "max_features": {
                "values": [None, "auto", "sqrt", "log2"]
            },
            "dropout_rate": {
                "values": [float(x) for x in np.linspace(0.0, 0.9, 10, endpoint=True)]
            },
            "subsample": {
                "values": [float(x) for x in np.linspace(0.1, 1.0, 10, endpoint=True)]
            }
        }
    }

def get_mcd_sweep_config():
    return {
        "method": "bayes",
        "metric": {
            "name": "val_ci",
            "goal": "maximize"
        },
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 3
        },
        "parameters": {
            "network_layers": {
                "values": [[16], [16, 16], [32], [32, 32],
                           [64], [32, 64],
                           [128], [64, 128], [32, 64, 128]]
            },
            "n_samples_train": {
                "values": [10, 50, 100, 200]
            },
            "n_samples_valid": {
                "values": [10, 50, 100, 200]
            },
            "n_samples_test": {
                "values": [100, 200]
            },
            "learning_rate": {
                "values": [0.0005, 0.001, 0.005, 0.01]
            },
            "weight_decay": {
                "values": [1e-3, 1e-4, 1e-5, None]
            },
            "optimizer": {
                "values": ["Adam"]
            },
            "activation_fn": {
                "values": ["relu"]
            },
            "dropout": {
                "values": [0.1, 0.2, 0.25, 0.5]
            },
            "batch_size": {
                "values": [32, 64, 128]
            },
            "num_epochs": {
                "values": [100]
            },
            "l2_reg": {
                "values": [0.001]
            },
            "early_stop": {
                "values": [True]
            },
            "patience": {
                "values": [5]
            },
        }
    }

def get_mlp_sweep_config():
    return {
        "method": "bayes",
        "metric": {
            "name": "val_ci",
            "goal": "maximize"
        },
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 3
        },
        "parameters": {
            "network_layers": {
                "values": [[16], [16, 16], [32], [32, 32], [32, 32, 32],
                           [64], [32, 64], [32, 64, 64],
                           [128], [64, 128], [32, 64, 128]]
            },
            "learning_rate": {
                "values": [0.001, 0.005, 0.01]
            },
            "weight_decay": {
                "values": [1e-3, 1e-4, 1e-5, None]
            },
            "optimizer": {
                "values": ["Adam"]
            },
            "activation_fn": {
                "values": ["relu"]
            },
            "dropout": {
                "values": [0.1, 0.2, 0.25]
            },
            "batch_size": {
                "values": [32, 64, 128]
            },
            "num_epochs": {
                "values": [100]
            },
            "l2_reg": {
                "values": [0.001]
            },
            "early_stop": {
                "values": [True]
            },
            "patience": {
                "values": [10]
            },
        }
    }

def get_mcd_sweep_config():
    return {
        "method": "bayes",
        "metric": {
            "name": "val_ci",
            "goal": "maximize"
        },
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 3
        },
        "parameters": {
            "network_layers": {
                "values": [[32]]
            },
            "learning_rate": {
                "values": [0.0005, 0.001, 0.005, 0.01]
            },
            "weight_decay": {
                "values": [1e-3, 1e-4, 1e-5, None]
            },
            "optimizer": {
                "values": ["Adam"]
            },
            "activation_fn": {
                "values": ["relu"]
            },
            "dropout": {
                "values": [0.1, 0.2, 0.25, 0.5]
            },
            "batch_size": {
                "values": [32, 64, 128]
            },
            "num_epochs": {
                "values": [100]
            },
            "l2_reg": {
                "values": [0.001]
            },
            "early_stop": {
                "values": [True]
            },
            "patience": {
                "values": [10]
            },
            "n_samples_train": {
                "values": [10, 50, 100, 200]
            },
            "n_samples_valid": {
                "values": [10, 50, 100, 200]
            },
            "n_samples_test": {
                "values": [100]
            }
        }
    }

def get_rsf_sweep_config():
    return {
        "method": "bayes",
        "metric": {
            "name": "val_ci",
            "goal": "maximize"
            },
        "parameters": {
            "n_estimators": {
                "values": [100, 200, 400]
            },
            "max_depth": {
                "values": [3, 5, 7]
            },
            "min_samples_split": {
                "values": [float(x) for x in np.linspace(0.1, 0.9, 5, endpoint=True)]
            },
            "min_samples_leaf": {
                "values": [float(x) for x in np.linspace(0.1, 0.5, 5, endpoint=True)]
            },
            "max_features": {
                "values": [None, 'auto', 'sqrt', 'log2']
            },
        }
    }

def get_cox_sweep_config():
    return {
        "method": "bayes",
        "metric": {
            "name": "val_ci",
            "goal": "maximize"
        },
        "parameters": {
            "n_iter": {
                "values": [10, 50, 100]
            },
            "tol": {
                "values": [1e-1, 1e-5, 1e-9]
            }
        }
    }

def get_coxnet_sweep_config():
    return {
        "method": "bayes",
        "metric": {
            "name": "val_ci",
            "goal": "maximize"
        },
        "parameters": {
            "l1_ratio": {
                "values": [0.5, 1.0]
            },
            "alpha_min_ratio": {
                "values": [0.01, 0.1, 0.5, "auto"]
            },
            "n_alphas": {
                "values": [10, 50, 100]
            },
            "normalize": {
                "values": [True, False]
            },
            "tol": {
                "values": [1e-1, 1e-5, 1e-7]
            },
            "max_iter": {
                "values": [100000]
            }

        }
    }
    
def get_dcm_sweep_config():
    return {
        "method": "bayes",
        "metric": {
            "name": "val_ci",
            "goal": "maximize"
        },
        "parameters": {
            "network_layers": {
                "values": [[16], [16, 16], [32], [32, 32], [32, 32, 32],
                           [64], [32, 64], [32, 64, 64],
                           [128], [64, 128], [32, 64, 128]]
            },
            "n_iter": {
                "values": [50, 100, 200, 500, 1000]
            },
            "learning_rate": {
                "values": [0.001, 0.005, 0.01, 0.05, 0.1]
            }
        }
    }

def get_dsm_sweep_config():
    return {
        "method": "bayes",
        "metric": {
            "name": "val_ci",
            "goal": "maximize"
        },
        "parameters": {
            "network_layers": {
                "values": [[32], [32, 32], [32, 32, 32],
                           [64], [32, 64], [32, 64, 64],
                           [128], [64, 128], [32, 64, 128]]
            },
            "n_iter": {
                "values": [50, 100, 200, 500, 1000]
            },
            "learning_rate": {
                "values": [0.001, 0.005, 0.01, 0.05, 0.1]
            }
        }
    }
    
def get_dcph_sweep_config():
    return {
        "method": "bayes",
        "metric": {
            "name": "val_ci",
            "goal": "maximize"
        },
        "parameters": {
            "network_layers": {
                "values": [[16], [16, 16], [32], [32, 32], [32, 32, 32],
                           [64], [32, 64], [32, 64, 64],
                           [128], [64, 128], [32, 64, 128]]
            },
            "n_iter": {
                "values": [50, 100, 200, 500, 1000]
            },
            "learning_rate": {
                "values": [0.001, 0.005, 0.01, 0.05, 0.1]
            }
        }
    }