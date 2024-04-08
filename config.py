MLP_DEFAULT_PARAMS = {
    'network_layers': [32],
    'learning_rate': [0.001],
    'momentum': [0.0],
    'optimizer': ["Adam"],
    'activation_fn': ["relu"],
    'weight_decay': [None],
    'dropout': [None],
    'l2_reg': [None],
    'patience': 10,
    'n_samples_train': 10,
    'n_samples_valid': 10,
    'n_samples_test': 100
    }

MCD_DEFAULT_PARAMS = {
    'network_layers': [32],
    'learning_rate': [0.001],
    'momentum': [0.0],
    'optimizer': ["Adam"],
    'activation_fn': ["relu"],
    'weight_decay': [None],
    'dropout': [None],
    'l2_reg': [None],
    'patience': 10,
    'n_samples_train': 10,
    'n_samples_valid': 10,
    'n_samples_test': 100
    }

COX_DEFAULT_PARAMS = {
    'n_estimators': [100],
    'max_depth' : [None],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'max_features': [None],
    "seed": 0
    }

COXBOOST_DEFAULT_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 3,
    'loss': 'coxph',
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': None,
    'dropout_rate': 0.0,
    'subsample': 1.0,
    'seed': 0,
    'test_size': 0.3,
    }


COXNET_DEFAULT_PARAMS = {
    'n_estimators': [100],
    'max_depth' : [None],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'max_features': [None],
    "seed": 0
    }


DCM_DEFAULT_PARAMS = {
    'network_layers': [32, 32],
     'learning_rate': 0.001,
    'n_iters' : 100
    }


DCPH_DEFAULT_PARAMS = {
    'network_layers': [32, 32],
    'n_iters' : 100,
    'learning_rate': 0.001,
    'batch_size': 32
    }


DSM_DEFAULT_PARAMS = {
    'network_layers': [32, 32],
    'learning_rate': 0.001,
    'n_iters' : 100
    }

RSF_DEFAULT_PARAMS = {
    'n_estimators': [100],
    'max_depth' : [None],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'max_features': [None],
    "seed": 0
    }


BAYCOX_DEFAULT_PARAMS = {
    'hidden_size': 32,
    'mu_scale': None,
    'rho_scale': -5,
    'sigma1': 1,
    'sigma2': 0.002,
    'pi': 0.5,
    'verbose': False,
    'lr': 0.005,
    'num_epochs': 5000,
    'dropout': 0.6,
    'n_samples_train': 10,
    'n_samples_test': 100,
    'batch_size': 32,
    'early_stop': True,
    'patience': 50
}


BAYMTLR_DEFAULT_PARAMS = {
    'hidden_size': 32,
    'mu_scale': None,
    'rho_scale': -5,
    'sigma1': 1,
    'sigma2': 0.002,
    'pi': 0.5,
    'verbose': False,
    'lr': 0.00008,
    'num_epochs': 5000,
    'dropout': 0.6,
    'n_samples_train': 10,
    'n_samples_test': 100,
    'batch_size': 32,
    'early_stop': True,
    'patience': 50
}

