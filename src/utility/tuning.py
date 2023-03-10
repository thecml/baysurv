def get_baseline_sweep_config():
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
                "values": [[16], [16, 16], [16, 16, 16],
                           [32], [32, 32], [32, 32, 32],
                           [64], [64, 64], [64, 64, 64]]
            },
            "learning_rate": {
                "values": [0.001, 0.005, 0.01, 0.05, 0.1]
            },
            "weight_decay": {
                "values": [1e-3, 1e-4, 1e-5, None]
            },
            "momentum": {
                "values": [0.99, 0.97, 0.95, 0.9, 0.0]
            },
            "optimizer": {
                "values": ["Adam", "SGD", "RMSprop"]
            },
            "activation_fn": {
                "values": ["relu", "selu"]
            },
            "dropout": {
                "values": [None, 0.25, 0.5]
            },
            "l2_reg": {
                "values": [None, 0.001, 0.01, 0.1]
            }
        }
    }