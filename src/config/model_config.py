"""
Model configuration files for AI-morphasis neural network training.

Defines complete configurations for DQN and Policy Gradient models
with environment, training, and hyperparameter settings.
"""

# Default DQN Configuration
dqn_config = {
    "model": {
        "type": "dqn",
        "state_size": 64,
        "action_size": 10,
        "learning_rate": 0.001,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.01,
        "device": "cpu",
        "hidden_layers": [128, 64],
        "activation": "relu"
    },
    "environment": {
        "state_size": 64,
        "action_size": 10,
        "max_steps": 500,
        "reward_scale": 1.0
    },
    "training": {
        "model_type": "dqn",
        "episodes": 200,
        "batch_size": 32,
        "buffer_size": 100000,
        "update_freq": 4,
        "target_update_freq": 1000,
        "learning_rate_decay": 0.9999,
        "verbose": 10,
        "eval_episodes": 10,
        "checkpoint_dir": "checkpoints/dqn",
        "early_stopping_patience": 20
    },
    "data": {
        "normalization_type": "minmax",
        "reward_normalization": true,
        "augmentation": {
            "enabled": true,
            "noise_std": 0.05,
            "mixup_alpha": 0.2
        }
    },
    "evaluation": {
        "eval_frequency": 10,
        "eval_episodes": 10,
        "save_best_model": true,
        "save_all_checkpoints": false
    }
}

# Policy Gradient Configuration
policy_config = {
    "model": {
        "type": "policy_gradient",
        "state_size": 64,
        "action_size": 10,
        "learning_rate": 0.0005,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "device": "cpu",
        "hidden_layers": [128, 64],
        "activation": "relu",
        "action_space": "discrete",
        "entropy_coeff": 0.01
    },
    "environment": {
        "state_size": 64,
        "action_size": 10,
        "max_steps": 500,
        "reward_scale": 1.0
    },
    "training": {
        "model_type": "policy_gradient",
        "episodes": 150,
        "batch_size": 64,
        "buffer_size": 50000,
        "epochs_per_batch": 3,
        "num_rollouts": 4,
        "learning_rate_decay": 0.9999,
        "verbose": 5,
        "eval_episodes": 10,
        "checkpoint_dir": "checkpoints/policy",
        "early_stopping_patience": 15
    },
    "data": {
        "normalization_type": "zscore",
        "reward_normalization": true,
        "augmentation": {
            "enabled": true,
            "noise_std": 0.05,
            "mixup_alpha": 0.15
        }
    },
    "evaluation": {
        "eval_frequency": 10,
        "eval_episodes": 10,
        "save_best_model": true,
        "save_all_checkpoints": false
    }
}

# Small Model Configuration (for testing/quick training)
small_config = {
    "model": {
        "type": "dqn",
        "state_size": 32,
        "action_size": 5,
        "learning_rate": 0.001,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.01,
        "device": "cpu",
        "hidden_layers": [64, 32],
        "activation": "relu"
    },
    "environment": {
        "state_size": 32,
        "action_size": 5,
        "max_steps": 200,
        "reward_scale": 1.0
    },
    "training": {
        "model_type": "dqn",
        "episodes": 50,
        "batch_size": 16,
        "buffer_size": 10000,
        "update_freq": 4,
        "target_update_freq": 100,
        "learning_rate_decay": 1.0,
        "verbose": 5,
        "eval_episodes": 5,
        "checkpoint_dir": "checkpoints/small",
        "early_stopping_patience": 10
    },
    "data": {
        "normalization_type": "minmax",
        "reward_normalization": false,
        "augmentation": {
            "enabled": false,
            "noise_std": 0.0,
            "mixup_alpha": 0.0
        }
    },
    "evaluation": {
        "eval_frequency": 5,
        "eval_episodes": 5,
        "save_best_model": true,
        "save_all_checkpoints": false
    }
}

# Large Model Configuration (for production)
large_config = {
    "model": {
        "type": "dqn",
        "state_size": 256,
        "action_size": 50,
        "learning_rate": 0.0005,
        "gamma": 0.999,
        "epsilon": 1.0,
        "epsilon_decay": 0.9995,
        "epsilon_min": 0.001,
        "device": "gpu",
        "hidden_layers": [512, 256, 128],
        "activation": "relu"
    },
    "environment": {
        "state_size": 256,
        "action_size": 50,
        "max_steps": 1000,
        "reward_scale": 1.0
    },
    "training": {
        "model_type": "dqn",
        "episodes": 500,
        "batch_size": 128,
        "buffer_size": 1000000,
        "update_freq": 4,
        "target_update_freq": 5000,
        "learning_rate_decay": 0.99999,
        "verbose": 20,
        "eval_episodes": 20,
        "checkpoint_dir": "checkpoints/large",
        "early_stopping_patience": 50
    },
    "data": {
        "normalization_type": "robust",
        "reward_normalization": true,
        "augmentation": {
            "enabled": true,
            "noise_std": 0.02,
            "mixup_alpha": 0.3
        }
    },
    "evaluation": {
        "eval_frequency": 50,
        "eval_episodes": 20,
        "save_best_model": true,
        "save_all_checkpoints": false
    }
}

# Continuous Control Configuration
continuous_config = {
    "model": {
        "type": "policy_gradient",
        "state_size": 64,
        "action_size": 6,
        "learning_rate": 0.0005,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "device": "cpu",
        "hidden_layers": [256, 256],
        "activation": "relu",
        "action_space": "continuous",
        "entropy_coeff": 0.001,
        "action_scale": 1.0
    },
    "environment": {
        "state_size": 64,
        "action_size": 6,
        "max_steps": 500,
        "reward_scale": 1.0
    },
    "training": {
        "model_type": "policy_gradient",
        "episodes": 200,
        "batch_size": 64,
        "buffer_size": 100000,
        "epochs_per_batch": 5,
        "num_rollouts": 8,
        "learning_rate_decay": 0.9999,
        "verbose": 10,
        "eval_episodes": 10,
        "checkpoint_dir": "checkpoints/continuous",
        "early_stopping_patience": 30
    },
    "data": {
        "normalization_type": "zscore",
        "reward_normalization": true,
        "augmentation": {
            "enabled": true,
            "noise_std": 0.05,
            "mixup_alpha": 0.2
        }
    },
    "evaluation": {
        "eval_frequency": 20,
        "eval_episodes": 10,
        "save_best_model": true,
        "save_all_checkpoints": false
    }
}

# Multi-Agent Configuration
multi_agent_config = {
    "model": {
        "type": "dqn",
        "state_size": 128,
        "action_size": 20,
        "learning_rate": 0.0005,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.01,
        "device": "gpu",
        "hidden_layers": [256, 128, 64],
        "activation": "relu"
    },
    "environment": {
        "state_size": 128,
        "action_size": 20,
        "max_steps": 800,
        "reward_scale": 1.0,
        "num_agents": 4
    },
    "training": {
        "model_type": "dqn",
        "episodes": 300,
        "batch_size": 64,
        "buffer_size": 500000,
        "update_freq": 4,
        "target_update_freq": 2000,
        "learning_rate_decay": 0.9999,
        "verbose": 15,
        "eval_episodes": 15,
        "checkpoint_dir": "checkpoints/multi_agent",
        "early_stopping_patience": 40
    },
    "data": {
        "normalization_type": "minmax",
        "reward_normalization": true,
        "augmentation": {
            "enabled": true,
            "noise_std": 0.05,
            "mixup_alpha": 0.2
        }
    },
    "evaluation": {
        "eval_frequency": 30,
        "eval_episodes": 15,
        "save_best_model": true,
        "save_all_checkpoints": false
    }
}


CONFIG_REGISTRY = {
    "dqn": dqn_config,
    "policy": policy_config,
    "small": small_config,
    "large": large_config,
    "continuous": continuous_config,
    "multi_agent": multi_agent_config
}


def get_config(config_name: str = "dqn") -> dict:
    """
    Get configuration by name.

    Args:
        config_name: Name of configuration (dqn, policy, small, large, continuous, multi_agent)

    Returns:
        Configuration dictionary

    Raises:
        ValueError: If configuration name not found
    """
    if config_name not in CONFIG_REGISTRY:
        raise ValueError(
            f"Unknown configuration: {config_name}. "
            f"Available: {list(CONFIG_REGISTRY.keys())}"
        )
    
    return CONFIG_REGISTRY[config_name].copy()


def list_configs() -> list:
    """Get list of available configurations."""
    return list(CONFIG_REGISTRY.keys())


if __name__ == "__main__":
    import json
    
    print("Available configurations:")
    for config_name in list_configs():
        print(f"  - {config_name}")
    
    print("\nExample DQN config:")
    print(json.dumps(get_config("dqn"), indent=2))
