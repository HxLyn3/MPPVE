humanoid_config = {
    "target_entropy": -8,
    "plan_length": 2,
    "dynamics_hidden_dims": [400, 400, 400, 400],
    "model_update_interval": 1000,
    "model_retain_steps": 5000,
    "rollout_schedule": [20000, 300000, 1, 15],
    "n_steps": 300000
}