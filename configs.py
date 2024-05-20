configs={
    "test_batchsize":50,
    "train_batchsize":50,
    "test_dataset_size":2000,
    "train_dataset_size":5000,
    "prior_mu":0,
    "prior_sigma":0.05,
    "snn_initialize_rounds":3,
    "head_finetune_epoch":5,
    "train_epoch":20,
    "kl_weight":1,
    "head_ft_learn_rate":0.1,
    "learn_rate":0.1,
    "grad_mc_times":10,
    "monte_carlo_times":100, # This is the test-time mc.
    "gpu":False,
    "gaussian_aggregation":"mm",
}