python run_pursuit.py --log pursuit_med_32_local.h5 --n_iter 1000 --n_evaders 50 --n_pursuers 50 --control decentralized --n_timesteps 25000 --obs_range 11 --sample_maps --sampler simple --map_file ../maps/map_pool32.npy --policy_hidden_spec MED_POLICY_ARCH --baseline_hidden_spec MED_POLICY_ARCH --sampler_workers 4 --flatten --reward_mech local --surround

