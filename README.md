# MADRL

This package provides implementations of the following multi-agent reinforcement learning environemnts:

- [Pursuit Evastion](https://github.com/sisl/MADRL/blob/master/madrl_environments/pursuit/pursuit_evade.py)
- [Waterworld](https://github.com/sisl/MADRL/blob/master/madrl_environments/pursuit/waterworld.py)
- [Multi-Agent Walker](https://github.com/sisl/MADRL/blob/master/madrl_environments/walker/multi_walker.py)

## Setup
Clone this repository.
```bash
git clone --recursive git@github.com:sisl/MADRL.git
```

Add directories to `PYTHONPATH`
```bash
export PYTHONPATH=$(pwd):$(pwd)/rltools:$(pwd)/rllab:$PYTHONPATH
```

Install the required dependencies. Good idea is to look into `rllab/environment.yml` file if using `anaconda` distribution.

## Usage

Example run with curriculum:

```bash
python3 runners/run_multiwalker.py rllab \
    --control decentralized --policy_hidden 100,50,25 \
    --n_iter 200 --n_walkers 2 --batch_size 24000 --curriculum lessons/multiwalker/env.yaml
```
