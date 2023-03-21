# The CraftEnv Environment

CraftEnv is a flexible Multi-Agent Reinforcement Learning (MARL) environment for Collective Robotic Construction (CRC) systems, written in Python.

The CraftEnv paper is accepted by the 22nd International Conference on Autonomous Agents and Multiagent Systems (AAMAS) 2023. 

## Installation instructions

To install the codebase, please clone this repo and install the `CraftEnv/setup.py` via `pip install -e .`. The file can be used to install the necessary packages into a virtual environment. 
We use the [PyMARL](https://github.com/oxwhirl/pymarl) and the [EPyMARL](https://github.com/uoe-agents/epymarl) framework for the deep multi-agent reinforcement learning algorithms.

## Run an experiment

```shell
cd PyMARL
python src/main.py --config=qmix --env-config=multicar
```

The config files act as defaults for an algorithm or environment.

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`

Note that the `multicar` environment corresponds to the goal-conditioned tasks, the `multicar2` environment corresponds to the free building tasks, and the `flag` environment corresponds to the breaking barrier tasks.

All results will be stored in the `Results` folder.

Currently, supported algos and environments are:

- IQL, MAPPO, QMIX, QTRAN, COMA, VDN
- multicar, multicar2, goal

## Saving and loading learnt models

### Saving models

You can save the learnt models to disk by setting `save_model = True`, which is set to `False` by default. The frequency of saving models can be adjusted using `save_model_interval` configuration. Models will be saved in the result directory, under the folder called *models*. The directory corresponding each run will contain models saved throughout the experiment, each within a folder corresponding to the number of timesteps passed since starting the learning process.

### Loading models

Learnt models can be loaded using the `checkpoint_path` parameter, after which the learning will proceed from the corresponding timestep.

## Citation 
```
@inproceedings{zhao2023craftenv,  
  title={CraftEnv: A Flexible Collective Robotic Construction Environment for Multi-Agent Reinforcement Learning},  
  author={Zhao, Rui and Liu, Xu and Zhang, Yizheng and Li, Minghao and Zhou, Cheng and Li, Shuai and Han, Lei},  
  booktitle={2023 International Joint Conference on Autonomous Agents and Multi-agent Systems (AAMAS)},  
  year={2023},  
}
```

## License

Use MIT license (see LICENSE.md) except for third-party softwares. They are all open-source softwares and have their own license types.
 
## Disclaimer
 
This is not an officially supported Tencent product. The code and data in this repository are for research purpose only. No representation or warranty whatsoever, expressed or implied, is made as to its accuracy, reliability or completeness. We assume no liability and are not responsible for any misuse or damage caused by the code and data. Your use of the code and data are subject to applicable laws and your use of them is at your own risk.

