# MPPVE: Model-based Planning Policy Learning with Multi-step Plan Value Estimation

This is the code for the paper "Model-based Reinforcement Learning with Multi-step Plan Value Estimation".

## Installation instructions

Install Python environment with:

```bash
conda create -n mppve python=3.9 -y
conda activate mppve
conda install pytorch cudatoolkit=11.3 -c pytorch -y
pip install -r ./requirements.txt
```

## Run an experiment 

```shell
python3 main.py --env-name=[Env name] 
```

The config files located in `config` act as defaults for a task. `env-name` refers to the config files in `config/` including Hopper-v3, Walker2d-v3, Swimmer-v3, HalfCheetah-v3, AntTruncatedObs-v3 and HumanoidTruncatedObs-v3.

All results will be stored in the `result` folder.

For example, run MPPVE on Hopper:

```bash
python main.py --env-name=Hopper-v3
```


## Citation
If you find this repository useful for your research, please cite:
```
@inproceedings{
    mppve,
    title={Model-based Reinforcement Learning with Multi-step Plan Value Estimation},
    author={Haoxin Lin and Yihao Sun and Jiaji Zhang and Yang Yu},
    booktitle={Proceedings of the 26th European Conference on Artificial Intelligence (ECAI'23)},
    address={Kraków, Poland},
    year=2023
}
```